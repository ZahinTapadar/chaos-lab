// The synthetic ecology: Flow-Lenia creatures (Plantec et al. 2023) that carry
// their own genome, feed on a nutrient field and convert each other in a
// rock–paper–scissors cycle. Parameters, seeded initial fields and a CPU
// reference implementation of one step.
//
// State per cell (three RGBA32F textures):
//   S = (m, R, H, clamp)       creature mass, nutrient, coupled-map chaos state,
//                               and a decaying record of non-trivial clamps
//   G = (μ, σ, φx, φy)         the genome carried by that mass: growth centre and
//                               width, and a species phase vector φ = |φ|(cos θ, sin θ)
//   C = (ψx, ψy, ψT, 0)        scent: diffusing fields each parcel of mass emits,
//                               ∂ψ/∂t = D_S∇²ψ + ω_S m φ − λ_S ψ for the phase-carrying
//                               part and the same for the total ψT (emission ω_S m).
//                               It is how creatures sense each other beyond their own
//                               kernel. The three unit phases sum to zero, so ψ alone
//                               cannot tell a species from the common background; the
//                               scent of the species at û is (ψ·û + ½ψT)/1.5, exact for
//                               pure phases.
//
// One step has three passes:
//
//  1. Affinity. With the ring kernel K, Um = K * m and Φ = K * (m φ):
//       U = G(Um; μ_eff, σ) + β_food·R + β_hunt·ln(p + ε) − β_flee·ln(q + ε)
//     where G(u) = 2 exp(−(u − μ)²/2σ²) − 1 is Lenia's growth function, used as
//     an affinity rather than a mass source, and p, q are the scents of prey
//     and predators recovered from ψ and ψT. Sensing is logarithmic (Weber's
//     law), so a faint trail from distant prey pulls as clearly as a strong one
//     nearby: the gradient of ln p depends on direction, not distance. The prey of phase
//     θ sits at θ − 120°, its predator at θ + 120°: raiders eat foragers,
//     cleansers eat raiders, foragers eat cleansers, and every phase in between.
//  2. Flow. F = (1 − α)∇U − α∇m with α = clip((m/θ_A)², 0, 1): mass climbs the
//     affinity gradient but spreads out where it gets crowded.
//  3. Reintegration and reactions. Each cell's mass moves by Δt·F (clipped to
//     ±d) and is spread over a square of half-width s; every cell gathers the
//     overlapping fractions, so transport conserves mass exactly. The genome of
//     arriving mass is the mass-weighted mean of its sources (species mix).
//     Then: feeding moves nutrient into mass where growth is positive, decay
//     returns mass to nutrient, predators rotate their prey's phase towards
//     their own where a predator's kernel field Φ·û_pred touches them (contact),
//     scent and nutrient diffuse and the logistic lattice advances. Biomass
//     plus nutrient is conserved by every term except clamps and the brush.
//
// The CPU reference mirrors the WGSL in shaders.ts operation for operation,
// including the half-precision packing of the convolution tile.

export const MODEL_VERSION = 3;
/** largest kernel radius: the GPU stages a halo this wide in workgroup memory */
export const MAX_RADIUS = 16;
/** reintegration gather window: sources up to this many cells away (max move + spread ≤ 4) */
export const GATHER = 4;
/** crowding exponent n in α = (m/θ_A)ⁿ */
export const CROWD_EXP = 2;
/** scent floor ε in the logarithmic sensing terms */
export const SCENT_FLOOR = 1e-3;

export const LIMITS = { m: 16, R: 4 };
/** clamp changes larger than this count as non-trivial */
export const CLAMP_TOLERANCE = 1e-4;

/** Every model parameter, in the order the WGSL `Params` struct declares them. */
export const PARAM_KEYS = [
  'radius', 'thetaA', 'maxMove', 'spread',
  'betaFood', 'feed', 'decay', 'DR',
  'betaHunt', 'betaFlee', 'convert', 'scentEmit', 'scentDecay', 'DS',
  'mu0', 'sig0', 'mu1', 'sig1', 'mu2', 'sig2',
  'chaosOn', 'mapR', 'mapKappa', 'xi', 'dt',
] as const;
export type ParamKey = (typeof PARAM_KEYS)[number];
export type Params = Record<ParamKey, number>;

export interface ParamSpec { key: ParamKey; group: string; label: string; min: number; max: number; step: number; tip: string }

const P = (key: ParamKey, group: string, label: string, min: number, max: number, step: number, tip: string): ParamSpec =>
  ({ key, group, label, min, max, step, tip });

export const PARAM_SPECS: ParamSpec[] = [
  P('radius', 'Body', 'Kernel radius', 6, MAX_RADIUS, 1, 'Radius of the ring kernel in cells: the scale of every creature'),
  P('thetaA', 'Body', 'Crowding θ_A', 0.3, 4, 0.01, 'Mass density at which creatures stop condensing and spread out'),
  P('maxMove', 'Body', 'Max move per step', 0.5, 3, 0.05, 'Displacement clip d, in cells'),
  P('spread', 'Body', 'Spread s', 0.3, 1, 0.01, 'Half-width of the square each moving parcel is spread over'),
  P('betaFood', 'Feeding', 'Food attraction', 0, 2, 0.01, 'Affinity for nutrient: creatures crawl towards food'),
  P('feed', 'Feeding', 'Feeding rate', 0, 0.2, 0.001, 'Nutrient turned into mass per unit mass, nutrient and positive growth'),
  P('decay', 'Feeding', 'Metabolic decay', 0, 0.05, 0.0005, 'Mass returned to nutrient per unit mass'),
  P('DR', 'Feeding', 'Nutrient diffusion', 0, 1, 0.005, 'Diffusion of the nutrient field'),
  P('betaHunt', 'Predation', 'Hunting', 0, 5, 0.01, 'Pull up the prey-scent gradient (logarithmic: direction matters, not distance)'),
  P('betaFlee', 'Predation', 'Fleeing', 0, 5, 0.01, 'Push down the predator-scent gradient'),
  P('convert', 'Predation', 'Conversion', 0, 2, 0.01, 'Rate at which a predator in contact turns its prey’s phase towards its own'),
  P('scentEmit', 'Predation', 'Scent emission', 0, 0.5, 0.001, 'Scent released per unit mass, carrying its species phase'),
  P('scentDecay', 'Predation', 'Scent decay', 0.0001, 0.05, 0.0001, 'Scent fades at this rate; range ≈ √(D_S/λ_S) cells'),
  P('DS', 'Predation', 'Scent diffusion', 0, 1, 0.005, 'How fast scent spreads'),
  P('mu0', 'New creatures', 'Forager μ', 0.05, 0.5, 0.001, 'Growth centre given to newly placed foragers'),
  P('sig0', 'New creatures', 'Forager σ', 0.005, 0.2, 0.001, 'Growth width given to newly placed foragers'),
  P('mu1', 'New creatures', 'Raider μ', 0.05, 0.5, 0.001, 'Growth centre given to newly placed raiders'),
  P('sig1', 'New creatures', 'Raider σ', 0.005, 0.2, 0.001, 'Growth width given to newly placed raiders'),
  P('mu2', 'New creatures', 'Cleanser μ', 0.05, 0.5, 0.001, 'Growth centre given to newly placed cleansers'),
  P('sig2', 'New creatures', 'Cleanser σ', 0.005, 0.2, 0.001, 'Growth width given to newly placed cleansers'),
  P('chaosOn', 'Chaos', 'Coupled-map field', 0, 1, 1, 'Advance the logistic-map lattice and let it modulate μ'),
  P('mapR', 'Chaos', 'Map r', 3.4, 4, 0.001, 'Logistic-map parameter'),
  P('mapKappa', 'Chaos', 'Coupling κ', 0, 0.35, 0.001, 'Diffusive coupling to the 8-neighbour mean'),
  P('xi', 'Chaos', 'μ modulation ξ', 0, 0.3, 0.001, 'μ_eff = μ[1 + ξ(H − ½)]'),
  P('dt', 'Simulation', 'Requested Δt', 0.02, 0.5, 0.005, 'Clamped to 0.20 / max(D_R, D_S)'),
];

export const DEFAULTS: Params = {
  radius: 12, thetaA: 1.5, maxMove: 2, spread: 0.65,
  betaFood: 0.3, feed: 0.04, decay: 0.004, DR: 0.15,
  betaHunt: 2, betaFlee: 1, convert: 0.6, scentEmit: 0.05, scentDecay: 0.0004, DS: 0.8,
  mu0: 0.16, sig0: 0.025, mu1: 0.16, sig1: 0.025, mu2: 0.16, sig2: 0.025,
  chaosOn: 0, mapR: 3.7, mapKappa: 0.2, xi: 0.05, dt: 0.2,
};

/** species phase angles: forager 0°, raider 120°, cleanser 240° */
export const SPECIES_PHASE = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
export const speciesGenome = (p: Params, s: number): [number, number, number, number] => [
  [p.mu0, p.mu1, p.mu2][s], [p.sig0, p.sig1, p.sig2][s], Math.cos(SPECIES_PHASE[s]), Math.sin(SPECIES_PHASE[s]),
];
/** genome of empty space: no phase, so it never wins a mixture */
export const EMPTY_GENOME: [number, number, number, number] = [0.16, 0.025, 0, 0];

export const dtLimit = (p: Params) => 0.2 / Math.max(p.DR, p.DS, 1e-6);
/** The timestep actually used: the request, clamped to the diffusion stability limit. */
export const activeDt = (p: Params) => Math.min(p.dt, dtLimit(p));

// ------------------------------------------------------------- kernel ---
/**
 * Lenia's exponential ring kernel of radius R, as a list of (dx, dy, weight)
 * taps normalised to sum to 1: K(r) = exp(4 − 1/(r(1 − r))) for r = |d|/R < 1.
 */
export function kernelTaps(radius: number): Float32Array {
  const R = Math.min(MAX_RADIUS, Math.max(1, Math.round(radius)));
  const taps: number[] = [];
  let sum = 0;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const r = Math.hypot(dx, dy) / R;
      if (r <= 0 || r >= 1) continue;
      const w = Math.exp(4 - 1 / (r * (1 - r)));
      if (w < 1e-7) continue;
      taps.push(dx, dy, w, 0);
      sum += w;
    }
  }
  const out = new Float32Array(taps);
  for (let i = 2; i < out.length; i += 4) out[i] /= sum;
  return out;
}

// ------------------------------------------------------------ half floats ---
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const e8 = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (e8 === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const exp = e8 - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h++;
    return sign | h;
  }
  let h = (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
  return sign | h;
}

const HALF_TABLE = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  HALF_TABLE[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
export const fromHalf = (h: number) => HALF_TABLE[h];
export const roundHalf = (v: number) => HALF_TABLE[toHalf(v)];

export function encodeHalf(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = toHalf(src[i]);
  return out;
}
export function decodeHalf(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = HALF_TABLE[src[i]];
  return out;
}

// ---------------------------------------------------------------- state ---
export interface State { w: number; h: number; S: Float32Array; G: Float32Array; C: Float32Array }

export function emptyState(w: number, h: number): State {
  const st = { w, h, S: new Float32Array(w * h * 4), G: new Float32Array(w * h * 4), C: new Float32Array(w * h * 4) };
  for (let i = 0; i < w * h; i++) st.G.set(EMPTY_GENOME, i * 4);
  return st;
}

/** Small deterministic PRNG (mulberry32). */
export function rng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Blob profile shared with the GPU brush: a soft disc of radius 1 with a little seeded texture. */
export const blobProfile = (r: number, noise: number) => (r < 1 ? (1 - r * r) * (0.75 + 0.5 * noise) : 0);

/**
 * Add one creature: a soft blob of mass of radius `radius` cells centred at
 * (cx, cy) with the given genome. Mass already there keeps its genome in
 * proportion (the same rule as transport).
 */
export function addCreature(st: State, cx: number, cy: number, radius: number, amount: number, genome: number[], rand: () => number) {
  const R = Math.ceil(radius);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
    const x = Math.floor(cx) + dx, y = Math.floor(cy) + dy;
    const r = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
    const add = amount * blobProfile(r, rand());
    if (add <= 0) continue;
    const i = (((y % st.h) + st.h) % st.h) * st.w + (((x % st.w) + st.w) % st.w);
    const m = st.S[i * 4], tot = m + add;
    for (let c = 0; c < 4; c++) st.G[i * 4 + c] = (m * st.G[i * 4 + c] + add * genome[c]) / tot;
    st.S[i * 4] = tot;
  }
}

export interface InitSpec {
  /** creatures of each species: forager, raider, cleanser */
  creatures: [number, number, number];
  /** starting nutrient level */
  nutrient: number;
  /** creature radius in kernel radii, and peak mass */
  size?: number;
  mass?: number;
  /** creatures start this far apart at least, in kernel radii */
  spacing?: number;
  /** optional nutrient pulse: [u, v, radius as a fraction of the width, amount] */
  pulse?: [number, number, number, number];
  /** optional nutrient patches: number of seeded fertile spots */
  patches?: number;
}

/**
 * Deterministic initial field for a seed: blobs of each species at seeded
 * positions on a nutrient field, with genomes from the "new creatures" blueprints.
 */
export function initialState(w: number, h: number, seed: number, init: InitSpec, p: Params): State {
  const st = emptyState(w, h);
  const rand = rng(seed);
  for (let i = 0; i < w * h; i++) { st.S[i * 4 + 1] = init.nutrient; st.S[i * 4 + 2] = 0.2 + 0.6 * rand(); }
  const spots: [number, number, number, number][] = [];
  if (init.pulse) spots.push(init.pulse);
  for (let k = 0; k < (init.patches ?? 0); k++) spots.push([rand(), rand(), 0.04 + 0.05 * rand(), 1 + rand()]);
  for (const [pu, pv, pr, amt] of spots) {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let dx = Math.abs(x / w - pu), dy = Math.abs(y / h - pv);
      dx = Math.min(dx, 1 - dx); dy = Math.min(dy, 1 - dy);
      const i = (y * w + x) * 4 + 1;
      st.S[i] = Math.min(LIMITS.R, st.S[i] + amt * Math.exp(-(dx * dx + dy * dy) / (pr * pr)));
    }
  }
  const radius = (init.size ?? 1) * p.radius;
  const placed: [number, number][] = [];
  const minD = (init.spacing ?? 3) * p.radius * 2;
  init.creatures.forEach((n, species) => {
    for (let k = 0; k < n; k++) {
      let x = 0, y = 0;
      for (let tries = 0; tries < 60; tries++) {
        x = rand() * w; y = rand() * h;
        const ok = placed.every(([px, py]) => {
          let dx = Math.abs(px - x), dy = Math.abs(py - y);
          dx = Math.min(dx, w - dx); dy = Math.min(dy, h - dy);
          return Math.hypot(dx, dy) >= minD;
        });
        if (ok) break;
      }
      placed.push([x, y]);
      addCreature(st, x, y, radius, init.mass ?? 1, speciesGenome(p, species), rand);
    }
  });
  return st;
}

// ---------------------------------------------------- CPU reference step ---
const wrap = (v: number, n: number) => ((v % n) + n) % n;
const fr = Math.fround;

/** Pass 1: affinity U, predator pressure q and growth g for every cell. */
function affinity(st: State, p: Params, taps: Float32Array): Float32Array {
  const { w, h, S, G } = st;
  const X = new Float32Array(w * h * 4);
  // the GPU stages m and m·φ in half precision for the convolution
  const T = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    T[i * 3] = roundHalf(S[i * 4]);
    T[i * 3 + 1] = roundHalf(S[i * 4] * G[i * 4 + 2]);
    T[i * 3 + 2] = roundHalf(S[i * 4] * G[i * 4 + 3]);
  }
  const c120 = -0.5, s120 = Math.sqrt(3) / 2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let Um = 0, Px = 0, Py = 0;
    for (let t = 0; t < taps.length; t += 4) {
      const j = (wrap(y + taps[t + 1], h) * w + wrap(x + taps[t], w)) * 3, k = taps[t + 2];
      Um += k * T[j]; Px += k * T[j + 1]; Py += k * T[j + 2];
    }
    const i = (y * w + x) * 4;
    const R = S[i + 1], H = S[i + 2];
    const mu = G[i] * (p.chaosOn > 0.5 ? 1 + p.xi * (H - 0.5) : 1), sig = Math.max(G[i + 1], 1e-3);
    const z = (Um - mu) / sig;
    const g = 2 * Math.exp(-0.5 * z * z) - 1;
    const fx = G[i + 2], fy = G[i + 3], len = Math.hypot(fx, fy);
    let pr = 0, q = 0, contact = 0;
    if (len > 1e-3) {
      const ux = fx / len, uy = fy / len;
      // prey at θ − 120°, predator at θ + 120°
      const preyX = c120 * ux + s120 * uy, preyY = -s120 * ux + c120 * uy;
      const predX = c120 * ux - s120 * uy, predY = s120 * ux + c120 * uy;
      const sx = st.C[i], sy = st.C[i + 1], sT = st.C[i + 2];
      pr = Math.max(0, (sx * preyX + sy * preyY + 0.5 * sT) / 1.5);
      q = Math.max(0, (sx * predX + sy * predY + 0.5 * sT) / 1.5);
      contact = Math.max(0, Px * predX + Py * predY);
    }
    X[i] = g + p.betaFood * R + p.betaHunt * Math.log(pr + SCENT_FLOOR) - p.betaFlee * Math.log(q + SCENT_FLOOR);
    X[i + 1] = contact;
    X[i + 2] = g;
    X[i + 3] = Um;
  }
  return X;
}

/** Pass 2: displacement Δt·F per cell, clipped to ±maxMove. */
function flow(st: State, X: Float32Array, p: Params): Float32Array {
  const { w, h, S } = st;
  const D = new Float32Array(w * h * 2);
  const dt = activeDt(p);
  const at = (arr: Float32Array, stride: number, x: number, y: number) => arr[(wrap(y, h) * w + wrap(x, w)) * stride];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    // Sobel gradients, normalised to unit slope
    const sob = (arr: Float32Array) => [
      (at(arr, 4, x + 1, y - 1) + 2 * at(arr, 4, x + 1, y) + at(arr, 4, x + 1, y + 1)
        - at(arr, 4, x - 1, y - 1) - 2 * at(arr, 4, x - 1, y) - at(arr, 4, x - 1, y + 1)) / 8,
      (at(arr, 4, x - 1, y + 1) + 2 * at(arr, 4, x, y + 1) + at(arr, 4, x + 1, y + 1)
        - at(arr, 4, x - 1, y - 1) - 2 * at(arr, 4, x, y - 1) - at(arr, 4, x + 1, y - 1)) / 8,
    ];
    const gU = sob(X), gm = sob(S);
    const m = S[(y * w + x) * 4];
    const a = Math.min(1, Math.max(0, (m / p.thetaA) ** CROWD_EXP));
    const o = (y * w + x) * 2;
    D[o] = Math.min(p.maxMove, Math.max(-p.maxMove, dt * ((1 - a) * gU[0] - a * gm[0])));
    D[o + 1] = Math.min(p.maxMove, Math.max(-p.maxMove, dt * ((1 - a) * gU[1] - a * gm[1])));
  }
  return D;
}

const finiteOr = (v: number, change: { v: number }, hi: number) => {
  if (!Number.isFinite(v)) { change.v += 1; return 0; }
  const c = Math.min(hi, Math.max(0, v));
  change.v += Math.abs(c - v);
  return c;
};

/** One full step on the CPU. Returns the new state and the number of non-trivially clamped cells. */
export function referenceStep(st: State, p: Params, taps = kernelTaps(p.radius)): { state: State; clamped: number } {
  const { w, h, S, G } = st;
  const dt = activeDt(p);
  const X = affinity(st, p, taps);
  const D = flow(st, X, p);
  const next = emptyState(w, h);
  const s = p.spread, area = 4 * s * s;
  const c120 = -0.5, s120 = Math.sqrt(3) / 2;
  let clamped = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    // gather every parcel whose spread square overlaps this cell
    let m = 0;
    const gs = [0, 0, 0, 0];
    for (let oy = -GATHER; oy <= GATHER; oy++) for (let ox = -GATHER; ox <= GATHER; ox++) {
      const j = wrap(y + oy, h) * w + wrap(x + ox, w);
      const ms = S[j * 4];
      if (ms <= 0) continue;
      const cx = ox + 0.5 + D[j * 2], cy = oy + 0.5 + D[j * 2 + 1];
      const wx = Math.max(0, Math.min(cx + s, 1) - Math.max(cx - s, 0));
      const wy = Math.max(0, Math.min(cy + s, 1) - Math.max(cy - s, 0));
      const f = (wx * wy) / area;
      if (f <= 0) continue;
      const dm = ms * f;
      m += dm;
      for (let c = 0; c < 4; c++) gs[c] += dm * G[j * 4 + c];
    }
    const gen = m > 1e-9 ? gs.map((v) => v / m) : Array.from(G.subarray(i, i + 4));
    // reactions
    const R = S[i + 1], g = X[i + 2], q = X[i + 1];
    const lapR = S[(y * w + wrap(x + 1, w)) * 4 + 1] + S[(y * w + wrap(x - 1, w)) * 4 + 1]
      + S[(wrap(y + 1, h) * w + x) * 4 + 1] + S[(wrap(y - 1, h) * w + x) * 4 + 1] - 4 * R;
    const eat = dt * p.feed * Math.max(R, 0) * m * Math.max(g, 0);
    const rot = dt * p.decay * m;
    const change = { v: 0 };
    const mN = finiteOr(m + eat - rot, change, LIMITS.m);
    const RN = finiteOr(R + dt * p.DR * lapR - eat + rot, change, LIMITS.R);
    // predators rotate their prey's phase towards their own (θ + 120°)
    const len = Math.hypot(gen[2], gen[3]);
    if (len > 1e-3 && q > 0) {
      const k = Math.min(1, dt * p.convert * q);
      const tx = c120 * gen[2] - s120 * gen[3], ty = s120 * gen[2] + c120 * gen[3];
      gen[2] += k * (tx - gen[2]);
      gen[3] += k * (ty - gen[3]);
    }
    let H = S[i + 2];
    if (p.chaosOn > 0.5) {
      let mean = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (dx || dy) mean += S[(wrap(y + dy, h) * w + wrap(x + dx, w)) * 4 + 2];
      }
      H = (1 - p.mapKappa) * p.mapR * H * (1 - H) + p.mapKappa * mean / 8;
    }
    const HN = finiteOr(H, change, 1);
    for (let c = 0; c < 4; c++) if (!Number.isFinite(gen[c])) { gen[c] = EMPTY_GENOME[c]; change.v += 1; }
    // scent: diffuse, decay, and take up this cell's emission with its (new) phase
    for (let c = 0; c < 3; c++) {
      const sc = st.C[i + c];
      const lap = st.C[(y * w + wrap(x + 1, w)) * 4 + c] + st.C[(y * w + wrap(x - 1, w)) * 4 + c]
        + st.C[(wrap(y + 1, h) * w + x) * 4 + c] + st.C[(wrap(y - 1, h) * w + x) * 4 + c] - 4 * sc;
      const source = c < 2 ? gen[2 + c] : 1;
      let v = sc + dt * (p.DS * lap + p.scentEmit * mN * source - p.scentDecay * sc);
      if (!Number.isFinite(v)) { v = 0; change.v += 1; }
      next.C[i + c] = v;
    }
    const flagged = change.v > CLAMP_TOLERANCE;
    if (flagged) clamped++;
    next.S[i] = mN; next.S[i + 1] = RN; next.S[i + 2] = HN;
    next.S[i + 3] = Math.min(1, Math.max(flagged ? 1 : 0, fr(S[i + 3] * 0.9)));
    for (let c = 0; c < 4; c++) next.G[i + c] = gen[c];
  }
  return { state: next, clamped };
}
