// Field statistics and the radial Fourier power spectrum, computed on the CPU
// from a block-averaged readback of the GPU state (or directly from a CPU state).

export interface Stats {
  biomass: number; // mean creature mass per cell
  nutrient: number;
  total: number; // mean mass + nutrient: conserved by the model
  fractions: [number, number, number]; // mass share by nearest species phase
  meanMu: number; // mass-weighted genome μ
  mixing: number; // 1 − |Σ m φ| / Σ m |φ|-free: 0 = pure species, larger = blended phases
  chaos: number;
  variance: number; // spatial variance of mass
  shannon: number; // Shannon diversity of the three species shares (nats, max ln 3)
}

/** Nearest of the three species to a phase vector (forager 0°, raider 120°, cleanser 240°). */
export const speciesOf = (fx: number, fy: number) =>
  Math.round(((Math.atan2(fy, fx) + 2 * Math.PI) % (2 * Math.PI)) / ((2 * Math.PI) / 3)) % 3;

/** `A` = block means of (m, R, H, clamp), `B` = block means of m·(μ, σ, φx, φy), over `cells` blocks. */
export function fieldStats(A: Float32Array, B: Float32Array, cells: number): Stats {
  const sp = [0, 0, 0];
  let m = 0, m2 = 0, R = 0, H = 0, mu = 0, purity = 0;
  for (let i = 0; i < cells; i++) {
    const mi = A[4 * i];
    m += mi; m2 += mi * mi; R += A[4 * i + 1]; H += A[4 * i + 2];
    mu += B[4 * i];
    if (mi > 1e-6) {
      sp[speciesOf(B[4 * i + 2], B[4 * i + 3])] += mi;
      purity += Math.hypot(B[4 * i + 2], B[4 * i + 3]);
    }
  }
  const fr = sp.map((v) => (m > 0 ? v / m : 0)) as [number, number, number];
  const mean = m / cells;
  return {
    biomass: mean,
    nutrient: R / cells,
    total: (m + R) / cells,
    fractions: fr,
    meanMu: m > 0 ? mu / m : 0,
    mixing: m > 0 ? 1 - purity / m : 0,
    chaos: H / cells,
    variance: m2 / cells - mean * mean,
    shannon: -fr.reduce((acc, q) => acc + (q > 0 ? q * Math.log(q) : 0), 0),
  };
}

export interface Creature { species: number; x: number; y: number; mass: number }

/**
 * Find creatures: 8-connected components (periodic) of the block-averaged mass
 * above `threshold`, each labelled with the species of its summed phase.
 */
export function findCreatures(A: Float32Array, B: Float32Array, n: number, threshold = 0.03): Creature[] {
  const found: Creature[] = [];
  const seen = new Uint8Array(n * n);
  const stack: number[] = [];
  for (let start = 0; start < n * n; start++) {
    if (seen[start] || A[start * 4] < threshold) continue;
    let size = 0, mass = 0, cx = 0, sx = 0, cy = 0, sy = 0, px = 0, py = 0;
    seen[start] = 1;
    stack.push(start);
    while (stack.length) {
      const c = stack.pop()!;
      size++;
      const x = c % n, y = (c / n) | 0;
      // periodic centre of mass, so creatures straddling an edge stay whole
      const v = A[c * 4];
      mass += v; px += B[c * 4 + 2]; py += B[c * 4 + 3];
      cx += v * Math.cos((2 * Math.PI * x) / n); sx += v * Math.sin((2 * Math.PI * x) / n);
      cy += v * Math.cos((2 * Math.PI * y) / n); sy += v * Math.sin((2 * Math.PI * y) / n);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const j = ((y + dy + n) % n) * n + ((x + dx + n) % n);
        if (!seen[j] && A[j * 4] >= threshold) { seen[j] = 1; stack.push(j); }
      }
    }
    const ang = (c: number, si: number) => (((Math.atan2(si, c) / (2 * Math.PI)) * n) + n + 0.5) % n;
    if (size >= 3) found.push({ species: speciesOf(px, py), x: ang(cx, sx), y: ang(cy, sy), mass });
  }
  return found;
}

/** In-place radix-2 complex FFT. */
function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const a = i + k, b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

/**
 * Radially averaged power spectrum of a real `n` × `n` field (n a power of two),
 * mean removed. Returns power per integer wavenumber 1 … n/2.
 */
export function radialSpectrum(field: Float32Array, n: number): Float64Array {
  let mean = 0;
  for (let i = 0; i < n * n; i++) mean += field[i];
  mean /= n * n;
  const re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) re[i] = field[i] - mean;
  const rr = new Float64Array(n), ri = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { rr[x] = re[y * n + x]; ri[x] = im[y * n + x]; }
    fft(rr, ri);
    for (let x = 0; x < n; x++) { re[y * n + x] = rr[x]; im[y * n + x] = ri[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { rr[y] = re[y * n + x]; ri[y] = im[y * n + x]; }
    fft(rr, ri);
    for (let y = 0; y < n; y++) { re[y * n + x] = rr[y]; im[y * n + x] = ri[y]; }
  }
  const half = n / 2;
  const power = new Float64Array(half + 1), count = new Float64Array(half + 1);
  for (let y = 0; y < n; y++) {
    const ky = y <= half ? y : y - n;
    for (let x = 0; x < n; x++) {
      const kx = x <= half ? x : x - n;
      const k = Math.round(Math.hypot(kx, ky));
      if (k < 1 || k > half) continue;
      const i = y * n + x;
      power[k] += (re[i] * re[i] + im[i] * im[i]) / (n * n);
      count[k]++;
    }
  }
  const out = new Float64Array(half);
  for (let k = 1; k <= half; k++) out[k - 1] = count[k] ? power[k] / count[k] : 0;
  return out;
}

/** Block-average an `w` × `w` interleaved RGBA field down to `n` × `n` (w divisible by n). */
export function blockAverage(src: Float32Array, w: number, n: number): Float32Array {
  const f = w / n, out = new Float32Array(n * n * 4), inv = 1 / (f * f);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) {
    const o = (Math.floor(y / f) * n + Math.floor(x / f)) * 4, i = (y * w + x) * 4;
    for (let c = 0; c < 4; c++) out[o + c] += src[i + c] * inv;
  }
  return out;
}
