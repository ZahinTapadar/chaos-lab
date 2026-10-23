// WGSL for the simulation, brush, diagnostics and rendering passes.
//
// The three simulation passes mirror `affinity`, `flow` and `referenceStep` in
// model.ts. Keep them in step: the validation suite compares them cell by cell.

import { CLAMP_TOLERANCE, CROWD_EXP, EMPTY_GENOME, GATHER, LIMITS, MAX_RADIUS, PARAM_KEYS, SCENT_FLOOR } from './model';

/** Uniform layout: every model parameter, then width, height and the kernel tap count, padded to 16 bytes. */
export const PARAM_FLOATS = Math.ceil((PARAM_KEYS.length + 3) / 4) * 4;
const pad = Array.from({ length: PARAM_FLOATS - PARAM_KEYS.length - 3 }, (_, i) => `pad${i}: f32,`).join(' ');
const PARAMS_STRUCT = `struct Params { ${PARAM_KEYS.map((k) => `${k}: f32,`).join(' ')} width: f32, height: f32, ntaps: f32, ${pad} }`;

const f = (v: number) => (Number.isInteger(v) ? `${v}.0` : `${v}`);
const vec4lit = (v: number[]) => `vec4f(${v.map(f).join(', ')})`;

const COMMON = /* wgsl */ `
${PARAMS_STRUCT}

fn wrapc(c: vec2i, size: vec2i) -> vec2i { return (c % size + size) % size; }
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
const C120: f32 = -0.5;
const S120: f32 = 0.8660254037844386;
`;

// ------------------------------------------------------------- affinity ---
export const AFFINITY_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var curS: texture_2d<f32>;
@group(0) @binding(2) var curG: texture_2d<f32>;
@group(0) @binding(3) var outX: texture_storage_2d<rgba32float, write>;
// ring-kernel taps: (dx, dy, weight, 0), weights sum to 1
@group(0) @binding(4) var<storage, read> taps: array<vec4f>;
@group(0) @binding(5) var curC: texture_2d<f32>;

// The 8 × 8 block plus a halo as wide as the largest kernel: m and m·φ packed as
// half floats, 40² × 8 bytes = 12.8 KB of workgroup memory.
const HALO: i32 = ${MAX_RADIUS};
const TILE: i32 = 8 + 2 * HALO;
var<workgroup> tileMX: array<u32, ${(8 + 2 * MAX_RADIUS) ** 2}>;
var<workgroup> tileY: array<u32, ${(8 + 2 * MAX_RADIUS) ** 2}>;
// set when any cell in the tile carries mass; empty tiles skip the convolution
var<workgroup> occupied: atomic<u32>;

// U = G(K*m; μ, σ) + β_food R + β_hunt p − β_flee q, written with q, G and K*m
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let size = vec2i(i32(P.width), i32(P.height));
  let origin = vec2i(wg.xy * 8u) - vec2i(HALO);
  if (li == 0u) { atomicStore(&occupied, 0u); }
  workgroupBarrier();
  var any = false;
  for (var k = i32(li); k < TILE * TILE; k += 64) {
    let c = wrapc(origin + vec2i(k % TILE, k / TILE), size);
    let m = textureLoad(curS, c, 0).x;
    let phi = textureLoad(curG, c, 0).zw;
    tileMX[k] = pack2x16float(vec2f(m, m * phi.x));
    tileY[k] = pack2x16float(vec2f(m * phi.y, 0.0));
    any = any || m > 0.0;
  }
  if (any) { atomicOr(&occupied, 1u); }
  workgroupBarrier();
  let busy = atomicLoad(&occupied) != 0u;
  let c = vec2i(wg.xy * 8u + lid.xy);
  if (c.x >= size.x || c.y >= size.y) { return; }

  var Um = 0.0;
  var Phi = vec2f(0.0);
  let base = (i32(lid.y) + HALO) * TILE + i32(lid.x) + HALO;
  for (var t = 0u; busy && t < u32(P.ntaps); t++) {
    let k = taps[t];
    let j = base + i32(k.y) * TILE + i32(k.x);
    let mx = unpack2x16float(tileMX[j]);
    Um += k.z * mx.x;
    Phi += k.z * vec2f(mx.y, unpack2x16float(tileY[j]).x);
  }

  let s = textureLoad(curS, c, 0);
  let gen = textureLoad(curG, c, 0);
  var mu = gen.x;
  if (P.chaosOn > 0.5) { mu = gen.x * (1.0 + P.xi * (s.z - 0.5)); }
  let sig = max(gen.y, 1e-3);
  let z = (Um - mu) / sig;
  let g = 2.0 * exp(-0.5 * z * z) - 1.0;
  var pr = 0.0;
  var q = 0.0;
  var contact = 0.0;
  let len = length(gen.zw);
  if (len > 1e-3) {
    let u = gen.zw / len;
    // prey at θ − 120°, predator at θ + 120°
    let prey = vec2f(C120 * u.x + S120 * u.y, -S120 * u.x + C120 * u.y);
    let pred = vec2f(C120 * u.x - S120 * u.y, S120 * u.x + C120 * u.y);
    let scent = textureLoad(curC, c, 0).xyz;
    pr = max(0.0, (dot(scent.xy, prey) + 0.5 * scent.z) / 1.5);
    q = max(0.0, (dot(scent.xy, pred) + 0.5 * scent.z) / 1.5);
    contact = max(0.0, dot(Phi, pred));
  }
  let U = g + P.betaFood * s.y + P.betaHunt * log(pr + ${SCENT_FLOOR}) - P.betaFlee * log(q + ${SCENT_FLOOR});
  textureStore(outX, c, vec4f(U, contact, g, Um));
}
`;

// ----------------------------------------------------------------- flow ---
export const FLOW_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var curS: texture_2d<f32>;
@group(0) @binding(2) var curX: texture_2d<f32>;
@group(0) @binding(3) var outD: texture_storage_2d<rgba32float, write>;

fn sobel(t: texture_2d<f32>, c: vec2i, size: vec2i) -> vec2f {
  let a = textureLoad(t, wrapc(c + vec2i(-1, -1), size), 0).x;
  let b = textureLoad(t, wrapc(c + vec2i(0, -1), size), 0).x;
  let d = textureLoad(t, wrapc(c + vec2i(1, -1), size), 0).x;
  let e = textureLoad(t, wrapc(c + vec2i(-1, 0), size), 0).x;
  let g = textureLoad(t, wrapc(c + vec2i(1, 0), size), 0).x;
  let h = textureLoad(t, wrapc(c + vec2i(-1, 1), size), 0).x;
  let i = textureLoad(t, wrapc(c + vec2i(0, 1), size), 0).x;
  let j = textureLoad(t, wrapc(c + vec2i(1, 1), size), 0).x;
  return vec2f((d + 2.0 * g + j - a - 2.0 * e - h) / 8.0, (h + 2.0 * i + j - a - 2.0 * b - d) / 8.0);
}

// displacement Δt·F, F = (1 − α)∇U − α∇m, α = clip((m/θ_A)ⁿ)
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(i32(P.width), i32(P.height));
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) { return; }
  let m = textureLoad(curS, c, 0).x;
  let a = clamp(pow(m / P.thetaA, ${f(CROWD_EXP)}), 0.0, 1.0);
  let F = (1.0 - a) * sobel(curX, c, size) - a * sobel(curS, c, size);
  textureStore(outD, c, vec4f(clamp(P.dt * F, vec2f(-P.maxMove), vec2f(P.maxMove)), 0.0, 0.0));
}
`;

// ---------------------------------------------------------- reintegrate ---
export const GATHER_WGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var curS: texture_2d<f32>;
@group(0) @binding(2) var curG: texture_2d<f32>;
@group(0) @binding(3) var curX: texture_2d<f32>;
@group(0) @binding(4) var curD: texture_2d<f32>;
@group(0) @binding(5) var outS: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var outG: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>, 4>;
@group(0) @binding(8) var curC: texture_2d<f32>;
@group(0) @binding(9) var outC: texture_storage_2d<rgba32float, write>;

fn clampF(v: f32, hi: f32, change: ptr<function, f32>) -> f32 {
  if (!finite(v)) { *change += 1.0; return 0.0; }
  let c = clamp(v, 0.0, hi);
  *change += abs(c - v);
  return c;
}

// gather the parcels that land on this cell, then feed, decay, convert and diffuse
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(i32(P.width), i32(P.height));
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) { return; }
  let s = P.spread;
  let area = 4.0 * s * s;
  var m = 0.0;
  var gs = vec4f(0.0);
  for (var oy = -${GATHER}; oy <= ${GATHER}; oy++) {
    for (var ox = -${GATHER}; ox <= ${GATHER}; ox++) {
      let src = wrapc(c + vec2i(ox, oy), size);
      let ms = textureLoad(curS, src, 0).x;
      if (ms <= 0.0) { continue; }
      let d = textureLoad(curD, src, 0).xy;
      let cx = f32(ox) + 0.5 + d.x;
      let cy = f32(oy) + 0.5 + d.y;
      let wx = max(0.0, min(cx + s, 1.0) - max(cx - s, 0.0));
      let wy = max(0.0, min(cy + s, 1.0) - max(cy - s, 0.0));
      let fr = wx * wy / area;
      if (fr <= 0.0) { continue; }
      let dm = ms * fr;
      m += dm;
      gs += dm * textureLoad(curG, src, 0);
    }
  }
  let cur = textureLoad(curS, c, 0);
  var gen = textureLoad(curG, c, 0);
  if (m > 1e-9) { gen = gs / m; }

  let x = textureLoad(curX, c, 0);
  let R = cur.y;
  let lapR = textureLoad(curS, wrapc(c + vec2i(1, 0), size), 0).y + textureLoad(curS, wrapc(c + vec2i(-1, 0), size), 0).y
           + textureLoad(curS, wrapc(c + vec2i(0, 1), size), 0).y + textureLoad(curS, wrapc(c + vec2i(0, -1), size), 0).y - 4.0 * R;
  let eat = P.dt * P.feed * max(R, 0.0) * m * max(x.z, 0.0);
  let rot = P.dt * P.decay * m;
  var change = 0.0;
  let mN = clampF(m + eat - rot, ${f(LIMITS.m)}, &change);
  let RN = clampF(R + P.dt * P.DR * lapR - eat + rot, ${f(LIMITS.R)}, &change);

  // predators rotate their prey's phase towards their own (θ + 120°)
  let len = length(gen.zw);
  if (len > 1e-3 && x.y > 0.0) {
    let k = min(1.0, P.dt * P.convert * x.y);
    let goal = vec2f(C120 * gen.z - S120 * gen.w, S120 * gen.z + C120 * gen.w);
    gen = vec4f(gen.xy, gen.zw + k * (goal - gen.zw));
  }

  var H = cur.z;
  if (P.chaosOn > 0.5) {
    var mean = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx != 0 || dy != 0) { mean += textureLoad(curS, wrapc(c + vec2i(dx, dy), size), 0).z; }
      }
    }
    H = (1.0 - P.mapKappa) * P.mapR * H * (1.0 - H) + P.mapKappa * mean / 8.0;
  }
  let HN = clampF(H, 1.0, &change);
  for (var i = 0; i < 4; i++) {
    if (!finite(gen[i])) { gen[i] = ${vec4lit(EMPTY_GENOME)}[i]; change += 1.0; }
  }
  // scent: diffuse, decay, and take up this cell's emission with its (new) phase
  let sc = textureLoad(curC, c, 0).xyz;
  let lapC = textureLoad(curC, wrapc(c + vec2i(1, 0), size), 0).xyz + textureLoad(curC, wrapc(c + vec2i(-1, 0), size), 0).xyz
           + textureLoad(curC, wrapc(c + vec2i(0, 1), size), 0).xyz + textureLoad(curC, wrapc(c + vec2i(0, -1), size), 0).xyz - 4.0 * sc;
  var scN = sc + P.dt * (P.DS * lapC + P.scentEmit * mN * vec3f(gen.zw, 1.0) - P.scentDecay * sc);
  for (var i = 0; i < 3; i++) {
    if (!finite(scN[i])) { scN[i] = 0.0; change += 1.0; }
  }
  let flagged = change > ${CLAMP_TOLERANCE};
  if (flagged) { atomicAdd(&counters[0], 1u); }
  textureStore(outS, c, vec4f(mN, RN, HN, min(1.0, max(select(0.0, 1.0, flagged), cur.w * 0.9))));
  textureStore(outG, c, gen);
  textureStore(outC, c, vec4f(scN, 0.0));
}
`;

// ---------------------------------------------------------------- brush ---
export const BRUSH_WGSL = /* wgsl */ `
${COMMON}
struct Brush { cx: f32, cy: f32, radius: f32, strength: f32, tool: f32, width: f32, height: f32, seed: f32, genome: vec4f }
@group(0) @binding(0) var<uniform> br: Brush;
@group(0) @binding(1) var curS: texture_2d<f32>;
@group(0) @binding(2) var curG: texture_2d<f32>;
@group(0) @binding(3) var outS: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var outG: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var curC: texture_2d<f32>;
@group(0) @binding(6) var outC: texture_storage_2d<rgba32float, write>;

fn hash(p: vec2u, s: u32) -> f32 {
  var x = (p.x * 0x8da6b343u) ^ (p.y * 0xd8163841u) ^ (s * 0xcb1ab31fu);
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return f32((x ^ (x >> 16u)) >> 8u) / 16777216.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2f(br.width, br.height);
  if (f32(gid.x) >= size.x || f32(gid.y) >= size.y) { return; }
  let c = vec2i(gid.xy);
  var s = textureLoad(curS, c, 0);
  var g = textureLoad(curG, c, 0);
  // shortest periodic offset to the brush centre
  var d = vec2f(c) + 0.5 - vec2f(br.cx, br.cy);
  d -= round(d / size) * size;
  let r = length(d) / br.radius;
  let tool = i32(br.tool);
  if (tool <= 2) {
    // a soft blob of a new creature (same profile as model.ts blobProfile)
    var add = 0.0;
    if (r < 1.0) { add = br.strength * (1.0 - r * r) * (0.75 + 0.5 * hash(gid.xy, u32(br.seed))); }
    if (add > 0.0) {
      let tot = s.x + add;
      g = (s.x * g + add * br.genome) / tot;
      s.x = min(tot, ${f(LIMITS.m)});
    }
  } else {
    let w = select(0.0, br.strength * exp(-2.0 * r * r), r < 1.0);
    if (tool == 3) {
      s.y = min(s.y + w, ${f(LIMITS.R)});
    } else {
      s.x *= 1.0 - min(w * 2.0, 1.0);
    }
  }
  textureStore(outS, c, s);
  textureStore(outG, c, g);
  textureStore(outC, c, textureLoad(curC, c, 0));
}
`;

// ------------------------------------------------------------ reduction ---
/** Block-average to a 128² pair for the diagnostics readback: (m, R, H, clamp) and m·(μ, σ, φx, φy). */
export const REDUCE_WGSL = /* wgsl */ `
@group(0) @binding(0) var curS: texture_2d<f32>;
@group(0) @binding(1) var curG: texture_2d<f32>;
@group(0) @binding(2) var outA: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var outB: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = textureDimensions(outA);
  if (gid.x >= n.x || gid.y >= n.y) { return; }
  let fct = textureDimensions(curS).x / n.x;
  var sa = vec4f(0.0);
  var sb = vec4f(0.0);
  for (var y = 0u; y < fct; y++) {
    for (var x = 0u; x < fct; x++) {
      let c = vec2i(gid.xy * fct + vec2u(x, y));
      let s = textureLoad(curS, c, 0);
      sa += s;
      sb += s.x * textureLoad(curG, c, 0);
    }
  }
  let inv = 1.0 / f32(fct * fct);
  textureStore(outA, vec2i(gid.xy), sa * inv);
  textureStore(outB, vec2i(gid.xy), sb * inv);
}
`;

// ------------------------------------------------------------- colourise ---
export const VIEW_MODES = [
  'Bioluminescent', 'Lenia classic', 'Species phase', 'Nutrient', 'Affinity', 'Flow speed',
  'Genome μ', 'Scent', 'Chaos field', 'Numerical clamp',
] as const;

export const COLOUR_WGSL = /* wgsl */ `
${COMMON}
struct View { mode: f32, exposure: f32, contours: f32, trails: f32, nutrient: f32, thetaA: f32, pad0: f32, pad1: f32 }
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var curS: texture_2d<f32>;
@group(0) @binding(2) var curG: texture_2d<f32>;
@group(0) @binding(3) var curX: texture_2d<f32>;
@group(0) @binding(4) var curD: texture_2d<f32>;
@group(0) @binding(5) var outC: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var curC: texture_2d<f32>;

const CYAN = vec3f(0.10, 0.85, 1.00);
const MAGENTA = vec3f(1.00, 0.16, 0.78);
const AMBER = vec3f(1.00, 0.62, 0.10);
const GREEN = vec3f(0.22, 0.75, 0.30);

fn lg(x: f32, k: f32) -> f32 { return log(1.0 + k * max(x, 0.0)) / log(1.0 + k); }

fn ramp(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  return clamp(vec3f(
    0.02 + x * (2.6 - 1.7 * x),
    0.01 + x * x * (1.25 - 0.3 * x),
    0.10 + x * (1.6 - x * (3.4 - 2.0 * x))), vec3f(0.0), vec3f(1.0));
}

// the classic Lenia look: a jet colour map over a dark-navy field
fn jet(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let c = clamp(vec3f(1.5) - abs(4.0 * x - vec3f(3.0, 2.0, 1.0)), vec3f(0.0), vec3f(1.0));
  return mix(vec3f(0.03, 0.03, 0.22), c, smoothstep(0.0, 0.05, x));
}

// colour of a species phase: cyan at 0°, magenta at 120°, amber at 240°, blended between
fn phaseColour(phi: vec2f) -> vec3f {
  let len = length(phi);
  if (len < 1e-3) { return vec3f(0.5); }
  let u = phi / len;
  let w = pow(max(vec3f(u.x, C120 * u.x + S120 * u.y, C120 * u.x - S120 * u.y), vec3f(0.0)), vec3f(2.0));
  let col = (CYAN * w.x + MAGENTA * w.y + AMBER * w.z) / max(w.x + w.y + w.z, 1e-4);
  // a mixed (short) phase vector desaturates towards white
  return mix(vec3f(0.85), col, clamp(len, 0.0, 1.0));
}

fn level(c: vec2i, size: vec2i) -> f32 {
  return floor(min(textureLoad(curS, wrapc(c, size), 0).x / V.thetaA, 1.0) * 6.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(curS);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let size = vec2i(dims);
  let c = vec2i(gid.xy);
  let s = textureLoad(curS, c, 0);
  let g = textureLoad(curG, c, 0);
  let x = textureLoad(curX, c, 0);
  let m = max(s.x, 0.0);
  let k = 3.0 * V.exposure;
  let mode = i32(V.mode);
  var col = vec3f(0.006, 0.008, 0.016);
  if (mode == 0) {
    col += V.nutrient * 0.10 * GREEN * lg(s.y, 1.5);
    let body = lg(m / V.thetaA, k);
    col += phaseColour(g.zw) * body + vec3f(0.5) * pow(clamp(m / V.thetaA, 0.0, 1.0), 3.0) * V.exposure * 0.5;
  } else if (mode == 1) {
    col = jet(m / V.thetaA);
  } else if (mode == 2) {
    col = phaseColour(g.zw) * select(0.0, 1.0, m > 0.02);
  } else if (mode == 3) {
    col = ramp(s.y / 2.0);
  } else if (mode == 4) {
    // diverging: blue below zero, orange above
    col = select(vec3f(0.15, 0.35, 1.0) * -x.x, vec3f(1.0, 0.55, 0.15) * x.x, x.x > 0.0) * 0.8;
  } else if (mode == 5) {
    col = ramp(length(textureLoad(curD, c, 0).xy) * 4.0) * select(0.15, 1.0, m > 0.02);
  } else if (mode == 6) {
    col = ramp((g.x - 0.05) / 0.45) * lg(m, k);
  } else if (mode == 7) {
    let sc = textureLoad(curC, c, 0).xy;
    col = phaseColour(sc) * lg(length(sc), 4.0 * V.exposure) + phaseColour(g.zw) * 0.25 * lg(m / V.thetaA, k);
  } else if (mode == 8) {
    col = mix(vec3f(0.05, 0.15, 0.6), vec3f(1.0, 0.85, 0.3), s.z) * (0.25 + 0.75 * s.z);
  } else {
    col = vec3f(0.15) * min(m, 1.0) + vec3f(1.0, 0.12, 0.08) * s.w;
  }
  if (V.contours > 0.5) {
    let l = level(c, size);
    if (l != level(c + vec2i(1, 0), size) || l != level(c + vec2i(0, 1), size)) { col += vec3f(0.10, 0.12, 0.14); }
  }
  textureStore(outC, c, vec4f(col, 1.0));
}
`;

// ---------------------------------------------------------------- bloom ---
export const BLOOM_DOWN_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = textureDimensions(dst);
  if (gid.x >= n.x || gid.y >= n.y) { return; }
  var s = vec3f(0.0);
  for (var y = 0u; y < 4u; y++) {
    for (var x = 0u; x < 4u; x++) { s += textureLoad(src, vec2i(gid.xy * 4u + vec2u(x, y)), 0).rgb; }
  }
  s /= 16.0;
  // soft knee: only the brighter structures bloom
  let lum = max(s.r, max(s.g, s.b));
  textureStore(dst, vec2i(gid.xy), vec4f(s * smoothstep(0.25, 0.9, lum), 1.0));
}
`;

export const BLOOM_BLUR_WGSL = /* wgsl */ `
struct Dir { x: f32, y: f32, pad0: f32, pad1: f32 }
@group(0) @binding(0) var<uniform> D: Dir;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec2i(textureDimensions(dst));
  if (i32(gid.x) >= n.x || i32(gid.y) >= n.y) { return; }
  let w = array<f32, 5>(0.2270, 0.1945, 0.1216, 0.0540, 0.0162);
  let dir = vec2i(i32(D.x), i32(D.y));
  var s = textureLoad(src, vec2i(gid.xy), 0).rgb * w[0];
  for (var i = 1; i < 5; i++) {
    s += textureLoad(src, (vec2i(gid.xy) + dir * i + n) % n, 0).rgb * w[i];
    s += textureLoad(src, (vec2i(gid.xy) - dir * i + n) % n, 0).rgb * w[i];
  }
  textureStore(dst, vec2i(gid.xy), vec4f(s, 1.0));
}
`;

// ------------------------------------------------------------- composite ---
export const COMPOSITE_WGSL = /* wgsl */ `
struct Screen { cx: f32, cy: f32, zoom: f32, bloom: f32, sw: f32, sh: f32, gw: f32, gh: f32 }
@group(0) @binding(0) var<uniform> S: Screen;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var colour: texture_2d<f32>;
@group(0) @binding(3) var glow: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let xy = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return VOut(vec4f(xy[i], 0.0, 1.0));
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  let screen = vec2f(S.sw, S.sh);
  // grid coordinates under this pixel; the sampler repeats, so the torus tiles
  let g = vec2f(S.cx, S.cy) + (v.pos.xy - 0.5 * screen) / S.zoom;
  let uv = g / vec2f(S.gw, S.gh);
  var col = textureSampleLevel(colour, samp, uv, 0.0).rgb;
  col += S.bloom * textureSampleLevel(glow, samp, uv, 0.0).rgb;
  // gentle shoulder so additive overlaps saturate smoothly
  col = col / (1.0 + 0.25 * col);
  let vig = 1.0 - 0.18 * pow(length((v.pos.xy - 0.5 * screen) / screen) * 1.6, 2.0);
  // ordered dither against banding in dark gradients
  let dither = (fract(sin(dot(v.pos.xy, vec2f(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  return vec4f(col * vig + dither, 1.0);
}
`;
