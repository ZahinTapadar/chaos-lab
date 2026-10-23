// In-browser validation suite (GPU against the CPU reference and against known
// dynamics) and the quality-level benchmark.

import {
  activeDt, addCreature, DEFAULTS, dtLimit, emptyState, initialState, kernelTaps, LIMITS, MODEL_VERSION, PARAM_SPECS,
  referenceStep, rng, speciesGenome, type Params, type State,
} from './model';
import { PRESETS } from './presets';
import { ChaosProbe } from './probe';
import { Simulation } from './sim';
import { speciesOf } from './diagnostics';
import { decodeSnapshot, encodeSnapshot, type SnapshotMeta } from './snapshot';

export interface Check { name: string; pass: boolean; detail: string }

/** No feeding, decay, predation or chaos: transport only. */
const QUIET: Partial<Params> = { betaFood: 0, feed: 0, decay: 0, betaHunt: 0, betaFlee: 0, convert: 0, chaosOn: 0 };

function uniformState(n: number, s: number[], g: number[]): State {
  const st = emptyState(n, n);
  for (let i = 0; i < n * n; i++) { st.S.set(s, i * 4); st.G.set(g, i * 4); }
  return st;
}

/** Worst relative difference, worst raw difference and RMS difference. */
function compare(gpu: State, cpu: State) {
  let rel = 0, raw = 0, sq = 0, n = 0;
  for (const k of ['S', 'G', 'C'] as const) {
    for (let i = 0; i < gpu[k].length; i++) {
      const d = Math.abs(gpu[k][i] - cpu[k][i]);
      const r = d / Math.max(Math.abs(cpu[k][i]), 1e-3);
      rel = Math.max(rel, Number.isFinite(r) ? r : Infinity);
      raw = Math.max(raw, Number.isFinite(d) ? d : Infinity);
      sq += d * d; n++;
    }
  }
  return { rel, raw, rms: Math.sqrt(sq / n) };
}

/** Total creature mass, total nutrient and mass by species. */
function budget(st: State) {
  let m = 0, R = 0;
  const sp = [0, 0, 0];
  for (let i = 0; i < st.w * st.h; i++) {
    const mi = st.S[i * 4];
    m += mi; R += st.S[i * 4 + 1];
    if (mi > 0) sp[speciesOf(st.G[i * 4 + 2], st.G[i * 4 + 3])] += mi;
  }
  return { m, R, sp };
}

const fmt = (v: number, d = 3) => (Math.abs(v) >= 1e-3 || v === 0 ? v.toFixed(d) : v.toExponential(2));

export async function runValidation(device: GPUDevice, report: (c: Check) => void): Promise<Check[]> {
  const results: Check[] = [];
  const add = (c: Check) => { results.push(c); report(c); };
  const check = async (name: string, fn: () => Promise<Omit<Check, 'name'>>) => {
    try { add({ name, ...(await fn()) }); } catch (e) { add({ name, pass: false, detail: `threw: ${(e as Error).message}` }); }
  };
  const twoCreatures = (p: Params) => {
    const st = initialState(64, 64, 7, { creatures: [0, 0, 0], nutrient: 0.6 }, p);
    const rand = rng(3);
    addCreature(st, 24, 30, p.radius, 1, speciesGenome(p, 0), rand);
    addCreature(st, 40, 34, p.radius, 1, speciesGenome(p, 1), rand);
    return st;
  };

  // 1. GPU against the CPU reference: two creatures in contact, every term on
  for (const steps of [1, 10]) {
    await check(`GPU vs CPU reference, ${steps} step${steps > 1 ? 's' : ''} (64², all terms on)`, async () => {
      const params: Params = { ...DEFAULTS, chaosOn: 1, xi: 0.1 };
      let cpu = twoCreatures(params);
      const sim = new Simulation(device, 64, 64, params);
      sim.upload(cpu);
      const taps = kernelTaps(params.radius);
      for (let i = 0; i < steps; i++) cpu = referenceStep(cpu, params, taps).state;
      sim.run(steps);
      const gpu = await sim.readState();
      sim.destroy();
      const { rel, raw, rms } = compare(gpu, cpu);
      return { pass: raw < 1e-3 && rms < 1e-4,
        detail: `max |Δ| = ${fmt(raw, 6)}, RMS ${fmt(rms, 7)}, worst relative ${fmt(rel, 4)} (single precision; the affinity convolution packs m and m·φ as half floats on both sides)` };
    });
  }

  // 2. conservation: transport alone conserves mass; with feeding and decay, mass + nutrient
  await check('Transport conserves creature mass', async () => {
    const params: Params = { ...DEFAULTS, ...QUIET };
    const st = initialState(128, 128, 11, { creatures: [2, 2, 2], nutrient: 0.5, spacing: 1 }, params);
    const sim = new Simulation(device, 128, 128, params);
    sim.upload(st);
    sim.run(500);
    const a = budget(st), b = budget(await sim.readState());
    sim.destroy();
    const err = Math.abs(b.m - a.m) / a.m;
    return { pass: err < 1e-4, detail: `creature mass ${fmt(a.m, 4)} → ${fmt(b.m, 4)} after 500 steps (relative drift ${fmt(err)})` };
  });
  await check('Feeding and decay conserve mass + nutrient', async () => {
    const params: Params = { ...DEFAULTS, feed: 0.1, decay: 0.01 };
    const st = initialState(128, 128, 12, { creatures: [2, 2, 2], nutrient: 0.5, spacing: 1 }, params);
    const sim = new Simulation(device, 128, 128, params);
    sim.upload(st);
    sim.run(500);
    const a = budget(st), b = budget(await sim.readState());
    sim.destroy();
    const err = Math.abs(b.m + b.R - (a.m + a.R)) / (a.m + a.R);
    return { pass: err < 1e-4 && Math.abs(b.m - a.m) > 1,
      detail: `mass ${fmt(a.m, 1)} → ${fmt(b.m, 1)}, nutrient ${fmt(a.R, 1)} → ${fmt(b.R, 1)}, total drift ${fmt(err)}` };
  });

  // 3. uniform fields stay uniform
  await check('Uniform field stays uniform', async () => {
    const params: Params = { ...DEFAULTS, chaosOn: 0 };
    const sim = new Simulation(device, 32, 32, params);
    sim.upload(uniformState(32, [0.4, 0.5, 0.6, 0], speciesGenome(params, 1)));
    sim.run(50);
    const st = await sim.readState();
    sim.destroy();
    let spread = 0;
    for (let i = 4; i < st.S.length; i++) {
      spread = Math.max(spread, Math.abs(st.S[i] - st.S[i % 4]), Math.abs(st.G[i] - st.G[i % 4]), Math.abs(st.C[i] - st.C[i % 4]));
    }
    return { pass: spread === 0, detail: `max cell-to-cell difference after 50 steps: ${spread} (m = ${fmt(st.S[0], 4)}, R = ${fmt(st.S[1], 4)})` };
  });

  // 4. periodic boundaries: a nutrient stamp on the left edge continues on the right and diffuses symmetrically
  await check('Periodic boundaries', async () => {
    const params: Params = { ...DEFAULTS, ...QUIET, DR: 0.2 };
    const sim = new Simulation(device, 32, 32, params);
    sim.upload(uniformState(32, [0, 0, 0.5, 0], [0.16, 0.025, 0, 0]));
    sim.paint({ x: 0.5, y: 16.5, radius: 5, strength: 1, tool: 3 });
    const stamped = await sim.readState();
    sim.run(20);
    const spread = await sim.readState();
    sim.destroy();
    const v = (st: State, x: number) => st.S[(16 * 32 + ((x + 32) % 32)) * 4 + 1];
    let asym = 0;
    for (let k = 1; k <= 6; k++) asym = Math.max(asym, Math.abs(v(spread, k) - v(spread, -k)) / Math.max(v(spread, k), 1e-6));
    const wrapped = v(stamped, 31) > 0 && v(stamped, 31) === v(stamped, 1);
    return { pass: wrapped && asym < 1e-4,
      detail: `stamp at x = 0: R(31) = ${fmt(v(stamped, 31))}, R(1) = ${fmt(v(stamped, 1))}; after 20 diffusion steps left/right asymmetry ${fmt(asym)}` };
  });

  // 5. predation converts prey mass to the predator's species
  await check('Predation converts prey into predator', async () => {
    const params: Params = { ...DEFAULTS, convert: 1 };
    const st = initialState(128, 128, 1, { creatures: [0, 0, 0], nutrient: 0 }, params);
    const rand = rng(5);
    addCreature(st, 56, 64, params.radius, 1, speciesGenome(params, 0), rand);
    addCreature(st, 72, 64, params.radius, 1, speciesGenome(params, 1), rand);
    const sim = new Simulation(device, 128, 128, { ...params, feed: 0, decay: 0 });
    sim.upload(st);
    sim.run(600);
    const a = budget(st), b = budget(await sim.readState());
    sim.destroy();
    return { pass: b.sp[0] < 0.5 * a.sp[0] && b.sp[1] > a.sp[1] && Math.abs(b.m - a.m) / a.m < 1e-4,
      detail: `forager mass ${fmt(a.sp[0], 1)} → ${fmt(b.sp[0], 1)}, raider ${fmt(a.sp[1], 1)} → ${fmt(b.sp[1], 1)}, total ${fmt(a.m, 1)} → ${fmt(b.m, 1)}` };
  });

  // 6. genomes mix: two species merged without conversion leave in-between phases
  await check('Merging creatures blend their genomes', async () => {
    const params: Params = { ...DEFAULTS, ...QUIET, thetaA: 1 };
    const st = initialState(128, 128, 1, { creatures: [0, 0, 0], nutrient: 0 }, params);
    const rand = rng(6);
    addCreature(st, 60, 64, params.radius, 1, speciesGenome(params, 0), rand);
    addCreature(st, 68, 64, params.radius, 1, speciesGenome(params, 2), rand);
    const sim = new Simulation(device, 128, 128, params);
    sim.upload(st);
    sim.run(400);
    const out = await sim.readState();
    sim.destroy();
    let mixed = 0, tot = 0;
    for (let i = 0; i < 128 * 128; i++) {
      const m = out.S[i * 4];
      if (m < 0.05) continue;
      tot += m;
      const purity = Math.hypot(out.G[i * 4 + 2], out.G[i * 4 + 3]);
      if (purity < 0.9) mixed += m;
    }
    return { pass: mixed > 0.05 * tot, detail: `${fmt((100 * mixed) / tot, 1)} % of the mass carries a blended phase (|φ| < 0.9)` };
  });

  // 7. bounded and finite across presets and extreme controls
  const scan = (st: State) => {
    let bad = 0, neg = 0, over = 0;
    const lim = [LIMITS.m, LIMITS.R, 1, 1];
    for (let i = 0; i < st.w * st.h; i++) for (let c = 0; c < 4; c++) {
      const v = st.S[i * 4 + c];
      if (!Number.isFinite(v) || !Number.isFinite(st.G[i * 4 + c])) bad++;
      else if (v < 0) neg++;
      else if (v > lim[c]) over++;
    }
    return { bad, neg, over };
  };
  for (const pr of PRESETS) {
    await check(`Bounded & finite: ${pr.name}`, async () => {
      const sim = new Simulation(device, 256, 256, pr.params);
      sim.upload(initialState(256, 256, pr.seed, pr.init, pr.params));
      sim.run(400);
      const diag = await sim.readDiagnostics();
      const s = scan(await sim.readState());
      sim.destroy();
      return { pass: s.bad + s.neg + s.over === 0,
        detail: `400 steps at Δt = ${fmt(activeDt(pr.params))}: ${s.bad} non-finite, ${s.neg} negative, ${s.over} out of range; ${diag?.clamped ?? 0} cell-steps clamped` };
    });
  }
  for (const end of ['max', 'min'] as const) {
    await check(`Bounded & finite: every control at its ${end}`, async () => {
      const params = { ...DEFAULTS };
      for (const s of PARAM_SPECS) params[s.key] = end === 'max' ? s.max : s.min;
      const sim = new Simulation(device, 256, 256, params);
      sim.upload(initialState(256, 256, 5, PRESETS[1].init, params));
      sim.run(200);
      const diag = await sim.readDiagnostics();
      const s = scan(await sim.readState());
      sim.destroy();
      return { pass: s.bad + s.neg + s.over === 0,
        detail: `Δt ${fmt(activeDt(params))} (limit ${fmt(dtLimit(params))}); ${s.bad} non-finite, ${s.neg} negative, ${s.over} out of range; clamp warnings on ${diag?.clamped ?? 0} cell-steps` };
    });
  }

  // 8. timestep clamping
  await check('Timestep clamped to 0.20 / max(D_R, D_S)', async () => {
    const p: Params = { ...DEFAULTS, dt: 0.5, DR: 1 };
    const sim = new Simulation(device, 32, 32, p);
    const got = sim.dt;
    sim.destroy();
    const ok = got === 0.2 && activeDt({ ...p, DR: 0.2, DS: 0.2 }) === 0.5 && activeDt({ ...p, dt: 0.1 }) === 0.1;
    return { pass: ok, detail: `requested 0.5 with D_R = 1 → ${got}; every preset within its limit: ${PRESETS.every((pr) => activeDt(pr.params) <= dtLimit(pr.params))}` };
  });

  // 9. logistic-map regimes on the GPU (uniform field, no creatures). Coupling is off: on a uniform
  // field it turns the map into 0.8 f(H) + 0.2 H, whose damped slope suppresses the period doubling
  const mapRun = async (r: number, steps: number) => {
    const sim = new Simulation(device, 32, 32, { ...DEFAULTS, ...QUIET, chaosOn: 1, mapR: r, mapKappa: 0 });
    sim.upload(uniformState(32, [0, 0, 0.3, 0], [0.16, 0.025, 0, 0]));
    const seq: number[] = [];
    for (let i = 0; i < steps; i++) {
      sim.run(1);
      seq.push((await sim.readState()).S[2]);
    }
    sim.destroy();
    return seq;
  };
  await check('Logistic map: fixed point below the cascade (r = 2.8)', async () => {
    const seq = await mapRun(2.8, 120);
    const fp = 1 - 1 / 2.8, last = seq[seq.length - 1];
    return { pass: Math.abs(last - fp) < 2e-3, detail: `H → ${fmt(last, 4)} (analytic 1 − 1/r = ${fmt(fp, 4)})` };
  });
  await check('Logistic map: period 2 past the first bifurcation (r = 3.2)', async () => {
    const seq = (await mapRun(3.2, 120)).slice(-4);
    const pass = Math.abs(seq[0] - seq[2]) < 2e-3 && Math.abs(seq[1] - seq[3]) < 2e-3 && Math.abs(seq[0] - seq[1]) > 0.1;
    return { pass, detail: `last four values ${seq.map((v) => fmt(v, 4)).join(', ')}` };
  });
  await check('Logistic map: positive Lyapunov exponent (r = 3.9)', async () => {
    const seq = (await mapRun(3.9, 260)).slice(60);
    const lam = seq.reduce((s, h) => s + Math.log(Math.abs(3.9 * (1 - 2 * h)) + 1e-12), 0) / seq.length;
    return { pass: lam > 0.2, detail: `λ ≈ ${fmt(lam)} per iteration along the GPU trajectory (literature ≈ 0.49)` };
  });

  // 10. chaos probe
  const probeRun = async (params: Params, init: (typeof PRESETS)[number]['init'], steps: number) => {
    const probe = new ChaosProbe(device, params, 404, init);
    await probe.measure();
    let last = null;
    for (let i = 0; i < steps / 50; i++) {
      const enc = device.createCommandEncoder();
      probe.encode(enc, 50);
      device.queue.submit([enc.finish()]);
      last = await probe.measure();
    }
    probe.destroy();
    return last!;
  };
  await check('Chaos probe: λ > 0 with interacting creatures (Cyclic hunt)', async () => {
    const s = await probeRun(PRESETS[1].params, PRESETS[1].init, 1500);
    return { pass: s.lambda > 0, detail: `finite-time λ = ${fmt(s.lambda, 4)} per unit time at t = ${fmt(s.t, 1)} (separation ${fmt(s.d)}). A numerical diagnostic, not a proof.` };
  });
  await check('Chaos probe: λ < 0 when a creature cannot hold together', async () => {
    // μ far above anything the body can reach: growth is −1 everywhere, so only crowding acts and the
    // blob spreads out like a diffusing drop, smoothing the perturbation away. Scent is off: emitted
    // scent integrates the difference between the copies and would grow it linearly
    const s = await probeRun({ ...DEFAULTS, ...QUIET, mu0: 0.5, scentEmit: 0 }, { creatures: [1, 0, 0], nutrient: 0 }, 1000);
    return { pass: s.lambda < 0, detail: `finite-time λ = ${fmt(s.lambda, 4)} (separation ${fmt(s.d)}). A resting creature that holds its shape reads slightly positive instead: a tiny drift moves the copies apart.` };
  });

  // 11. snapshot round trip
  await check('Snapshot export/import round trip', async () => {
    const pr = PRESETS[1];
    const sim = new Simulation(device, 256, 256, pr.params);
    sim.upload(initialState(256, 256, pr.seed, pr.init, pr.params));
    sim.run(30);
    const before = await sim.readRaw();
    const meta: SnapshotMeta = {
      app: 'chaos-lab', modelVersion: MODEL_VERSION, width: 256, height: 256, format: 'rgba32float', rowPitch: 256 * 16,
      channels: { state: ['m', 'R', 'H', 'clamp'], genome: ['mu', 'sigma', 'phase x', 'phase y'], scent: ['scent x', 'scent y', 'scent total', '-'] },
      dtActive: sim.dt, step: sim.step, time: sim.time, seed: pr.seed, preset: pr.name, params: pr.params, palette: {}, created: 'test',
    };
    const file = encodeSnapshot(meta, before);
    const back = decodeSnapshot(file.buffer as ArrayBuffer);
    const sim2 = new Simulation(device, 256, 256, back.meta.params);
    sim2.uploadRaw(back.raw, back.meta.step, back.meta.time);
    const after = await sim2.readRaw();
    sim.destroy(); sim2.destroy();
    let diff = 0;
    for (const k of ['S', 'G', 'C'] as const) for (let i = 0; i < before[k].length; i++) if (!Object.is(before[k][i], after[k][i])) diff++;
    const metaOk = JSON.stringify(back.meta) === JSON.stringify(meta);
    return { pass: diff === 0 && metaOk, detail: `${(file.length / 1024).toFixed(0)} KiB file; ${diff} differing values; metadata ${metaOk ? 'identical' : 'differs'}` };
  });

  return results;
}

// ------------------------------------------------------------ benchmark ---
export interface BenchRow { size: number; stepsPerSecond: number; frameMs: number; memoryMiB: number }

export async function runBenchmark(device: GPUDevice, sizes: number[], report: (r: BenchRow) => void): Promise<BenchRow[]> {
  const rows: BenchRow[] = [];
  const pr = PRESETS[1];
  for (const size of sizes) {
    const sim = new Simulation(device, size, size, pr.params);
    sim.upload(initialState(size, size, pr.seed, pr.init, pr.params));
    sim.run(10);
    await device.queue.onSubmittedWorkDone();
    const frames = 30;
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) sim.run(4);
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    // colour target + two quarter-resolution bloom targets
    const renderBytes = size * size * 8 + 2 * (size / 4) ** 2 * 8;
    const row = { size, stepsPerSecond: (frames * 4) / (ms / 1000), frameMs: ms / frames, memoryMiB: (sim.memoryBytes + renderBytes) / 2 ** 20 };
    sim.destroy();
    rows.push(row);
    report(row);
  }
  return rows;
}
