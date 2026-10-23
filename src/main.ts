import {
  activeDt, dtLimit, initialState, MODEL_VERSION, PARAM_SPECS, speciesGenome, type ParamKey, type Params,
} from './model';
import { PRESETS } from './presets';
import { Simulation, type BrushStroke } from './sim';
import { Renderer, type Camera, type ViewSettings } from './renderer';
import { VIEW_MODES } from './shaders';
import { findCreatures, fieldStats, radialSpectrum, type Creature, type Stats } from './diagnostics';
import { ChaosProbe, type ProbeSample } from './probe';
import { decodeSnapshot, encodeSnapshot, type SnapshotMeta } from './snapshot';
import { runBenchmark, runValidation } from './validation';

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;

// ------------------------------------------------------------- WebGPU ---
async function initGPU() {
  const why = (msg: string) => { $('#why').textContent = msg; $('#unsupported').hidden = false; return null; };
  if (!('gpu' in navigator)) return why('This browser does not expose navigator.gpu.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
  if (!adapter) return why('WebGPU is present, but no suitable GPU adapter was found.');
  const device = await adapter.requestDevice().catch((e: Error) => { why(`The GPU device could not be created: ${e.message}`); return null; });
  if (!device) return null;
  device.lost.then((info) => { if (info.reason !== 'destroyed') why(`The GPU device was lost: ${info.message}`); });
  return device;
}

const device = await initGPU();
if (device) start(device);

function start(device: GPUDevice) {
  const canvas = $<HTMLCanvasElement>('#c');
  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  device.addEventListener('uncapturederror', (e) => console.error((e as GPUUncapturedErrorEvent).error.message));

  // ---------------------------------------------------------- lab state ---
  const QUALITIES = [256, 512, 768, 1024];
  const TOOLS = [
    { name: 'Forager', colour: 'var(--forager)' }, { name: 'Raider', colour: 'var(--raider)' },
    { name: 'Cleanser', colour: 'var(--cleanser)' }, { name: 'Nutrient', colour: '#3cc050' },
    { name: 'Erase', colour: '#888' },
  ];
  const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4];

  const lab = {
    preset: 0,
    seed: PRESETS[0].seed,
    quality: 512,
    params: { ...PRESETS[0].params } as Params,
    running: true,
    speed: 1,
    acc: 0,
    tool: 0,
    radius: 18,
    strength: 0.35,
  };
  const view: ViewSettings = { mode: 0, exposure: 1.4, bloom: 0.6, contours: false, nutrient: true };
  const cam: Camera = { cx: 0, cy: 0, zoom: 1 };
  // start close: a creature should fill a good part of the view
  let zoomRel = 2;

  let sim = new Simulation(device, lab.quality, lab.quality, lab.params);
  const renderer = new Renderer(device, ctx, format);
  renderer.attach(sim);
  let probe: ChaosProbe | null = null;

  function fitCamera() {
    const fit = Math.max(canvas.width / sim.w, canvas.height / sim.h);
    cam.zoom = fit * zoomRel;
    if (!cam.cx && !cam.cy) { cam.cx = sim.w / 2; cam.cy = sim.h / 2; }
  }

  /** Recreate the experiment from the seed. Resolution only changes here, never mid-run. */
  function reset() {
    if (sim.w !== lab.quality) {
      sim.destroy();
      sim = new Simulation(device, lab.quality, lab.quality, lab.params);
      renderer.attach(sim);
      cam.cx = cam.cy = 0;
    }
    sim.setParams(lab.params);
    sim.upload(initialState(sim.w, sim.h, lab.seed, PRESETS[lab.preset].init, lab.params));
    sim.seed = lab.seed;
    history.length = 0;
    lastStats = null;
    clampRecent = 0;
    stopProbe();
    stopFollow();
    found = [];
    fitCamera();
    syncAll();
    // open on a creature, not on empty space
    setTimeout(async () => { await pollDiagnostics(); if (!follow) followNext(); }, 50);
  }

  function loadPreset(i: number) {
    lab.preset = i;
    lab.seed = PRESETS[i].seed;
    lab.params = { ...PRESETS[i].params };
    view.mode = PRESETS[i].view ?? 0;
    $('#blurb').textContent = PRESETS[i].blurb;
    presetButtons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
    reset();
  }

  function setParam(key: ParamKey, v: number) {
    lab.params[key] = v;
    sim.setParams({ [key]: v });
    probe?.setParams({ [key]: v });
    updateDt();
  }

  // ------------------------------------------------------------- panel ---
  const ctl = $('#controls');
  const syncers: (() => void)[] = [];
  const syncAll = () => syncers.forEach((f) => f());

  function section(title: string, open = false) {
    const d = document.createElement('details');
    d.open = open;
    d.innerHTML = `<summary>${title}</summary>`;
    ctl.appendChild(d);
    return d;
  }
  function row(sec: HTMLElement, label: string, tip = '') {
    const r = document.createElement('div');
    r.className = 'row';
    const id = `c${Math.random().toString(36).slice(2, 8)}`;
    r.innerHTML = `<label for="${id}" title="${tip || label}">${label}</label>`;
    sec.appendChild(r);
    return { r, id };
  }
  function slider(sec: HTMLElement, label: string, min: number, max: number, step: number,
    get: () => number, set: (v: number) => void, tip = '', fmt?: (v: number) => string) {
    const { r, id } = row(sec, label, tip);
    const dec = step >= 1 ? 0 : Math.min(4, Math.max(0, -Math.floor(Math.log10(step))));
    const f = fmt ?? ((v: number) => v.toFixed(dec));
    r.insertAdjacentHTML('beforeend', `<input id="${id}" type="range" min="${min}" max="${max}" step="${step}"><output></output>`);
    const inp = r.querySelector('input')!, out = r.querySelector('output')!;
    const sync = () => { inp.value = String(get()); out.textContent = f(get()); };
    inp.addEventListener('input', () => { set(+inp.value); out.textContent = f(+inp.value); });
    syncers.push(sync); sync();
  }
  function check(sec: HTMLElement, label: string, get: () => boolean, set: (v: boolean) => void) {
    const l = document.createElement('label');
    l.className = 'chk';
    l.innerHTML = `<input type="checkbox"> ${label}`;
    const inp = l.querySelector('input')!;
    const sync = () => { inp.checked = get(); };
    inp.addEventListener('change', () => set(inp.checked));
    sec.appendChild(l); syncers.push(sync); sync();
  }
  function select(sec: HTMLElement, label: string, opts: [string, string][], get: () => string, set: (v: string) => void) {
    const { r, id } = row(sec, label);
    r.insertAdjacentHTML('beforeend', `<select id="${id}" style="grid-column: span 2">${opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>`);
    const s = r.querySelector('select')!;
    const sync = () => { s.value = get(); };
    s.addEventListener('change', () => set(s.value));
    syncers.push(sync); sync();
  }
  function buttons(sec: HTMLElement, list: [string, () => void, string?][], cls = '') {
    const b = document.createElement('div');
    b.className = `btns ${cls}`;
    const els = list.map(([t, fn, title]) => {
      const e = document.createElement('button');
      e.innerHTML = t; e.onclick = fn;
      if (title) e.title = title;
      b.appendChild(e);
      return e;
    });
    sec.appendChild(b);
    return els;
  }
  function note(sec: HTMLElement, html: string) {
    const p = document.createElement('div');
    p.className = 'note'; p.innerHTML = html;
    sec.appendChild(p);
    return p;
  }

  const presetButtons = PRESETS.map((pr, i) => {
    const b = document.createElement('button');
    b.textContent = pr.name;
    b.onclick = () => loadPreset(i);
    $('#presets').appendChild(b);
    return b;
  });

  // simulation
  let s = section('Simulation', true);
  const [runBtn] = buttons(s, [
    ['Pause', () => togglePause(), 'Space'],
    ['Step', () => { lab.running = false; syncRun(); sim.run(1); probe && stepProbe(1); }, 'Period'],
    ['Reset from seed', () => reset(), 'R'],
  ]);
  const syncRun = () => { runBtn.textContent = lab.running ? 'Pause' : 'Run'; runBtn.setAttribute('aria-pressed', String(!lab.running)); };
  const togglePause = () => { lab.running = !lab.running; syncRun(); };
  slider(s, 'Speed', 0, SPEEDS.length - 1, 1, () => SPEEDS.indexOf(lab.speed), (v) => { lab.speed = SPEEDS[v]; },
    'Simulation steps per animation frame (at most 4)', (v) => `${SPEEDS[v]}×`);
  select(s, 'Resolution', QUALITIES.map((q) => [String(q), `${q}² cells`]), () => String(lab.quality), (v) => { lab.quality = +v; reset(); });
  {
    const { r, id } = row(s, 'Seed', 'Deterministic seed for the initial field');
    r.style.gridTemplateColumns = '112px 1fr auto';
    r.insertAdjacentHTML('beforeend', `<input id="${id}" type="number" min="0" max="4294967295" step="1"><button title="Random seed">⟳</button>`);
    const inp = r.querySelector('input')!, btn = r.querySelector('button')!;
    syncers.push(() => { inp.value = String(lab.seed); });
    inp.addEventListener('change', () => { lab.seed = Math.max(0, Math.min(4294967295, Math.round(+inp.value) || 0)); reset(); });
    btn.onclick = () => { lab.seed = (Math.random() * 2 ** 32) >>> 0; reset(); };
  }
  const dtSpec = PARAM_SPECS.find((p) => p.key === 'dt')!;
  slider(s, dtSpec.label, dtSpec.min, dtSpec.max, dtSpec.step, () => lab.params.dt, (v) => setParam('dt', v), dtSpec.tip);
  const dtNote = note(s, '');
  function updateDt() {
    const lim = dtLimit(lab.params), act = activeDt(lab.params);
    dtNote.innerHTML = `Δt = <b>${act.toFixed(3)}</b> · stability limit 0.20 / max D = ${lim.toFixed(3)}` +
      (lab.params.dt > lim ? ' <span style="color:var(--warn)">(request clamped)</span>' : '');
  }

  // model parameters, grouped
  const groups = [...new Set(PARAM_SPECS.map((p) => p.group))].filter((g) => g !== 'Simulation');
  for (const g of groups) {
    s = section(g, g === 'Body');
    if (g === 'Predation') note(s, 'Species is a phase carried by every parcel of mass. Raiders (120°) convert foragers (0°), cleansers (240°) convert raiders, foragers convert cleansers; in-between phases follow the same cycle.');
    if (g === 'Body') note(s, 'Mass flows up the affinity gradient U = G(K*m; μ, σ) + food + hunting − fleeing, and spreads out where it exceeds θ_A. Mass is only moved, never created by the flow.');
    if (g === 'New creatures') note(s, 'Genomes given to creatures placed from now on (presets, reset, brush). Living creatures keep their own genome, which travels and mixes with their mass.');
    for (const spec of PARAM_SPECS.filter((p) => p.group === g)) {
      if (spec.key === 'chaosOn') check(s, spec.label, () => lab.params[spec.key] > 0.5, (v) => setParam(spec.key, +v));
      else slider(s, spec.label, spec.min, spec.max, spec.step, () => lab.params[spec.key], (v) => setParam(spec.key, v), spec.tip);
    }
  }

  // rendering
  s = section('Rendering');
  select(s, 'View', VIEW_MODES.map((m, i) => [String(i), m]), () => String(view.mode), (v) => { view.mode = +v; });
  slider(s, 'Exposure', 0.1, 4, 0.05, () => view.exposure, (v) => { view.exposure = v; });
  slider(s, 'Bloom', 0, 2, 0.05, () => view.bloom, (v) => { view.bloom = v; });
  check(s, 'Biomass contours', () => view.contours, (v) => { view.contours = v; });
  check(s, 'Nutrient glow (green)', () => view.nutrient, (v) => { view.nutrient = v; });
  const [followBtn] = buttons(s, [
    ['Follow a creature', () => (follow ? stopFollow() : followNext()), 'F: follow the next creature'],
    ['Next', () => followNext(), 'Tab-like: jump to the next creature'],
    ['Reset view', () => { stopFollow(); zoomRel = 2; cam.cx = sim.w / 2; cam.cy = sim.h / 2; fitCamera(); }],
  ]);
  const syncFollow = () => {
    followBtn.textContent = follow ? `Following a ${['forager', 'raider', 'cleanser'][follow.species]}` : 'Follow a creature';
    followBtn.setAttribute('aria-pressed', String(!!follow));
  };

  // brush
  s = section('Brush', true);
  const toolButtons = buttons(s, TOOLS.map((t, i) => [`<span class="swatch" style="background:${t.colour}"></span>${t.name}`, () => setTool(i), `Key ${i + 1}`] as [string, () => void, string]), 'tools');
  const setTool = (i: number) => { lab.tool = i; toolButtons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i))); };
  setTool(0);
  note(s, 'Species tools drop a blob of new creature mass with that species’ genome; it organises itself. Nutrient and erase are soft brushes.');
  slider(s, 'Soft radius (cells)', 2, 80, 1, () => lab.radius, (v) => { lab.radius = v; });
  slider(s, 'Soft strength', 0.02, 1, 0.01, () => lab.strength, (v) => { lab.strength = v; });

  // diagnostics
  s = section('Diagnostics', true);
  const statsEl = document.createElement('div');
  statsEl.className = 'stats';
  s.appendChild(statsEl);
  const seriesCv = plotCanvas(s);
  note(s, 'Mass share of each species (colours) and total mass (white) over time.');
  const specCv = plotCanvas(s);
  note(s, 'Radially averaged power spectrum of total mass (log–log, 128² block average).');

  // chaos probe
  s = section('Chaos probe');
  note(s, `Runs a 256² copy of this experiment beside a shadow copy whose mass differs by 10⁻⁴ in an 8×8 patch of one creature, and reports
    λ(t) = (1/t) ln(‖u(t) − u′(t)‖ / ‖u(0) − u′(0)‖). This is a finite-time numerical diagnostic, not a rigorous proof of chaos.`);
  const [probeBtn] = buttons(s, [['Start probe', () => (probe ? stopProbe() : startProbe())]]);
  const probeEl = document.createElement('div');
  probeEl.className = 'stats';
  s.appendChild(probeEl);
  const probeCv = plotCanvas(s);

  // snapshots
  s = section('Snapshot');
  note(s, 'A .chaoslab file stores the exact current state (RGBA16F textures) and every parameter. Evolution after import on different hardware may not be bit-identical.');
  const fileInput = document.createElement('input');
  fileInput.type = 'file'; fileInput.accept = '.chaoslab'; fileInput.hidden = true;
  s.appendChild(fileInput);
  buttons(s, [['Export snapshot', () => exportSnapshot()], ['Import…', () => fileInput.click()]]);
  const snapNote = note(s, '');
  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0];
    fileInput.value = '';
    if (f) importSnapshot(await f.arrayBuffer()).catch((e) => { snapNote.textContent = (e as Error).message; });
  });

  // validation & benchmark
  s = section('Validation & benchmark');
  note(s, 'Runs GPU-vs-CPU, boundary, stability, dynamical-regime and snapshot checks. The lab pauses while it runs.');
  const [valBtn, benchBtn] = buttons(s, [['Run validation', () => validate()], ['Benchmark', () => benchmark()]]);
  const logEl = document.createElement('div');
  logEl.className = 'log';
  s.appendChild(logEl);

  function plotCanvas(sec: HTMLElement) {
    const c = document.createElement('canvas');
    c.className = 'plot';
    sec.appendChild(c);
    return c;
  }

  // -------------------------------------------------------- diagnostics ---
  interface Sample { t: number; P: number; f: [number, number, number] }
  const history: Sample[] = [];
  let lastStats: Stats | null = null;
  let creatures: [number, number, number] = [0, 0, 0];
  let found: Creature[] = [];
  /** the creature the camera follows, matched to the nearest same-species blob at each diagnostics poll */
  let follow: { species: number; x: number; y: number } | null = null;
  const camTarget = { x: 0, y: 0, active: false };
  const wrapDelta = (d: number, n: number) => d - Math.round(d / n) * n;

  function updateFollow() {
    if (!follow) return;
    let best: Creature | null = null, bd = Infinity;
    for (const c of found) {
      if (c.species !== follow.species) continue;
      const d = Math.hypot(wrapDelta(c.x - follow.x, sim.w), wrapDelta(c.y - follow.y, sim.h));
      if (d < bd) { bd = d; best = c; }
    }
    if (!best || bd > 6 * lab.params.radius) { follow = null; camTarget.active = false; syncFollow(); return; }
    follow = { species: best.species, x: best.x, y: best.y };
    camTarget.x = cam.cx + wrapDelta(best.x - cam.cx, sim.w);
    camTarget.y = cam.cy + wrapDelta(best.y - cam.cy, sim.h);
    camTarget.active = true;
  }
  let followIndex = -1;
  function followNext() {
    if (!found.length) return;
    followIndex = (followIndex + 1) % found.length;
    const c = found[followIndex];
    follow = { species: c.species, x: c.x, y: c.y };
    zoomRel = Math.max(zoomRel, 3);
    fitCamera();
    updateFollow();
    syncFollow();
  }
  function stopFollow() { follow = null; camTarget.active = false; syncFollow(); }
  let spectrum: Float64Array | null = null;
  let clampRecent = 0;
  let diagBusy = false;

  async function pollDiagnostics() {
    if (diagBusy) return;
    diagBusy = true;
    try {
      const d = await sim.readDiagnostics();
      if (!d || d.n === 0) return;
      lastStats = fieldStats(d.A, d.B, d.n * d.n);
      found = findCreatures(d.A, d.B, d.n).map((c) => ({ ...c, x: (c.x * sim.w) / d.n, y: (c.y * sim.h) / d.n }));
      creatures = [0, 1, 2].map((s) => found.filter((c) => c.species === s).length) as [number, number, number];
      updateFollow();
      clampRecent = d.clamped;
      const bio = new Float32Array(d.n * d.n);
      for (let i = 0; i < bio.length; i++) bio[i] = d.A[4 * i] + d.A[4 * i + 1] + d.A[4 * i + 2];
      spectrum = radialSpectrum(bio, d.n);
      history.push({ t: d.time, P: lastStats.biomass, f: lastStats.fractions });
      if (history.length > 400) history.shift();
      drawDiagnostics();
    } finally {
      diagBusy = false;
    }
  }

  function prepPlot(c: HTMLCanvasElement) {
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  }
  function line(g: CanvasRenderingContext2D, pts: [number, number][], colour: string) {
    g.strokeStyle = colour; g.lineWidth = 1.25; g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.stroke();
  }

  function drawDiagnostics() {
    const S = lastStats!;
    const f = (v: number, d = 3) => v.toFixed(d);
    statsEl.innerHTML = [
      ['<span class="swatch" style="background:var(--forager)"></span>Foragers', `${creatures[0]} · ${f(S.fractions[0] * 100, 1)} % of mass`],
      ['<span class="swatch" style="background:var(--raider)"></span>Raiders', `${creatures[1]} · ${f(S.fractions[1] * 100, 1)} % of mass`],
      ['<span class="swatch" style="background:var(--cleanser)"></span>Cleansers', `${creatures[2]} · ${f(S.fractions[2] * 100, 1)} % of mass`],
      ['Creature mass (mean per cell)', f(S.biomass, 4)],
      ['Nutrient (mean per cell)', f(S.nutrient, 4)],
      ['Mass + nutrient (conserved)', f(S.total, 5)],
      ['Genome μ (mass-weighted)', f(S.meanMu, 4)],
      ['Species mixing', f(S.mixing)],
      ['Spatial variance (mass)', S.variance.toExponential(2)],
      ['Shannon diversity of species', `${f(S.shannon)} / ${f(Math.log(3))}`],
    ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');

    let { g, w, h } = prepPlot(seriesCv);
    if (history.length > 1) {
      const t0 = history[0].t, t1 = history[history.length - 1].t || 1;
      const X = (t: number) => ((t - t0) / Math.max(t1 - t0, 1e-6)) * w;
      const pMax = Math.max(...history.map((s) => s.P), 1e-3);
      ['#1ad9ff', '#ff29c7', '#ff9e1a'].forEach((c, k) => line(g, history.map((s) => [X(s.t), h - 4 - s.f[k] * (h - 8)]), c));
      line(g, history.map((s) => [X(s.t), h - 4 - (s.P / pMax) * (h - 8)]), 'rgba(255,255,255,0.55)');
    }
    ({ g, w, h } = prepPlot(specCv));
    if (spectrum) {
      const ks = spectrum.length;
      const logs = Array.from(spectrum, (v) => Math.log10(v + 1e-14));
      const lo = Math.min(...logs), hi = Math.max(...logs);
      line(g, logs.map((v, i) => [(Math.log(i + 1) / Math.log(ks)) * w, h - 4 - ((v - lo) / Math.max(hi - lo, 1e-6)) * (h - 8)]), '#4fd8ff');
      g.fillStyle = '#8d97a1'; g.font = '10px ui-monospace, monospace';
      const peak = logs.indexOf(hi) + 1;
      g.fillText(`peak k = ${peak} → wavelength ≈ ${(sim.w / peak).toFixed(0)} cells`, 6, 12);
    }
  }

  // ------------------------------------------------------------ probe ---
  function startProbe() {
    probe = new ChaosProbe(device, lab.params, lab.seed, PRESETS[lab.preset].init);
    probe.measure();
    probeBtn.textContent = 'Stop probe';
    probeEl.innerHTML = '<span>Status</span><span>measuring…</span>';
  }
  function stopProbe() {
    probe?.destroy();
    probe = null;
    probeBtn.textContent = 'Start probe';
  }
  function stepProbe(n: number) {
    const enc = device.createCommandEncoder();
    probe!.encode(enc, n);
    device.queue.submit([enc.finish()]);
  }
  function drawProbe(sm: ProbeSample) {
    const p = probe!;
    probeEl.innerHTML = [
      ['Finite-time λ', `${sm.lambda.toFixed(4)} / time`],
      ['Elapsed t', sm.t.toFixed(1)],
      ['Separation ‖u − u′‖', `${sm.d.toExponential(2)} (from ${p.d0.toExponential(2)})`],
      ['Reading', sm.d === 0 ? 'perturbation erased' : sm.lambda > 0.002 ? 'perturbations grow' : sm.lambda < -0.002 ? 'perturbations decay' : 'marginal'],
    ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
    const { g, w, h } = prepPlot(probeCv);
    const sm2 = p.samples;
    if (sm2.length > 1) {
      const lam = sm2.map((s) => s.lambda);
      const lo = Math.min(0, ...lam), hi = Math.max(0, ...lam), span = Math.max(hi - lo, 1e-6);
      const Y = (v: number) => h - 4 - ((v - lo) / span) * (h - 8);
      g.strokeStyle = 'rgba(255,255,255,0.2)'; g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.stroke();
      line(g, sm2.map((s, i) => [(i / (sm2.length - 1)) * w, Y(s.lambda)]), '#ffb547');
    }
  }

  // --------------------------------------------------------- snapshots ---
  async function exportSnapshot() {
    const raw = await sim.readRaw();
    const meta: SnapshotMeta = {
      app: 'chaos-lab', modelVersion: MODEL_VERSION, width: sim.w, height: sim.h, format: 'rgba32float', rowPitch: sim.w * 16,
      channels: { state: ['m mass', 'R nutrient', 'H chaos', 'clamp record'], genome: ['mu', 'sigma', 'phase x', 'phase y'], scent: ['scent x', 'scent y', 'scent total', '-'] },
      dtActive: sim.dt, step: sim.step, time: sim.time, seed: lab.seed, preset: PRESETS[lab.preset].name, params: { ...lab.params },
      palette: { forager: '#1ad9ff', raider: '#ff29c7', cleanser: '#ff9e1a', nutrient: '#3cc050' },
      created: new Date().toISOString(),
    };
    const blob = new Blob([encodeSnapshot(meta, raw) as BlobPart], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `chaoslab-${meta.preset.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-seed${lab.seed}-step${sim.step}.chaoslab`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    snapNote.textContent = `Exported step ${sim.step} (${(blob.size / 2 ** 20).toFixed(1)} MiB).`;
  }

  async function importSnapshot(buf: ArrayBuffer) {
    const { meta, raw } = decodeSnapshot(buf);
    lab.params = { ...lab.params, ...meta.params };
    lab.seed = meta.seed;
    lab.preset = Math.max(0, PRESETS.findIndex((p) => p.name === meta.preset));
    if (sim.w !== meta.width) {
      if (!QUALITIES.includes(meta.width)) throw new Error(`Snapshot grid ${meta.width}² is not a lab resolution`);
      lab.quality = meta.width;
      sim.destroy();
      sim = new Simulation(device, meta.width, meta.height, lab.params);
      renderer.attach(sim);
      cam.cx = cam.cy = 0;
    }
    sim.setParams(lab.params);
    sim.seed = meta.seed;
    sim.uploadRaw(raw, meta.step, meta.time);
    history.length = 0;
    stopProbe();
    fitCamera();
    syncAll();
    $('#blurb').textContent = `Imported snapshot of “${meta.preset}” at step ${meta.step}.`;
    snapNote.textContent = `Imported ${meta.width}² state, step ${meta.step}, t = ${meta.time.toFixed(1)}.`;
  }

  // ------------------------------------------------ validation/benchmark ---
  async function validate() {
    const was = lab.running;
    lab.running = false; syncRun();
    valBtn.disabled = benchBtn.disabled = true;
    logEl.innerHTML = '';
    const t0 = performance.now();
    const res = await runValidation(device, (c) => {
      logEl.insertAdjacentHTML('beforeend', `<div><span class="${c.pass ? 'pass' : 'fail'}">${c.pass ? 'PASS' : 'FAIL'}</span> ${c.name}\n   ${c.detail}</div>`);
      logEl.scrollTop = logEl.scrollHeight;
    });
    const passed = res.filter((c) => c.pass).length;
    logEl.insertAdjacentHTML('beforeend', `<div>${passed}/${res.length} passed in ${((performance.now() - t0) / 1000).toFixed(1)} s</div>`);
    console.table(res);
    (window as unknown as { validation: unknown }).validation = res;
    valBtn.disabled = benchBtn.disabled = false;
    lab.running = was; syncRun();
  }

  async function benchmark() {
    const was = lab.running;
    lab.running = false; syncRun();
    valBtn.disabled = benchBtn.disabled = true;
    logEl.innerHTML = `<div>${navigator.userAgent}</div><div>size   steps/s   ms per 4-step frame   GPU MiB</div>`;
    const rows = await runBenchmark(device, QUALITIES, (r) => {
      logEl.insertAdjacentHTML('beforeend', `<div>${String(r.size).padEnd(6)} ${r.stepsPerSecond.toFixed(0).padStart(7)}   ${r.frameMs.toFixed(2).padStart(8)}              ${r.memoryMiB.toFixed(1)}</div>`);
    });
    console.table(rows);
    (window as unknown as { benchmark: unknown }).benchmark = rows;
    valBtn.disabled = benchBtn.disabled = false;
    lab.running = was; syncRun();
  }

  // ------------------------------------------------------------ input ---
  let painting = false, panning = false;
  let last: { x: number; y: number } | null = null;
  const pending: BrushStroke[] = [];

  const toGrid = (e: PointerEvent) => {
    const dpr = devicePixelRatio || 1;
    return {
      x: cam.cx + (e.clientX * dpr - canvas.width / 2) / cam.zoom,
      y: cam.cy + (e.clientY * dpr - canvas.height / 2) / cam.zoom,
    };
  };
  const stamp = (g: { x: number; y: number }) => {
    const x = ((g.x % sim.w) + sim.w) % sim.w, y = ((g.y % sim.h) + sim.h) % sim.h;
    pending.push({ x, y, radius: lab.radius, strength: lab.strength, tool: lab.tool });
  };
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 2 || e.button === 1 || e.shiftKey) { panning = true; stopFollow(); last = { x: e.clientX, y: e.clientY }; return; }
    if (lab.tool <= 2) {
      // one blob of new creature mass, about a kernel radius across
      const g = toGrid(e);
      pending.push({
        x: ((g.x % sim.w) + sim.w) % sim.w, y: ((g.y % sim.h) + sim.h) % sim.h,
        radius: lab.params.radius, strength: 1, tool: lab.tool, genome: speciesGenome(lab.params, lab.tool),
      });
      return;
    }
    painting = true;
    last = toGrid(e);
    stamp(last);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (panning && last) {
      const dpr = devicePixelRatio || 1;
      cam.cx -= ((e.clientX - last.x) * dpr) / cam.zoom;
      cam.cy -= ((e.clientY - last.y) * dpr) / cam.zoom;
      last = { x: e.clientX, y: e.clientY };
    } else if (painting && last) {
      const g = toGrid(e);
      const dist = Math.hypot(g.x - last.x, g.y - last.y), spacing = Math.max(1, lab.radius * 0.5);
      const n = Math.min(8, Math.floor(dist / spacing));
      for (let i = 1; i <= n; i++) stamp({ x: last.x + ((g.x - last.x) * i) / n, y: last.y + ((g.y - last.y) * i) / n });
      if (n) last = g;
    }
  });
  const endPointer = () => { painting = panning = false; last = null; };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const dpr = devicePixelRatio || 1;
    const before = toGrid(e as unknown as PointerEvent);
    zoomRel = Math.min(32, Math.max(0.25, zoomRel * Math.exp(-e.deltaY * 0.0015)));
    fitCamera();
    cam.cx = before.x - (e.clientX * dpr - canvas.width / 2) / cam.zoom;
    cam.cy = before.y - (e.clientY * dpr - canvas.height / 2) / cam.zoom;
  }, { passive: false });

  addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
    if (e.key === ' ') { e.preventDefault(); togglePause(); }
    else if (e.key === '.') { lab.running = false; syncRun(); sim.run(1); }
    else if (e.key === 'r' || e.key === 'R') reset();
    else if (e.key === 'h' || e.key === 'H') $('#panel').classList.toggle('hidden');
    else if (e.key === 'f' || e.key === 'F') followNext();
    else if (e.key === 'Escape') stopFollow();
    else if (e.key >= '1' && e.key <= '5') setTool(+e.key - 1);
    else if (e.key === '[') { lab.radius = Math.max(2, lab.radius - 2); syncAll(); }
    else if (e.key === ']') { lab.radius = Math.min(80, lab.radius + 2); syncAll(); }
  });
  $('#toggle').onclick = () => $('#panel').classList.toggle('hidden');

  function resize() {
    const dpr = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(innerWidth * dpr));
    canvas.height = Math.max(1, Math.round(innerHeight * dpr));
    fitCamera();
  }
  addEventListener('resize', resize);

  // ------------------------------------------------------------- loop ---
  let frame = 0, fpsT = performance.now(), fps = 0, sps = 0, stepsAtFps = 0;
  const hud = $('#hud');

  function loop() {
    frame++;
    for (const st of pending.splice(0, 8)) sim.paint(st);
    if (lab.running) {
      lab.acc += lab.speed;
      const n = Math.min(4, Math.floor(lab.acc));
      lab.acc -= n;
      if (n > 0) {
        const enc = device.createCommandEncoder();
        sim.encodeSteps(enc, n);
        probe?.encode(enc, n);
        device.queue.submit([enc.finish()]);
      }
    }
    if (camTarget.active) {
      // ease towards the followed creature
      cam.cx += (camTarget.x - cam.cx) * 0.05;
      cam.cy += (camTarget.y - cam.cy) * 0.05;
    }
    renderer.draw(view, cam, canvas);
    if (frame % 20 === 0) pollDiagnostics();
    if (probe && frame % 30 === 0) probe.measure().then((sm) => sm && probe && drawProbe(sm));

    const now = performance.now();
    if (now - fpsT > 500) {
      fps = (frame * 1000) / (now - fpsT); sps = ((sim.step - stepsAtFps) * 1000) / (now - fpsT);
      frame = 0; fpsT = now; stepsAtFps = sim.step;
      const warn = clampRecent > 0
        ? `<span class="warn">⚠ clamping: ${clampRecent} cell-steps since last check</span>`
        : '<span class="ok">numerics: no clamping</span>';
      hud.innerHTML = `${PRESETS[lab.preset].name}\n` +
        `${sim.w}²  step ${sim.step}  t = ${sim.time.toFixed(1)}\n` +
        `Δt ${sim.dt.toFixed(3)}  ${fps.toFixed(0)} fps  ${sps.toFixed(0)} steps/s${lab.running ? '' : '  (paused)'}\n${warn}`;
    }
    requestAnimationFrame(loop);
  }

  // console handle for experiments: chaosLab.set('s', 0.3)
  Object.assign(window, { chaosLab: { lab, view, cam, PRESETS, get sim() { return sim; }, set: (k: ParamKey, v: number) => { setParam(k, v); syncAll(); }, reset, loadPreset } });

  resize();
  loadPreset(0);
  syncRun();
  updateDt();
  requestAnimationFrame(loop);
}
