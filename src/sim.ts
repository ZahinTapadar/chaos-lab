// The GPU simulation: state textures, the three Flow-Lenia passes, the brush and
// readbacks. One instance per grid (the main lab, the chaos probe's pair, the
// validation suite's small grids all share the same compiled pipelines).

import { activeDt, emptyState, kernelTaps, PARAM_KEYS, type Params, type State } from './model';
import { AFFINITY_WGSL, BRUSH_WGSL, FLOW_WGSL, GATHER_WGSL, PARAM_FLOATS, REDUCE_WGSL } from './shaders';

export const STATE_FORMAT: GPUTextureFormat = 'rgba32float';
export const DIAG_SIZE = 128;
/** room for the ring kernel up to MAX_RADIUS */
const MAX_TAPS = 1024;

class Kernels {
  affinity: GPUComputePipeline;
  flow: GPUComputePipeline;
  gather: GPUComputePipeline;
  brush: GPUComputePipeline;
  reduce: GPUComputePipeline;
  constructor(device: GPUDevice) {
    const mk = (code: string, label: string) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    this.affinity = mk(AFFINITY_WGSL, 'affinity');
    this.flow = mk(FLOW_WGSL, 'flow');
    this.gather = mk(GATHER_WGSL, 'gather');
    this.brush = mk(BRUSH_WGSL, 'brush');
    this.reduce = mk(REDUCE_WGSL, 'reduce');
  }
}
const kernelCache = new WeakMap<GPUDevice, Kernels>();
const kernelsFor = (d: GPUDevice) => kernelCache.get(d) ?? (kernelCache.set(d, new Kernels(d)), kernelCache.get(d)!);

/** tools 0–2 add a creature blob with `genome`; 3 adds nutrient; 4 erases mass */
export interface BrushStroke { x: number; y: number; radius: number; strength: number; tool: number; genome?: number[] }

export interface Diagnostics { A: Float32Array; B: Float32Array; n: number; clamped: number; step: number; time: number }

/** raw texture contents, rows of w × 16 bytes */
export interface RawState { S: Float32Array; G: Float32Array; C: Float32Array }

export class Simulation {
  readonly w: number;
  readonly h: number;
  params: Params;
  seed = 0;
  step = 0;
  time = 0;
  /** ping-pong state (mass, nutrient, chaos, clamp) and genome slots */
  readonly S: GPUTexture[];
  readonly G: GPUTexture[];
  /** ping-pong scent */
  readonly C: GPUTexture[];
  /** per-step intermediates: affinity (U, q, g, K*m) and displacement */
  readonly X: GPUTexture;
  readonly D: GPUTexture;
  cur = 0;
  private k: Kernels;
  private paramBuf: GPUBuffer;
  private brushBuf: GPUBuffer;
  private counters: GPUBuffer;
  private tapBuf: GPUBuffer;
  private ntaps = 0;
  private tapRadius = -1;
  private brushSeed = 1;
  private affBG: GPUBindGroup[];
  private flowBG: GPUBindGroup[];
  private gatherBG: GPUBindGroup[];
  private brushBG: GPUBindGroup[];
  private reduceBG: GPUBindGroup[] = [];
  private diagA?: GPUTexture;
  private diagB?: GPUTexture;
  private paramsDirty = true;

  constructor(readonly device: GPUDevice, w: number, h: number, params: Params) {
    if (w % 8 || h % 8) throw new Error('grid dimensions must be multiples of 8');
    this.w = w; this.h = h; this.params = { ...params };
    this.k = kernelsFor(device);
    const tex = (label: string) => device.createTexture({
      label, size: [w, h], format: STATE_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    this.S = [tex('S0'), tex('S1')];
    this.G = [tex('G0'), tex('G1')];
    this.C = [tex('C0'), tex('C1')];
    this.X = tex('affinity');
    this.D = tex('displacement');
    this.paramBuf = device.createBuffer({ size: PARAM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.brushBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.counters = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.tapBuf = device.createBuffer({ size: MAX_TAPS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const v = (t: GPUTexture) => t.createView();
    const bg = (pipe: GPUComputePipeline, entries: [number, GPUBindingResource][]) =>
      device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.map(([binding, resource]) => ({ binding, resource })) });
    const pb = { buffer: this.paramBuf };
    this.affBG = [0, 1].map((c) => bg(this.k.affinity, [
      [0, pb], [1, v(this.S[c])], [2, v(this.G[c])], [3, v(this.X)], [4, { buffer: this.tapBuf }], [5, v(this.C[c])],
    ]));
    this.flowBG = [0, 1].map((c) => bg(this.k.flow, [[0, pb], [1, v(this.S[c])], [2, v(this.X)], [3, v(this.D)]]));
    this.gatherBG = [0, 1].map((c) => bg(this.k.gather, [
      [0, pb], [1, v(this.S[c])], [2, v(this.G[c])], [3, v(this.X)], [4, v(this.D)],
      [5, v(this.S[1 - c])], [6, v(this.G[1 - c])], [7, { buffer: this.counters }], [8, v(this.C[c])], [9, v(this.C[1 - c])],
    ]));
    this.brushBG = [0, 1].map((c) => bg(this.k.brush, [
      [0, { buffer: this.brushBuf }], [1, v(this.S[c])], [2, v(this.G[c])], [3, v(this.S[1 - c])], [4, v(this.G[1 - c])],
      [5, v(this.C[c])], [6, v(this.C[1 - c])],
    ]));
    const n = Math.min(DIAG_SIZE, w);
    if (w === h && w % n === 0) {
      const dt = (label: string) => device.createTexture({
        label, size: [n, n], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      this.diagA = dt('diagA'); this.diagB = dt('diagB');
      this.reduceBG = [0, 1].map((c) => bg(this.k.reduce, [[0, v(this.S[c])], [1, v(this.G[c])], [2, v(this.diagA!)], [3, v(this.diagB!)]]));
    }
  }

  get dt() { return activeDt(this.params); }

  setParams(p: Partial<Params>) { Object.assign(this.params, p); this.paramsDirty = true; }

  private writeParams() {
    const R = Math.round(this.params.radius);
    if (R !== this.tapRadius) {
      const taps = kernelTaps(R);
      this.ntaps = Math.min(MAX_TAPS, taps.length / 4);
      this.device.queue.writeBuffer(this.tapBuf, 0, taps, 0, this.ntaps * 4);
      this.tapRadius = R;
    }
    const data = new Float32Array(PARAM_FLOATS);
    PARAM_KEYS.forEach((key, i) => { data[i] = key === 'dt' ? this.dt : this.params[key]; });
    data[PARAM_KEYS.length] = this.w;
    data[PARAM_KEYS.length + 1] = this.h;
    data[PARAM_KEYS.length + 2] = this.ntaps;
    this.device.queue.writeBuffer(this.paramBuf, 0, data);
    this.paramsDirty = false;
  }

  /** Replace the whole state (seeded initial field, snapshot import, validation). */
  upload(st: State, step = 0, time = 0) {
    if (st.w !== this.w || st.h !== this.h) throw new Error('state size mismatch');
    this.uploadRaw({ S: Float32Array.from(st.S), G: Float32Array.from(st.G), C: Float32Array.from(st.C) }, step, time);
  }

  uploadRaw(raw: RawState, step = 0, time = 0) {
    this.cur = 0;
    this.device.queue.writeTexture({ texture: this.S[0] }, raw.S, { bytesPerRow: this.w * 16 }, [this.w, this.h]);
    this.device.queue.writeTexture({ texture: this.G[0] }, raw.G, { bytesPerRow: this.w * 16 }, [this.w, this.h]);
    this.device.queue.writeTexture({ texture: this.C[0] }, raw.C, { bytesPerRow: this.w * 16 }, [this.w, this.h]);
    this.step = step; this.time = time;
    this.device.queue.writeBuffer(this.counters, 0, new Uint32Array(4));
  }

  /** Encode `n` steps: affinity, flow, then reintegration with reactions. */
  encodeSteps(enc: GPUCommandEncoder, n: number) {
    if (this.paramsDirty) this.writeParams();
    const pass = enc.beginComputePass();
    const gx = Math.ceil(this.w / 8), gy = Math.ceil(this.h / 8);
    for (let i = 0; i < n; i++) {
      for (const [pipe, groups] of [[this.k.affinity, this.affBG], [this.k.flow, this.flowBG], [this.k.gather, this.gatherBG]] as const) {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, groups[this.cur]);
        pass.dispatchWorkgroups(gx, gy);
      }
      this.cur = 1 - this.cur;
      this.step++;
      this.time += this.dt;
    }
    pass.end();
  }

  run(n: number) {
    const enc = this.device.createCommandEncoder();
    this.encodeSteps(enc, n);
    this.device.queue.submit([enc.finish()]);
  }

  /** Apply one brush stamp. Coordinates are in cells; the brush wraps across edges. */
  paint(s: BrushStroke) {
    const g = s.genome ?? [0, 0, 0, 0];
    this.device.queue.writeBuffer(this.brushBuf, 0, new Float32Array([
      s.x, s.y, s.radius, s.strength, s.tool, this.w, this.h, this.brushSeed++, g[0], g[1], g[2], g[3],
    ]));
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.k.brush);
    pass.setBindGroup(0, this.brushBG[this.cur]);
    pass.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.cur = 1 - this.cur;
  }

  /** Raw texture contents of the current state. */
  async readRaw(): Promise<RawState> {
    const pitch = this.w * 16;
    const bufs = [0, 1, 2].map(() => this.device.createBuffer({ size: pitch * this.h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const enc = this.device.createCommandEncoder();
    [this.S[this.cur], this.G[this.cur], this.C[this.cur]].forEach((tex, i) =>
      enc.copyTextureToBuffer({ texture: tex }, { buffer: bufs[i], bytesPerRow: pitch }, [this.w, this.h]));
    this.device.queue.submit([enc.finish()]);
    const out = await Promise.all(bufs.map(async (b) => {
      await b.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(b.getMappedRange().slice(0));
      b.destroy();
      return copy;
    }));
    return { S: out[0], G: out[1], C: out[2] };
  }

  async readState(): Promise<State> {
    const { S, G, C } = await this.readRaw();
    const st = emptyState(this.w, this.h);
    st.S.set(S); st.G.set(G); st.C.set(C);
    return st;
  }

  /** Block-averaged 128² fields plus the clamp count since the last call (which resets it). */
  async readDiagnostics(): Promise<Diagnostics | null> {
    if (!this.diagA || !this.diagB) return null;
    const n = this.diagA.width, bytes = n * n * 16;
    const step = this.step, time = this.time;
    const bufs = [0, 1].map(() => this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const cnt = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.k.reduce);
    pass.setBindGroup(0, this.reduceBG[this.cur]);
    pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8));
    pass.end();
    [this.diagA, this.diagB].forEach((t, i) => enc.copyTextureToBuffer({ texture: t }, { buffer: bufs[i], bytesPerRow: n * 16 }, [n, n]));
    enc.copyBufferToBuffer(this.counters, 0, cnt, 0, 16);
    this.device.queue.submit([enc.finish()]);
    this.device.queue.writeBuffer(this.counters, 0, new Uint32Array(4));
    const [A, B, C] = await Promise.all([...bufs, cnt].map(async (b) => {
      await b.mapAsync(GPUMapMode.READ);
      const copy = b.getMappedRange().slice(0);
      b.destroy();
      return copy;
    }));
    return { A: new Float32Array(A), B: new Float32Array(B), n, clamped: new Uint32Array(C)[0], step, time };
  }

  /** GPU memory held by this grid's textures, kernel and diagnostics, in bytes. */
  get memoryBytes() {
    const n = this.diagA?.width ?? 0;
    return 8 * this.w * this.h * 16 + 2 * n * n * 16 + MAX_TAPS * 16;
  }

  destroy() {
    for (const t of [...this.S, ...this.G, ...this.C, this.X, this.D, this.diagA, this.diagB]) t?.destroy();
    for (const b of [this.paramBuf, this.brushBuf, this.counters, this.tapBuf]) b.destroy();
  }
}
