// Draws a Simulation: colourise the state at grid resolution, bloom it at a
// quarter of that, then composite onto the canvas with pan/zoom over the torus.

import type { Simulation } from './sim';
import { BLOOM_BLUR_WGSL, BLOOM_DOWN_WGSL, COLOUR_WGSL, COMPOSITE_WGSL } from './shaders';

export interface ViewSettings {
  mode: number;
  exposure: number;
  bloom: number;
  contours: boolean;
  nutrient: boolean;
}

export interface Camera { cx: number; cy: number; zoom: number }

export class Renderer {
  private colour!: GPUTexture;
  private glow: GPUTexture[] = [];
  private colourPipe: GPUComputePipeline;
  private downPipe: GPUComputePipeline;
  private blurPipe: GPUComputePipeline;
  private compPipe: GPURenderPipeline;
  private viewBuf: GPUBuffer;
  private screenBuf: GPUBuffer;
  private dirBufs: GPUBuffer[];
  private sampler: GPUSampler;
  private sim?: Simulation;
  private colourBG: GPUBindGroup[] = [];
  private bloomBGs: GPUBindGroup[] = [];
  private compBG?: GPUBindGroup;

  constructor(private device: GPUDevice, private ctx: GPUCanvasContext, format: GPUTextureFormat) {
    const mk = (code: string, label: string) => device.createComputePipeline({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    this.colourPipe = mk(COLOUR_WGSL, 'colour');
    this.downPipe = mk(BLOOM_DOWN_WGSL, 'bloom-down');
    this.blurPipe = mk(BLOOM_BLUR_WGSL, 'bloom-blur');
    const module = device.createShaderModule({ label: 'composite', code: COMPOSITE_WGSL });
    this.compPipe = device.createRenderPipeline({
      label: 'composite', layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
    this.viewBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.screenBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dirBufs = [[1, 0], [0, 1]].map(([x, y]) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Float32Array([x, y, 0, 0]));
      return b;
    });
    this.sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat',
    });
  }

  /** (Re)build the targets that depend on the grid. */
  attach(sim: Simulation) {
    this.sim = sim;
    this.colour?.destroy();
    this.glow.forEach((t) => t.destroy());
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    this.colour = this.device.createTexture({ label: 'colour', size: [sim.w, sim.h], format: 'rgba16float', usage });
    this.glow = [0, 1].map((i) => this.device.createTexture({ label: `glow${i}`, size: [sim.w / 4, sim.h / 4], format: 'rgba16float', usage }));
    const bg = (p: GPUComputePipeline | GPURenderPipeline, entries: GPUBindingResource[], first = 0) =>
      this.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map((resource, i) => ({ binding: i + first, resource })) });
    this.colourBG = [0, 1].map((c) => bg(this.colourPipe, [
      { buffer: this.viewBuf }, sim.S[c].createView(), sim.G[c].createView(), sim.X.createView(), sim.D.createView(), this.colour.createView(),
      sim.C[c].createView(),
    ]));
    this.bloomBGs = [
      bg(this.downPipe, [this.colour.createView(), this.glow[0].createView()]),
      bg(this.blurPipe, [{ buffer: this.dirBufs[0] }, this.glow[0].createView(), this.glow[1].createView()]),
      bg(this.blurPipe, [{ buffer: this.dirBufs[1] }, this.glow[1].createView(), this.glow[0].createView()]),
    ];
    this.compBG = bg(this.compPipe, [{ buffer: this.screenBuf }, this.sampler, this.colour.createView(), this.glow[0].createView()]);
  }

  draw(view: ViewSettings, cam: Camera, canvas: HTMLCanvasElement) {
    const sim = this.sim!;
    const { device } = this;
    device.queue.writeBuffer(this.viewBuf, 0, new Float32Array([
      view.mode, view.exposure, +view.contours, 0, +view.nutrient, sim.params.thetaA, 0, 0,
    ]));
    device.queue.writeBuffer(this.screenBuf, 0, new Float32Array([
      cam.cx, cam.cy, cam.zoom, view.bloom, canvas.width, canvas.height, sim.w, sim.h,
    ]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.colourPipe);
    pass.setBindGroup(0, this.colourBG[sim.cur]);
    pass.dispatchWorkgroups(Math.ceil(sim.w / 8), Math.ceil(sim.h / 8));
    const gw = Math.ceil(sim.w / 32), gh = Math.ceil(sim.h / 32);
    pass.setPipeline(this.downPipe);
    pass.setBindGroup(0, this.bloomBGs[0]);
    pass.dispatchWorkgroups(gw, gh);
    pass.setPipeline(this.blurPipe);
    for (const bgi of [1, 2]) {
      pass.setBindGroup(0, this.bloomBGs[bgi]);
      pass.dispatchWorkgroups(gw, gh);
    }
    pass.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.compPipe);
    rp.setBindGroup(0, this.compBG!);
    rp.draw(3);
    rp.end();
    device.queue.submit([enc.finish()]);
  }

  get memoryBytes() {
    const s = this.sim;
    return s ? s.w * s.h * 8 + 2 * (s.w / 4) * (s.h / 4) * 8 : 0;
  }
}
