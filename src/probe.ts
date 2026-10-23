// The chaos probe: a 256² copy of the experiment and a shadow copy that differs
// by a microscopic perturbation. The growth of their separation gives a
// finite-time Lyapunov estimate,
//
//     λ(t) = (1/t) · ln( ‖u(t) − u′(t)‖ / ‖u(0) − u′(0)‖ ),
//
// a numerical diagnostic of sensitivity to initial conditions, not a proof of
// chaos: it saturates once the separation reaches the size of the attractor,
// and a creature that merely drifts at a slightly different speed also reads
// as a positive λ.

import { initialState, type InitSpec, type Params, type State } from './model';
import { Simulation } from './sim';

export const PROBE_SIZE = 256;
/** added to the mass in an 8 × 8 patch at the densest cell */
export const PROBE_DELTA = 1e-4;

export interface ProbeSample { t: number; d: number; lambda: number }

/** Euclidean distance over mass, nutrient, chaos state, genome and scent (the clamp record is excluded). */
export function separation(a: State, b: State): number {
  let s = 0;
  for (let i = 0; i < a.S.length; i++) {
    if ((i & 3) !== 3) { const d = a.S[i] - b.S[i]; s += d * d; }
    const g = a.G[i] - b.G[i], c = a.C[i] - b.C[i];
    s += g * g + c * c;
  }
  return Math.sqrt(s);
}

export class ChaosProbe {
  readonly a: Simulation;
  readonly b: Simulation;
  d0 = 0;
  samples: ProbeSample[] = [];
  private busy = false;

  constructor(device: GPUDevice, params: Params, seed: number, init: InitSpec, start?: State) {
    const base = start ?? initialState(PROBE_SIZE, PROBE_SIZE, seed, init, params);
    const shadow: State = { ...base, S: base.S.slice(), G: base.G.slice(), C: base.C.slice() };
    let best = 0, bx = PROBE_SIZE / 2, by = PROBE_SIZE / 2;
    for (let y = 0; y < PROBE_SIZE; y++) for (let x = 0; x < PROBE_SIZE; x++) {
      const m = base.S[(y * PROBE_SIZE + x) * 4];
      if (m > best) { best = m; bx = x; by = y; }
    }
    for (let y = by - 4; y < by + 4; y++) for (let x = bx - 4; x < bx + 4; x++) {
      const i = ((((y % PROBE_SIZE) + PROBE_SIZE) % PROBE_SIZE) * PROBE_SIZE + (((x % PROBE_SIZE) + PROBE_SIZE) % PROBE_SIZE)) * 4;
      if (shadow.S[i] > 0) shadow.S[i] += PROBE_DELTA;
    }
    this.a = new Simulation(device, PROBE_SIZE, PROBE_SIZE, params);
    this.b = new Simulation(device, PROBE_SIZE, PROBE_SIZE, params);
    this.a.upload(base);
    this.b.upload(shadow);
  }

  setParams(p: Partial<Params>) { this.a.setParams(p); this.b.setParams(p); }

  encode(enc: GPUCommandEncoder, n: number) { this.a.encodeSteps(enc, n); this.b.encodeSteps(enc, n); }

  /** Read both grids and append a sample. Skips if a measurement is already in flight. */
  async measure(): Promise<ProbeSample | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const t = this.a.time;
      const [sa, sb] = await Promise.all([this.a.readState(), this.b.readState()]);
      const d = separation(sa, sb);
      if (!this.d0) {
        this.d0 = d;
        return null;
      }
      const sample = { t, d, lambda: t > 0 ? Math.log(Math.max(d, 1e-30) / this.d0) / t : 0 };
      this.samples.push(sample);
      if (this.samples.length > 600) this.samples.shift();
      return sample;
    } finally {
      this.busy = false;
    }
  }

  destroy() { this.a.destroy(); this.b.destroy(); }
}
