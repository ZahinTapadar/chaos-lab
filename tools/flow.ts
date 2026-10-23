// Headless Flow-Lenia check: conservation and creature shape on the CPU reference.
// usage: npm run flow -- [steps] [size] ['{"param":value}'] [creatures f,r,c]
import { DEFAULTS, initialState, kernelTaps, referenceStep } from '../src/model';
const steps = +(process.argv[2] ?? 300), n = +(process.argv[3] ?? 96);
const p = { ...DEFAULTS, ...JSON.parse(process.argv[4] ?? '{}') };
const cr = (process.argv[5] ?? '1,0,0').split(',').map(Number) as [number, number, number];
let st = initialState(n, n, 7, { creatures: cr, nutrient: 0.5, spacing: 1.5 }, p);
const taps = kernelTaps(p.radius);
const report = (s: number) => {
  let m = 0, R = 0, mx = 0, occ = 0;
  const sp = [0, 0, 0];
  for (let i = 0; i < n * n; i++) {
    const v = st.S[i * 4]; m += v; R += st.S[i * 4 + 1]; mx = Math.max(mx, v); if (v > 0.05) occ++;
    const th = Math.atan2(st.G[i * 4 + 3], st.G[i * 4 + 2]);
    sp[Math.round(((th + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI / 3)) % 3] += v;
  }
  console.log(`step ${String(s).padStart(4)} mass ${m.toFixed(3)} nutrient ${R.toFixed(2)} total ${(m + R).toFixed(3)} max ${mx.toFixed(3)} cells>0.05 ${occ} species ${sp.map((v) => v.toFixed(1)).join('/')}`);
};
report(0);
for (let s = 1; s <= steps; s++) { st = referenceStep(st, p, taps).state; if (s % Math.max(1, Math.floor(steps / 6)) === 0) report(s); }
