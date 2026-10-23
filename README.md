# Chaos Lab

**[Try it live →](https://zahintapadar.github.io/chaos-lab/)** (needs a WebGPU browser)

A WebGPU artificial-life laboratory. Flow-Lenia creatures carry their own genome, feed on a nutrient field, smell each other, hunt and convert one another in a rock–paper–scissors cycle, bud into colonies and blend into new in-between species. Mass is never created by the dynamics, only moved, eaten or converted.

It is a **synthetic ecology**: a playground of local rules, not a model of any real organism or ecosystem.

Runs in any browser with WebGPU (recent Chrome or Edge, Safari 26+, Firefox 141+ on Windows). There is no WebGL fallback.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + static build in dist/
```

Pushing to `main` deploys `dist/` to GitHub Pages (`.github/workflows/pages.yml`).

## The model

Every cell holds three RGBA32F textures:

| Texture | Channels | Meaning |
| --- | --- | --- |
| `S` | m, R, H, clamp | creature mass, nutrient, coupled-map chaos state, record of numerical clamps |
| `G` | μ, σ, φx, φy | the genome carried by that mass: growth centre and width, species phase vector φ |
| `C` | ψx, ψy, ψT, – | scent: the phase-carrying and total scent of nearby mass |

Species identity is a **phase** θ on a circle: forager 0°, raider 120°, cleanser 240°. The prey of θ sits at θ − 120° and its predator at θ + 120°, so raiders eat foragers, cleansers eat raiders and foragers eat cleansers, and every phase in between follows the same cycle.

One step has three GPU passes.

**1. Affinity.** With Lenia's ring kernel K (radius R, K(r) = exp(4 − 1/(r(1 − r))), normalised), Um = K * m and Φ = K * (m φ):

```
U = G(Um; μ_eff, σ) + β_food·R + β_hunt·ln(p + ε) − β_flee·ln(q + ε)
G(u) = 2 exp(−(u − μ)² / 2σ²) − 1          μ_eff = μ[1 + ξ(H − ½)]
```

Lenia's growth function is used as an *affinity*, not as a source of mass. p and q are the scents of prey and predators, recovered from the scent field as (ψ·û + ½ψT)/1.5. The total ψT is needed because the three unit phases sum to zero, so ψ alone cannot separate a species from the common background. Sensing is logarithmic (Weber's law): the gradient of ln p depends on the direction to the prey, not its distance, so a faint trail pulls as clearly as a strong one.

**2. Flow.** Mass climbs the affinity gradient and spreads where it gets crowded:

```
F = (1 − α)∇U − α∇m        α = clip((m / θ_A)², 0, 1)
```

Gradients are Sobel filters. The displacement Δt·F is clipped to ±d cells.

**3. Reintegration and reactions.** Every parcel moves by Δt·F and is spread over a square of half-width s. Each cell gathers the overlapping fractions from a 9 × 9 neighbourhood, so transport conserves mass exactly. Arriving genomes are mass-weighted means of their sources, which is how species mix. Then:

```
feeding     Δm = +Δt·f·R·m·max(G, 0)     ΔR = −(same)
decay       Δm = −Δt·λ·m                  ΔR = +(same)
nutrient    ΔR += Δt·D_R ∇²R
conversion  φ → φ + Δt·κ·c·(rot₊₁₂₀(φ) − φ)    where c = max(0, Φ·û_pred) is predator contact
scent       Δψ = Δt (D_S ∇²ψ + ω_S m (φ, 1) − λ_S ψ)
chaos       H' = (1 − κ_H) r H(1 − H) + κ_H ⟨H⟩₈
```

Biomass plus nutrient is conserved by every term except clamps and the brush. Nothing tells a creature to divide: bodies that outgrow what their kernel can hold together split into chains of cells by themselves.

**Timestep.** Δt is clamped to 0.20 / max(D_R, D_S), the explicit diffusion stability limit. Flow displacement is clipped separately, so it cannot violate the gather window.

### Where it departs from the original plan

- **The core is Flow-Lenia, not a reaction–diffusion PDE.** The PDE version filled the screen with fast, field-wide spirals. The goal was slow, sparse creatures.
- **Classic Lenia was tried and dropped.** Orbium is an attractor of Lenia's own discrete update A ← clip(A + Δt·G) at T = 10. It dies under a Heun/RK2 step at the same Δt. Classic Lenia creatures also could not reproduce. Flow-Lenia's mass conservation keeps bodies alive, and feeding lets them grow and split.
- **All state is single precision.** In half precision the logistic map stuck exactly on its unstable fixed point at r = 3.2 (0.6875 is representable in f16) and fell into a rounding cycle at r = 3.9.
- **Scent and log-sensing were added.** Without them, creatures sensed only one kernel radius and never met.

## Using the lab

| Input | Action |
| --- | --- |
| 1 · 2 · 3 | Drop a new forager, raider or cleanser (click) with the current blueprint genome |
| 4 · 5 | Soft brushes: add nutrient, erase mass |
| right-drag / shift-drag | Pan (the world is a torus) |
| wheel | Zoom |
| F / Esc | Follow the next creature / stop following |
| space · . · R | Pause · single step · reset from seed |
| H | Hide the panel |

**Presets:** First contact, Cyclic hunt, Nutrient bloom, Mixing pot, Near-critical noise, Recovery experiment. Each is a seed plus a full parameter set, and the seed recreates the starting field exactly.

**Panel sections:**
- **Simulation:** speed (0.1–4 steps per frame), resolution (256²–1024², changed only on reset), seed and Δt with its stability limit.
- **Model groups:** Body, Feeding, Predation, New creatures (blueprint genomes for placed creatures; living ones keep their own) and Chaos.
- **Rendering:** nine views, including bioluminescent (colour from species phase), Lenia classic, species phase, nutrient, affinity, flow speed, genome μ, scent, chaos field and numerical clamps. Bloom and contours can be toggled.
- **Diagnostics:** creature count and mass share per species, mass, nutrient, the conserved total, mean genome μ, species mixing, spatial variance, Shannon diversity, a time series and a radial power spectrum.
- **Chaos probe:** a 256² copy beside a shadow copy perturbed by 10⁻⁴. It reports the finite-time Lyapunov estimate λ(t) = (1/t) ln(‖u − u′‖ / ‖u₀ − u′₀‖). This is a numerical diagnostic, not a proof of chaos. A creature that merely drifts also reads as positive.
- **Snapshot:** `.chaoslab` export and import of the exact state.

### Snapshot format

```
"CHAOSLAB" · uint16 version (3) · uint32 metadata length · UTF-8 JSON metadata
state block   w × h × 16 bytes  RGBA32F (m, R, H, clamp)
genome block  w × h × 16 bytes  RGBA32F (μ, σ, φx, φy)
scent block   w × h × 16 bytes  RGBA32F (ψx, ψy, ψT, 0)
```

The metadata records the model version, grid size, row pitch, Δt, step, time, seed, preset, every parameter and the palette. Import validates magic, version, format, size and length. Resuming on different hardware starts from the same bits, but later evolution may differ in rounding.

## Validation

**Validation & benchmark → Run validation** runs 23 checks on the GPU (about 40 s). The latest run passed all 23:

| Check | Result |
| --- | --- |
| GPU vs CPU reference, 1 and 10 steps (64², all terms on) | max \|Δ\| 2.4 × 10⁻⁷ after 1 step, 1.3 × 10⁻⁵ after 10 |
| Transport conserves creature mass (500 steps) | relative drift 2.7 × 10⁻⁵ |
| Feeding and decay conserve mass + nutrient | mass 1343 → 2472, total drift 4.2 × 10⁻⁶ |
| Uniform field stays uniform | exactly 0 spread after 50 steps |
| Periodic boundaries | an edge stamp wraps; diffusion asymmetry 1.2 × 10⁻⁷ |
| Predation converts prey into predator | forager 224 → 76, raider 223 → 371, total unchanged |
| Merging creatures blend their genomes | blended phases appear |
| Bounded and finite: six presets, every control at min and at max | no non-finite, negative or out-of-range values |
| Timestep clamp | 0.5 requested with D = 1 → 0.2 |
| Logistic map r = 2.8 / 3.2 / 3.9 (GPU) | fixed point 0.6429; period 2 (0.5130, 0.7995); λ ≈ 0.479 (literature ≈ 0.49) |
| Chaos probe with interacting creatures / a dissolving body | λ = +0.033 / −0.004 |
| Snapshot round trip | 0 differing values, identical metadata |

The CPU reference in `src/model.ts` mirrors the WGSL pass for pass, including the half-precision packing of the convolution tile. `npm run flow -- [steps] [size] ['{"param":value}'] [f,r,c]` runs it headless and prints mass, nutrient, the conserved total and mass per species.

### Benchmark

Apple-silicon Mac, Chrome (Claude desktop browser pane), Cyclic hunt preset, four steps per frame:

| Grid | Steps / s | ms per 4-step frame | GPU memory |
| --- | --- | --- | --- |
| 256² | 441 | 9.1 | 9 MiB |
| 512² | 165 | 24.3 | 35 MiB |
| 768² | 91 | 43.9 | 78 MiB |
| 1024² | 61 | 65.9 | 138 MiB |

The default speed is one step per frame. The affinity pass stages each 8 × 8 tile plus a 16-cell halo in workgroup memory, and skips the convolution for empty tiles. The 9 × 9 reintegration gather is the main cost.

## Known limits

- **First contact is slow.** The three creatures can rest apart for thousands of steps, because the scent takes about 1/λ_S time units to build up across the field.
- **The chaos probe can't tell drift from chaos.** A drifting creature also reads as a positive λ.
- **The kernel radius is capped at 16 cells**, the size of the workgroup halo.

## Code map

```
src/model.ts        parameters, kernel, seeded initial fields, CPU reference step
src/shaders.ts      WGSL: affinity, flow, gather + reactions, brush, reduction, colour, bloom, composite
src/sim.ts          GPU textures, pipelines, stepping, brush, readbacks
src/renderer.ts     colourise → bloom → composite onto the canvas
src/presets.ts      the six experiments
src/diagnostics.ts  statistics, creature finding, radial spectrum
src/probe.ts        finite-time Lyapunov probe
src/snapshot.ts     .chaoslab encode / decode
src/validation.ts   in-browser validation suite and benchmark
src/main.ts         lab UI, input, follow camera, frame loop
tools/flow.ts       headless CPU run
```

## References

- B. W.-C. Chan, [Lenia: Biology of Artificial Life](https://arxiv.org/abs/1812.05433) (2019)
- E. Plantec et al., [Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization](https://arxiv.org/abs/2212.07906) (2023)
- T. Reichenbach, M. Mobilia, E. Frey, [Mobility promotes and jeopardizes biodiversity in rock–paper–scissors games](https://www.nature.com/articles/nature06095), Nature 448 (2007)
- [WebGPU](https://www.w3.org/TR/webgpu/) and [WGSL](https://www.w3.org/TR/WGSL/) specifications
