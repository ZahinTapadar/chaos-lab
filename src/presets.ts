// The curated experiments. Each is a seed, a full parameter set and a recipe
// for the initial field; the seed alone recreates the starting state.

import { DEFAULTS, type InitSpec, type Params } from './model';

export interface Preset {
  name: string;
  blurb: string;
  seed: number;
  params: Params;
  init: InitSpec;
  /** suggested render mode: 0 is the bioluminescent view */
  view?: number;
}

export const PRESETS: Preset[] = [
  {
    name: 'First contact',
    blurb: 'One forager, one raider, one cleanser. Each blob of mass organises itself into a body, smells the others and closes in. Follow one (F).',
    seed: 1729,
    params: { ...DEFAULTS },
    init: { creatures: [1, 1, 1], nutrient: 0.24, spacing: 1.5 },
  },
  {
    name: 'Cyclic hunt',
    blurb: 'Three of each. Raiders turn foragers into raiders, cleansers turn raiders, foragers turn cleansers; mass is never lost, only converted.',
    seed: 404,
    params: { ...DEFAULTS, convert: 0.8 },
    init: { creatures: [3, 3, 3], nutrient: 0.24, spacing: 1.4 },
  },
  {
    name: 'Nutrient bloom',
    blurb: 'Rich fertile patches. Creatures that find them swell until they no longer hold together and split.',
    seed: 2718,
    params: { ...DEFAULTS, feed: 0.08, betaFood: 0.8, betaHunt: 0.5 },
    init: { creatures: [2, 2, 2], nutrient: 0.2, spacing: 2, patches: 8 },
  },
  {
    name: 'Mixing pot',
    blurb: 'Weak conversion and strong crowding: colliding creatures merge, and their genomes blend into new in-between species.',
    seed: 666,
    params: { ...DEFAULTS, convert: 0.05, betaFlee: 0, thetaA: 1.2 },
    init: { creatures: [3, 3, 3], nutrient: 0.24, spacing: 1.2 },
  },
  {
    name: 'Near-critical noise',
    blurb: 'A coupled logistic lattice nudges every creature’s preferred density. Bodies wobble, stretch and tear.',
    seed: 31415,
    params: { ...DEFAULTS, chaosOn: 1, mapR: 3.86, mapKappa: 0.3, xi: 0.2 },
    init: { creatures: [2, 2, 2], nutrient: 0.24, spacing: 1.5 },
  },
  {
    name: 'Recovery experiment',
    blurb: 'A starved field with one nutrient pulse in the middle. Creatures waste away unless they find food: paint nutrient (key 4) and watch who recovers.',
    seed: 8128,
    params: { ...DEFAULTS, decay: 0.006, feed: 0.06 },
    init: { creatures: [2, 2, 2], nutrient: 0.05, spacing: 1.5, pulse: [0.5, 0.5, 0.12, 1.5] },
  },
];
