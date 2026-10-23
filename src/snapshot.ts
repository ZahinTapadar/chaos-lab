// The .chaoslab snapshot container (little-endian):
//
//   magic            8 bytes  "CHAOSLAB"
//   version          uint16
//   metadataLength   uint32
//   metadata         UTF-8 JSON (SnapshotMeta)
//   state block      width × height × 16 bytes: RGBA32F (m, R, H, clamp record), rows of rowPitch bytes
//   genome block     width × height × 16 bytes: RGBA32F (μ, σ, φx, φy), rows of rowPitch bytes
//   scent block      width × height × 16 bytes: RGBA32F (ψx, ψy, ψT, 0), rows of rowPitch bytes
//
// A snapshot preserves the exact state and every parameter. Resuming on other
// hardware continues from the same bits, but GPU drivers may round differently
// afterwards, so later evolution is not promised to be pixel-identical.

import { MODEL_VERSION, type Params } from './model';
import type { RawState } from './sim';

export const SNAPSHOT_MAGIC = 'CHAOSLAB';
export const SNAPSHOT_VERSION = 3;
export const SNAPSHOT_SIZES = [256, 512, 768, 1024];

export interface SnapshotMeta {
  app: 'chaos-lab';
  modelVersion: number;
  width: number;
  height: number;
  format: 'rgba32float';
  rowPitch: number;
  channels: { state: string[]; genome: string[]; scent: string[] };
  dtActive: number;
  step: number;
  time: number;
  seed: number;
  preset: string;
  params: Params;
  palette: Record<string, string>;
  created: string;
}

export function encodeSnapshot(meta: SnapshotMeta, { S, G, C }: RawState): Uint8Array {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('big-endian hosts are not supported');
  const json = new TextEncoder().encode(JSON.stringify(meta));
  const out = new Uint8Array(14 + json.length + S.byteLength + G.byteLength + C.byteLength);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) out[i] = SNAPSHOT_MAGIC.charCodeAt(i);
  dv.setUint16(8, SNAPSHOT_VERSION, true);
  dv.setUint32(10, json.length, true);
  out.set(json, 14);
  out.set(new Uint8Array(S.buffer, S.byteOffset, S.byteLength), 14 + json.length);
  out.set(new Uint8Array(G.buffer, G.byteOffset, G.byteLength), 14 + json.length + S.byteLength);
  out.set(new Uint8Array(C.buffer, C.byteOffset, C.byteLength), 14 + json.length + S.byteLength + G.byteLength);
  return out;
}

export function decodeSnapshot(buf: ArrayBuffer): { meta: SnapshotMeta; raw: RawState } {
  const u8 = new Uint8Array(buf);
  const fail = (m: string): never => { throw new Error(`Not a valid .chaoslab snapshot: ${m}`); };
  if (u8.length < 14) fail('file too short');
  const magic = String.fromCharCode(...u8.subarray(0, 8));
  if (magic !== SNAPSHOT_MAGIC) fail('bad magic');
  const dv = new DataView(buf);
  const version = dv.getUint16(8, true);
  if (version !== SNAPSHOT_VERSION) fail(`unsupported version ${version}`);
  const len = dv.getUint32(10, true);
  if (14 + len > u8.length) fail('metadata overruns file');
  let meta: SnapshotMeta;
  try {
    meta = JSON.parse(new TextDecoder().decode(u8.subarray(14, 14 + len)));
  } catch {
    return fail('metadata is not JSON');
  }
  if (meta.app !== 'chaos-lab') fail('metadata is not from Chaos Lab');
  if (meta.modelVersion !== MODEL_VERSION) fail(`model version ${meta.modelVersion} (this build runs ${MODEL_VERSION})`);
  if (meta.format !== 'rgba32float') fail(`texture format ${meta.format}`);
  if (meta.width !== meta.height || !SNAPSHOT_SIZES.includes(meta.width)) fail(`grid ${meta.width}×${meta.height} is outside the supported sizes`);
  if (meta.rowPitch !== meta.width * 16) fail('unexpected row pitch');
  const block = meta.width * meta.height * 16;
  const start = 14 + len;
  if (u8.length !== start + 3 * block) fail(`expected ${start + 3 * block} bytes, found ${u8.length}`);
  // copy out so the blocks are 4-byte aligned
  const [S, G, C] = [0, 1, 2].map((k) => new Float32Array(u8.slice(start + k * block, start + (k + 1) * block).buffer));
  return { meta, raw: { S, G, C } };
}
