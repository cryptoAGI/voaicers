#!/usr/bin/env node
// Record faicey's face_clone (geometry.js proportions + symmetry, faceprint.js measureFaceprint) on N landmark sets,
// as the oracle for src/fclone.rs.
//   node testing/fclone/make_oracle.mjs <path/to/faicey/src/face_clone> [N] > tests/fixtures/faceprint_oracle.jsonl
// Only the landmarks the measures read are written (sparse, by index); numbers are written with String(), which
// round-trips a double. Awkward cases on purpose: coincident points (a zero distance becomes 1e-6), tiny and huge
// coordinates, negative z, a face of all zeros, and quality values outside [0, 1].
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const dir = process.argv[2];
const n = +(process.argv[3] || 1000);
const { proportions, symmetry, LM } = await import(dir + '/geometry.js');
const { measureFaceprint } = await import(dir + '/faceprint.js');
const used = [...new Set(Object.values(LM))].sort((a, b) => a - b);
let s = 20261008;
const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
const coord = () => { const r = rnd();
  return r < 0.6 ? rnd() : r < 0.75 ? (rnd() - 0.5) * 1e-7 : r < 0.85 ? (rnd() - 0.5) * 1e6 : r < 0.95 ? -rnd() : 0; };
const sha = (f) => createHash('sha256').update(readFileSync(dir + '/' + f)).digest('hex');
console.log(JSON.stringify({ oracle: 'faicey face_clone', geometry_sha256: sha('geometry.js'), faceprint_sha256: sha('faceprint.js'), cases: n, indices: used }));
for (let c = 0; c < n; c++) {
  const lms = new Array(478).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));
  const kind = c % 50;
  for (const i of used) lms[i] = kind === 0 ? { x: 0, y: 0, z: 0 } : { x: coord(), y: coord(), z: kind === 1 ? 0 : coord() * 0.2 };
  if (kind === 2) lms[LM.cheekL] = { ...lms[LM.cheekR] };            // a zero face width
  const q = { confidence: rnd() * 1.4 - 0.2, symmetry: symmetry(lms), frontality: rnd() };
  const p = proportions(lms);
  const fp = await measureFaceprint(p, q);
  console.log(JSON.stringify({
    lm: used.map((i) => [String(lms[i].x), String(lms[i].y), String(lms[i].z)]),
    confidence: String(q.confidence), frontality: String(q.frontality), symmetry: String(q.symmetry),
    measures: fp.measuresStr, precisionScore: fp.precisionScore.toString(), hash: fp.hash, canonical: JSON.stringify(fp.payload),
  }));
}
