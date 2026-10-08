#!/usr/bin/env node
// Record faicey's persona.js personaPrint() as the oracle for src/fclone.rs persona_print().
//   node testing/fclone/make_persona_oracle.mjs <faicey/src/face_clone> > tests/fixtures/persona_oracle.json
// Face prints come from faceprint.js on generated proportions; voice prints are given in both shapes persona.js
// accepts (forensic: hash + measuresStr + precisionScore; and a dvscope/1 vprint's wei strings with no precision,
// which persona.js reads as precision 0). Face only, voice only, both, and neither (which must throw).
import { createHash } from 'node:crypto';
const dir = process.argv[2];
const { personaPrint } = await import(dir + '/persona.js');
const { measureFaceprint } = await import(dir + '/faceprint.js');
const M31 = 2147483647; let s = 77; const r = () => (s = (s * 16807) % M31) / M31;
const hex = () => '0x' + createHash('sha256').update(String(r())).digest('hex');
const cases = [];
for (let c = 0; c < 200; c++) {
  const props = Object.fromEntries(['faceAspect','jawRatio','cheekRatio','interocularRatio','eyeWidthRatio','noseLengthRatio','noseWidthRatio','mouthWidthRatio','lipHeightRatio','philtrumRatio','browEyeRatio','chinRatio'].map((k) => [k, r() * 2]));
  const fp = await measureFaceprint(props, { confidence: r(), symmetry: r(), frontality: r() });
  const face = { hash: fp.hash, measuresStr: fp.measuresStr, precisionScore: fp.precisionScore.toString() };
  const forensic = { hash: hex(), measuresStr: Array.from({ length: 6 }, () => String(Math.floor(r() * 1e15) * 1000)),
                     precisionScore: String(Math.floor(r() * 1e9) * 1e9) };
  const vprint = { hash: hex().slice(2), m: Array.from({ length: 8 }, () => String(Math.floor(r() * 1e15))) };
  const which = c % 4;
  const parts = which === 0 ? { face } : which === 1 ? { voice: forensic } : which === 2 ? { face, voice: forensic } : { face, voice: vprint };
  const p = await personaPrint(parts);
  cases.push({ parts, hash: p.hash, modalities: p.modalities, precisionScore: p.registerArgs.precisionScore, measures: p.registerArgs.m.length });
}
let threw = false; try { await personaPrint({}); } catch { threw = true; }
console.log(JSON.stringify({ oracle: 'faicey persona.js', neither_throws: threw, cases }));
