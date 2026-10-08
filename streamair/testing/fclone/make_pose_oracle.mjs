#!/usr/bin/env node
// Record faicey's pose pipeline as the oracle for src/fclone.rs: Math.atan2 (V8's fdlibm port), frontality,
// poseNormalize, aggregate, and fclone.js's whole capture → faceprint path.
//   node testing/fclone/make_pose_oracle.mjs <faicey/src/face_clone> <ollywoo fclone.js> > tests/fixtures/pose_oracle.json
//
// The inputs are NOT stored: both sides generate them from the same Park–Miller LCG (s = s·16807 mod 2³¹−1; every
// product is below 2⁵³, so JavaScript's doubles compute it exactly, as Rust's integers do), and only digests of the
// outputs' raw f64 bits are recorded (sha256 over little-endian f64s). Rust regenerates, recomputes and compares.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const [dir, fclonePath] = process.argv.slice(2);
const G = await import(dir + '/geometry.js');
const FC = await import(fclonePath);

const M31 = 2147483647;
function lcg(seed) { let s = seed % M31 || 1; return () => (s = (s * 16807) % M31) / M31; }
const digest = (nums) => createHash('sha256').update(Buffer.from(new Float64Array(nums).buffer)).digest('hex');
const bits = (x) => Buffer.from(new Float64Array([x]).buffer).toString('hex');

// atan2: specials (explicit) and 200,000 generated pairs over many magnitudes (digest)
const SPEC = [0, -0, 1, -1, 0.5, -2, 1e-310, -1e-310, 5e-324, 1e308, -1e308, Infinity, -Infinity, NaN, 3.0e-20, 7.5e19, Math.PI, 0.1];
const specials = [];
for (const y of SPEC) for (const x of SPEC) specials.push([bits(y), bits(x), bits(Math.atan2(y, x))]);
const r1 = lcg(1);
const val = (r) => { const m = r(); const e = Math.floor(r() * 80) - 40; const sgn = r() < 0.5 ? -1 : 1; return sgn * m * 2 ** e; };
const atanOut = [];
for (let i = 0; i < 200000; i++) { const y = val(r1), x = val(r1); atanOut.push(Math.atan2(y, x)); }

// frontality over 50,000 generated matrices (entries in [-1.5, 1.5])
const r2 = lcg(2);
const mat = (r) => Array.from({ length: 16 }, () => r() * 3 - 1.5);
const frOut = [];
for (let i = 0; i < 50000; i++) frOut.push(G.frontality(mat(r2)));

// the capture pipeline on generated frame sets
function frames(seed, n, matrixKind) {
  const r = lcg(seed);
  const base = Array.from({ length: 468 }, () => ({ x: r(), y: r(), z: (r() - 0.5) * 0.2 }));
  return Array.from({ length: n }, (_, k) => ({
    landmarks: base.map((p) => ({ x: p.x + (r() - 0.5) * 1e-3, y: p.y + (r() - 0.5) * 1e-3, z: p.z + (r() - 0.5) * 1e-3 })),
    matrix: matrixKind === 'none' ? null : matrixKind === 'identity' ? [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1] : mat(r),
    confidence: 1,
  }));
}
const tess = JSON.parse(readFileSync(new URL('../../tests/fixtures/mediapipe_tessellation.json', import.meta.url))).map(([s, e]) => ({ start: s, end: e }));
const cases = [];
for (let c = 0; c < 60; c++) {
  const n = [1, 2, 5, 12][c % 4], kind = ['matrix', 'identity', 'none'][c % 3], seed = 1000 + c;
  const fr = frames(seed, n, kind);
  const norm = G.poseNormalize(fr[0].landmarks, fr[0].matrix);
  const agg = G.aggregate(fr);
  let fprint = null;
  if (n >= 5) { const out = await FC.fclone(fr, tess, { id: 'oracle' }); fprint = out.faice.fprint.hash; }
  cases.push({ seed, n, kind,
    poseNormalize: digest(norm.flatMap((p) => [p.x, p.y, p.z])),
    aggregate: digest(agg.landmarks.flatMap((p) => [p.x, p.y, p.z])),
    frontalIndex: agg.frontalIndex, meanFrontality: bits(agg.meanFrontality), fprint });
}
const sha = (f) => createHash('sha256').update(readFileSync(f)).digest('hex');
console.log(JSON.stringify({
  oracle: 'faicey face_clone + ollywoo fclone.js under node ' + process.version,
  geometry_sha256: sha(dir + '/geometry.js'), fclone_sha256: sha(fclonePath),
  lcg: 'park-miller 16807 mod 2^31-1; value = s / (2^31-1)',
  atan2: { specials, generated: 200000, seed: 1, digest: digest(atanOut) },
  frontality: { generated: 50000, seed: 2, digest: digest(frOut) },
  cases,
}, null, 1));
