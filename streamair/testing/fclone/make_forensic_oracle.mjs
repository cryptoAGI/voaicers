#!/usr/bin/env node
// Record voaice's Forensic.voiceprint() (the print ollywoo's /voicey/measure returns) as the oracle for
// streamair's forensic_print(): the per-feature means as String(), framesUsed, sampleRate, and the hash.
//   node testing/fclone/make_forensic_oracle.mjs <mindX/voaice/src> <wav>... > tests/fixtures/forensic_oracle.json
// The DSP that produces the features (VAD, analyser, pitch) is not ported yet; this pins the encoding and the hash
// over them, at several sample rates and frame budgets.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const [src, ...wavs] = process.argv.slice(2);
const { Forensic } = await import(src + '/Forensic.js');
const { decodeWav } = await import(src + '/audio/wav.js');
const out = [];
for (const w of wavs) {
  const d = decodeWav(readFileSync(w));
  for (const maxFrames of [400, 40, 0]) {
    const f = new Forensic({ sampleRate: d.sampleRate, maxFrames });
    const p = f.voiceprint(Float32Array.from(d.samples));
    out.push({ wav: w.split('/').pop(), sampleRate: d.sampleRate, maxFrames,
      features: Object.fromEntries(Object.entries(p.features).map(([k, v]) => [k, String(v)])),
      framesUsed: p.framesUsed, measures: p.measuresStr, hash: p.hash });
  }
}
const sha = (f) => createHash('sha256').update(readFileSync(f)).digest('hex');
console.log(JSON.stringify({ oracle: 'voaice Forensic.voiceprint', forensic_sha256: sha(src + '/Forensic.js'),
  scientific_sha256: sha(src + '/Scientific.js'), cases: out }, null, 1));
