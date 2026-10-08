#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Write the oracle's test audio into .audio/ (gitignored) and check it against testing/pins/audio.sha256.

Every file is 16 kHz, mono, 16-bit PCM. Synthetic signals use integer arithmetic (a 32-bit LCG) or Python's
math.sin rounded to int16, and are pinned by sha256, so a different machine either writes the same bytes or stops.

  jfk          the JFK sample shipped with whisper.cpp (samples/jfk.wav, 11.0 s): real speech
  jfk_x3       jfk three times over (33.0 s): longer than one 30 s window, more than 3000 mel frames of speech
  chirp        2.5 s linear chirp 100 -> 6000 Hz with low noise: energy sweeping every mel band
  noise_loud   1.0 s white noise at full scale, including -32768 and 32767: the clamp and the largest powers
  silence      1.0 s of zeros: every FFT input zero, the log10(1e-10) floor everywhere
  short        0.3 s (4800 samples) 440 Hz: fewer samples than one 30 s pad, n_len_org small
  odd_len      12345 samples of quiet noise: a length that is not a multiple of the 160-sample hop
  min_len      201 samples: the shortest input whisper's reflective pad can read (it reads samples[1..201])
"""
import hashlib, math, os, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".audio")
PIN = os.path.join(ROOT, "testing", "pins", "audio.sha256")
JFK = os.path.join(ROOT, "upstream", "whisper.cpp", "samples", "jfk.wav")
SR = 16000


def wav_bytes(samples):
    data = struct.pack("<%dh" % len(samples), *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, SR, SR * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + \
        b"data" + struct.pack("<I", len(data)) + data


def pcm_of(path):
    b = open(path, "rb").read()
    pos = 12
    while pos + 8 <= len(b):
        cid, ln = b[pos:pos + 4], struct.unpack_from("<I", b, pos + 4)[0]
        if cid == b"data":
            return list(struct.unpack_from("<%dh" % (ln // 2), b, pos + 8))
        pos += 8 + ln + (ln & 1)
    raise SystemExit("no data chunk in " + path)


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next16(self):  # Numerical Recipes LCG, top 16 bits as signed
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        v = self.s >> 16
        return v - 65536 if v >= 32768 else v


def clamp16(x):
    return max(-32768, min(32767, int(round(x))))


def make():
    jfk = pcm_of(JFK)
    sets = {"jfk": jfk, "jfk_x3": jfk * 3}
    r = Lcg(1)
    n = int(2.5 * SR)
    sets["chirp"] = [clamp16(16000 * math.sin(2 * math.pi * (100 * t / SR + (5900 / (2 * 2.5)) * (t / SR) ** 2))
                             + r.next16() / 64) for t in range(n)]
    r = Lcg(2)
    loud = [r.next16() for _ in range(SR)]
    loud[100], loud[101], loud[5000] = -32768, 32767, -32768
    sets["noise_loud"] = loud
    sets["silence"] = [0] * SR
    sets["short"] = [clamp16(8000 * math.sin(2 * math.pi * 440 * t / SR)) for t in range(4800)]
    r = Lcg(3)
    sets["odd_len"] = [r.next16() // 16 for _ in range(12345)]
    r = Lcg(4)
    sets["min_len"] = [r.next16() // 4 for _ in range(201)]
    return sets


def main():
    os.makedirs(OUT, exist_ok=True)
    got = {}
    for name, s in make().items():
        b = wav_bytes(s)
        open(os.path.join(OUT, name + ".wav"), "wb").write(b)
        got[name] = hashlib.sha256(b).hexdigest()
    if "--write-pins" in sys.argv:
        with open(PIN, "w") as f:
            for k, v in got.items():
                f.write("%s  %s.wav\n" % (v, k))
        print("pinned", len(got), "files")
        return
    want = {}
    for line in open(PIN):
        h, f = line.split()
        want[f[:-4]] = h
    bad = [k for k in got if want.get(k) != got[k]]
    if bad or set(want) != set(got):
        raise SystemExit("audio does not match its pins: " + ", ".join(bad or sorted(set(want) ^ set(got))))
    print("audio ok:", len(got), "files match testing/pins/audio.sha256")


if __name__ == "__main__":
    main()
