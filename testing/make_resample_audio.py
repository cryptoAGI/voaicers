#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Write 0.0.5's resampler corpus into .audio/resample/ (gitignored) and check it against testing/pins/resample.sha256.

Every rate whisper-cli is likely to be handed, mono and stereo, s16 and f32, plus the other WAV sample formats
miniaudio's dr_wav converts (u8, s24, s32), more than two channels (a WAVE_FORMAT_EXTENSIBLE 5.1 file), f32 values
outside [-1, 1] and subnormal, inputs of 1..7 frames (the resampler's start-up and the length rule's edges), a LIST
chunk before the data, and real speech (whisper.cpp's jfk.wav held to 48 kHz). Signals are integer arithmetic (a
32-bit LCG) or Python's math.sin rounded to the sample format, so a different machine either writes the same bytes or
stops.

  r<rate>_<m|s>_<s16|f32>   1.3 s + a few frames: a chirp from 60 Hz to 0.45 x the rate (every resampler band) with
                            low noise; stereo's right channel is a 440 Hz sine plus louder noise (so L != R)
  r<rate>_<..>_<u8|s24|s32> the same signal in the other integer formats
  r48000_c6_s16_ext         six channels, WAVE_FORMAT_EXTENSIBLE (mask 0x3F): the mixdown's six-term sum
  r44100_m_f32_hot          f32 beyond full scale (+-4), subnormals, -0.0, exact +-1
  r<rate>_m_s16_n<k>        k frames only
  r48000_s_s16_list         a LIST chunk (odd length, padded) between fmt and data
  r48000_m_s16_trunc        a data chunk that claims 48,000 frames and holds 30,000 (a cut-off upload)
  jfk_48k                   jfk.wav (16 kHz speech) with each sample held three times: 11 s of real speech at 48 kHz
  bench_44k1_s_60s          60 s of stereo s16 at 44.1 kHz, for the efficiency step only
"""
import hashlib, math, os, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".audio", "resample")
PIN = os.path.join(ROOT, "testing", "pins", "resample.sha256")
JFK = os.path.join(ROOT, "upstream", "whisper.cpp", "samples", "jfk.wav")
RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next16(self):  # Numerical Recipes LCG, top 16 bits as signed
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        v = self.s >> 16
        return v - 65536 if v >= 32768 else v


def signal(rate, n, seed):
    """Two channels of values in [-1, 1): the chirp (left) and the sine (right), each with its noise."""
    r = Lcg(seed)
    f1 = 0.45 * rate
    dur = n / rate
    left, right = [], []
    for t in range(n):
        s = t / rate
        ph = 2 * math.pi * (60 * s + (f1 - 60) / (2 * dur) * s * s)
        left.append(0.5 * math.sin(ph) + r.next16() / 65536 / 16)
        right.append(0.3 * math.sin(2 * math.pi * 440 * s) + r.next16() / 65536 / 4)
    return left, right


def q16(x):
    return max(-32768, min(32767, int(round(x * 32768))))


def enc(fmt, x):
    if fmt == "u8":
        return struct.pack("<B", max(0, min(255, int(round(x * 128)) + 128)))
    if fmt == "s16":
        return struct.pack("<h", q16(x))
    if fmt == "s24":
        v = max(-(1 << 23), min((1 << 23) - 1, int(round(x * (1 << 23)))))
        return struct.pack("<i", v)[:3]
    if fmt == "s32":
        v = max(-(1 << 31), min((1 << 31) - 1, int(round(x * (1 << 31)))))
        return struct.pack("<i", v)
    if fmt == "f32":
        return struct.pack("<f", x)
    raise ValueError(fmt)


BITS = {"u8": 8, "s16": 16, "s24": 24, "s32": 32, "f32": 32}


def wav(rate, chans, fmt, frames, extensible=False, extra=b""):
    """frames: list of per-frame tuples of floats. A plain WAVEFORMAT (16-byte fmt) unless `extensible`."""
    data = b"".join(enc(fmt, x) for fr in frames for x in fr)
    bits = BITS[fmt]
    align = chans * bits // 8
    tag = 3 if fmt == "f32" else 1
    if extensible:
        guid = struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        body = struct.pack("<HHIIHH", 0xFFFE, chans, rate, rate * align, align, bits) + \
            struct.pack("<HHI", 22, bits, 0x3F) + guid
    else:
        body = struct.pack("<HHIIHH", tag, chans, rate, rate * align, align, bits)
    fmtc = b"fmt " + struct.pack("<I", len(body)) + body
    datac = b"data" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    riff = b"WAVE" + fmtc + extra + datac
    return b"RIFF" + struct.pack("<I", len(riff)) + riff


def pcm_of(path):
    b = open(path, "rb").read()
    pos = 12
    while pos + 8 <= len(b):
        cid, ln = b[pos:pos + 4], struct.unpack_from("<I", b, pos + 4)[0]
        if cid == b"data":
            return list(struct.unpack_from("<%dh" % (ln // 2), b, pos + 8))
        pos += 8 + ln + (ln & 1)
    raise SystemExit("no data chunk in " + path)


def make():
    files = {}
    for i, rate in enumerate(RATES):
        n = int(1.3 * rate) + 7 + i  # not a multiple of any reduced input rate
        left, right = signal(rate, n, 10 + i)
        for fmt in ("s16", "f32"):
            files["r%d_m_%s" % (rate, fmt)] = wav(rate, 1, fmt, [(x,) for x in left])
            files["r%d_s_%s" % (rate, fmt)] = wav(rate, 2, fmt, list(zip(left, right)))
        if rate in (22050, 44100, 48000):
            for fmt in ("u8", "s24", "s32"):
                lay = "s" if rate == 44100 else "m"
                frames = list(zip(left, right)) if lay == "s" else [(x,) for x in left]
                files["r%d_%s_%s" % (rate, lay, fmt)] = wav(rate, 2 if lay == "s" else 1, fmt, frames)
    left, right = signal(48000, 48000 + 11, 30)
    six = [(a, b, 0.5 * a - 0.25 * b, a * b, -a, 0.125 * b) for a, b in zip(left, right)]
    files["r48000_c6_s16_ext"] = wav(48000, 6, "s16", six, extensible=True)
    hot, _ = signal(44100, 44100 + 3, 31)
    hot = [4 * x for x in hot]
    hot[10:16] = [1.0, -1.0, -0.0, 1e-40, -3e-39, 2.0 ** -149]
    hot[2000:2003] = [1e-42, -1e-42, 0.0]
    files["r44100_m_f32_hot"] = wav(44100, 1, "f32", [(x,) for x in hot])
    r = Lcg(32)
    for rate, ks in ((48000, (1, 2, 3, 4, 5)), (44100, (1, 2, 7)), (8000, (1, 2, 3)), (16000, (1,))):
        for k in ks:
            files["r%d_m_s16_n%d" % (rate, k)] = wav(rate, 1, "s16", [(r.next16() / 32768,) for _ in range(k)])
    left, right = signal(48000, 24000 + 5, 33)
    lst = b"LIST" + struct.pack("<I", 17) + b"INFOISFT" + struct.pack("<I", 5) + b"voais" + b"\0"  # odd: padded
    files["r48000_s_s16_list"] = wav(48000, 2, "s16", list(zip(left, right)), extra=lst)
    left, _ = signal(48000, 30000, 35)
    full = wav(48000, 1, "s16", [(x,) for x in left])
    files["r48000_m_s16_trunc"] = full[:40] + struct.pack("<I", 2 * 48000) + full[44:]  # data size says 48,000 frames
    jfk = pcm_of(JFK)
    files["jfk_48k"] = wav(48000, 1, "s16", [(s / 32768,) for s in jfk for _ in range(3)])
    left, right = signal(44100, 60 * 44100, 34)
    files["bench_44k1_s_60s"] = wav(44100, 2, "s16", list(zip(left, right)))
    return files


def main():
    os.makedirs(OUT, exist_ok=True)
    got = {}
    for name, b in make().items():
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
        raise SystemExit("resampler audio does not match its pins: " + ", ".join(bad or sorted(set(want) ^ set(got))))
    print("resampler audio ok:", len(got), "files match testing/pins/resample.sha256")


if __name__ == "__main__":
    main()
