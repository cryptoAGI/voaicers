#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The inputs opusenc encodes for the 0.0.4 oracle, derived from the pinned WAVs in .audio/ (testing/make_audio.py):
the eight mono files as they are, plus a stereo file (JFK left, the chirp right), a six-channel file (mapping family 1:
six slices of JFK) and a PNG (random pixels, incompressible: about 50 KB, so OpusTags with the picture is longer than
one page's 65,025 body bytes). Deterministic: integer arithmetic and a fixed LCG; out dir as the first argument."""
import os, struct, sys, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO = os.path.join(ROOT, ".audio")


def pcm(name):
    b = open(os.path.join(AUDIO, name + ".wav"), "rb").read()
    pos = 12
    while pos + 8 <= len(b):
        cid, ln = b[pos:pos + 4], struct.unpack_from("<I", b, pos + 4)[0]
        if cid == b"data":
            return list(struct.unpack_from("<%dh" % (ln // 2), b, pos + 8))
        pos += 8 + ln + (ln & 1)


def wav(path, chans, rate=16000):
    n = len(chans[0])
    inter = [chans[c][i] for i in range(n) for c in range(len(chans))]
    data = struct.pack("<%dh" % len(inter), *inter)
    k = len(chans)
    fmt = struct.pack("<HHIIHH", 1, k, rate, rate * 2 * k, 2 * k, 16)
    open(path, "wb").write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt +
                           b"data" + struct.pack("<I", len(data)) + data)


def png(path, w, h):
    s = 12345
    rows = b""
    for _ in range(h):
        row = bytearray([0])
        for _ in range(w * 3):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            row.append(s >> 23)
        rows += bytes(row)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for n in ["jfk", "jfk_x3", "chirp", "noise_loud", "silence", "short", "odd_len", "min_len"]:
        open(os.path.join(out, n + ".wav"), "wb").write(open(os.path.join(AUDIO, n + ".wav"), "rb").read())
    jfk, chirp = pcm("jfk"), pcm("chirp")
    wav(os.path.join(out, "stereo.wav"), [jfk[:len(chirp)], chirp])
    wav(os.path.join(out, "six.wav"), [jfk[16000 * c:16000 * (c + 1)] for c in range(6)])
    png(os.path.join(out, "art.png"), 128, 128)


main()
