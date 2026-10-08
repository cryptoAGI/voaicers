#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The adversarial files of the 0.0.4 oracle: one corruption each of a pinned good file. tests/opus.rs makes the same
bytes from the same rules (and checks their sha256 against testing/opus/mutations.jsonl, so the files the reference
saw are the files voaice.rs refuses). Pages are counted from 0; "re-CRC" means the page's CRC is recomputed so that
only the named check can catch the change.
usage: mutate.py <files dir> <out dir>   — prints one JSON line per mutation: name, base, sha256, expected kind"""
import hashlib, json, os, struct, sys

TABLE = []
for i in range(256):
    r = i << 24
    for _ in range(8):
        r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
    TABLE.append(r & 0xFFFFFFFF)


def crc(b):
    c = 0
    for x in b:
        c = ((c << 8) & 0xFFFFFFFF) ^ TABLE[((c >> 24) ^ x) & 0xFF]
    return c


def pages(f):
    out, i = [], 0
    while i < len(f):
        n = f[i + 26]
        ln = 27 + n + sum(f[i + 27:i + 27 + n])
        out.append((i, ln))
        i += ln
    return out


def recrc(f, at, ln):
    f[at + 22:at + 26] = b"\0\0\0\0"
    f[at + 22:at + 26] = struct.pack("<I", crc(bytes(f[at:at + ln])))


def granule(f, at):
    return struct.unpack_from("<Q", f, at + 6)[0]


def set_granule(f, at, ln, g):
    f[at + 6:at + 14] = struct.pack("<Q", g & 0xFFFFFFFFFFFFFFFF)
    recrc(f, at, ln)


A = "e_jfk_m_20ms_24k.opus"
B = "e_min_len_24k.opus"


def mutations(load):
    a = load(A)
    P = pages(a)
    (p3, l3), (p4, l4), (pl, ll) = P[3], P[4], P[-1]
    out = []
    def m(name, base, kind, f):
        out.append((name, base, kind, bytes(f)))
    f = bytearray(a); f[p3 + 22] ^= 0x01; m("m_crc_field", A, "Crc", f)
    f = bytearray(a); f[p3 + 27 + a[p3 + 26] + 100] ^= 0x80; m("m_body_bit", A, "Crc", f)
    m("m_truncated_mid_page", A, "TruncatedPage", a[:pl + ll // 2])
    m("m_truncated_last_byte", A, "TruncatedPage", a[:-1])
    m("m_truncated_header", A, "TruncatedPage", a[:pl + 10])
    m("m_no_eos", A, "NoEndOfStream", a[:pl])
    m("m_dropped_page", A, "Sequence", a[:p4] + a[p4 + l4:])
    m("m_duplicated_page", A, "Sequence", a[:p4 + l4] + a[p4:])
    f = bytearray(a); f[p3 + 3] = ord("X"); m("m_capture", A, "CapturePattern", f)
    f = bytearray(a); f[p3 + 4] = 1; recrc(f, p3, l3); m("m_version", A, "Version", f)
    f = bytearray(a); set_granule(f, p4, l4, granule(a, p3) - 960); m("m_granule_backwards", A, "GranuleBackwards", f)
    f = bytearray(a); set_granule(f, p4, l4, granule(a, p4) + 960); m("m_granule_mismatch", A, "GranuleMismatch", f)
    f = bytearray(a); set_granule(f, pl, ll, granule(a, pl) + 5760); m("m_granule_beyond_eos", A, "GranuleBeyondSamples", f)
    f = bytearray(a); f[p3 + 14] ^= 1; recrc(f, p3, l3); m("m_serial", A, "Serial", f)
    f = bytearray(a); f[p3 + 5] |= 0x02; recrc(f, p3, l3); m("m_bos_again", A, "HeaderFlags", f)
    f = bytearray(a); f[p3 + 5] ^= 0x01; recrc(f, p3, l3); m("m_continued_flag", A, "Continuation", f)
    f = bytearray(a); f[pl + 5] &= ~0x04; recrc(f, pl, ll); m("m_eos_removed", A, "GranuleMismatch", f)  # its trimmed granule is now mid-stream
    f = bytearray(a); f[P[0][0] + 28 + 7] = ord("X"); recrc(f, *P[0]); m("m_head_magic", A, "OpusHead", f)
    f = bytearray(a); f[P[1][0] + 27 + a[P[1][0] + 26] + 7] = ord("X"); recrc(f, *P[1]); m("m_tags_magic", A, "OpusTags", f)
    m("m_appended_stream", A, "DataAfterEnd", a + a)
    f = bytearray(a)
    for at, ln in P[2:]:
        if granule(a, at) != 0xFFFFFFFFFFFFFFFF:
            set_granule(f, at, ln, granule(a, at) + 48000)
    m("m_start_offset", A, "ok", f)
    b = load(B)
    f = bytearray(b); f[28 + 10:28 + 12] = struct.pack("<H", 2000); recrc(f, *pages(b)[0]); m("m_preskip_over", B, "PreSkip", f)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for name, base, kind, data in mutations(lambda n: open(os.path.join(src, n), "rb").read()):
        open(os.path.join(dst, name + ".opus"), "wb").write(data)
        print(json.dumps(dict(name=name, base=base, expect=kind, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)), sort_keys=True))


main()
