#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The 0.0.4 reference, run where opus-tools is (mindX production: opus-tools 0.2, libopus 1.4, libogg 1.3.5).

For each .opus file named on the command line, one JSON line of what the reference says about it:
  opusinfo   its fields as printed (pre-skip, gain, channels, original rate, vendor, comments, playback length,
             total data length) and how many warning/error lines it printed
  opusdec    `opusdec --rate 48000 f f.wav`: its exit status and the WAV's frame count (from the data chunk) — the
             exact number of 48 kHz samples the reference plays, pre-skip and end trimming applied
  libogg     the library both tools read with, through ctypes: pages and packets as ogg_sync_pageout /
             ogg_stream_packetout give them, an FNV-1a digest of every page's (sequence, granule, flags) and of every
             audio packet's (bytes, samples), the samples counted by libopus's own opus_packet_get_nb_samples, and
             the pages it lost to resynchronisation (bad CRC or capture) and the holes it reported
Nothing here writes outside the directory the files are in; nothing is installed.
"""
import ctypes, hashlib, json, os, re, struct, subprocess, sys

ogg = ctypes.CDLL("libogg.so.0")
opus = ctypes.CDLL("libopus.so.0")


class Page(ctypes.Structure):
    _fields_ = [("header", ctypes.POINTER(ctypes.c_ubyte)), ("header_len", ctypes.c_long),
                ("body", ctypes.POINTER(ctypes.c_ubyte)), ("body_len", ctypes.c_long)]


class Packet(ctypes.Structure):
    _fields_ = [("packet", ctypes.POINTER(ctypes.c_ubyte)), ("bytes", ctypes.c_long), ("b_o_s", ctypes.c_long),
                ("e_o_s", ctypes.c_long), ("granulepos", ctypes.c_int64), ("packetno", ctypes.c_int64)]


ogg.ogg_sync_buffer.restype = ctypes.POINTER(ctypes.c_char)
ogg.ogg_page_granulepos.restype = ctypes.c_int64
ogg.ogg_page_serialno.restype = ctypes.c_int
ogg.ogg_page_pageno.restype = ctypes.c_long
opus.opus_packet_get_nb_samples.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.c_int32]
opus.opus_get_version_string.restype = ctypes.c_char_p

FNV_OFFSET, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3


def fnv(h, b):
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def libogg_walk(path):
    data = open(path, "rb").read()
    oy = ctypes.create_string_buffer(256)
    os_ = ctypes.create_string_buffer(16384)
    ogg.ogg_sync_init(oy)
    buf = ogg.ogg_sync_buffer(oy, ctypes.c_long(len(data) + 1))
    ctypes.memmove(buf, data, len(data))
    ogg.ogg_sync_wrote(oy, ctypes.c_long(len(data)))
    pg, pk = Page(), Packet()
    r = dict(pages=0, sync_lost=0, other_serial=0, holes=0, packets=0, audio_packets=0, audio_bytes=0,
             libopus_samples=0, bad_toc=0, page_fnv=FNV_OFFSET, packet_fnv=FNV_OFFSET, eos_pages=0, last_granule=None)
    serial = None
    while True:
        k = ogg.ogg_sync_pageout(oy, ctypes.byref(pg))
        if k == 0:
            break
        if k < 0:
            r["sync_lost"] += 1
            continue
        s = ogg.ogg_page_serialno(ctypes.byref(pg)) & 0xFFFFFFFF
        if serial is None:
            serial = s
            ogg.ogg_stream_init(os_, ctypes.c_int(s))
        elif s != serial:
            r["other_serial"] += 1
            continue
        r["pages"] += 1
        g = ogg.ogg_page_granulepos(ctypes.byref(pg)) & 0xFFFFFFFFFFFFFFFF
        seq = ogg.ogg_page_pageno(ctypes.byref(pg)) & 0xFFFFFFFF
        flags = pg.header[5]
        r["page_fnv"] = fnv(r["page_fnv"], struct.pack("<IQB", seq, g, flags))
        if ogg.ogg_page_eos(ctypes.byref(pg)):
            r["eos_pages"] += 1
            r["last_granule"] = g
        ogg.ogg_stream_pagein(os_, ctypes.byref(pg))
        while True:
            k = ogg.ogg_stream_packetout(os_, ctypes.byref(pk))
            if k == 0:
                break
            if k < 0:
                r["holes"] += 1
                continue
            r["packets"] += 1
            if r["packets"] <= 2:
                continue  # OpusHead, OpusTags
            n = pk.bytes
            ns = opus.opus_packet_get_nb_samples(pk.packet, n, 48000) if n > 0 else -1
            if ns < 0:
                r["bad_toc"] += 1
                ns = 0
            r["audio_packets"] += 1
            r["audio_bytes"] += n
            r["libopus_samples"] += ns
            r["packet_fnv"] = fnv(r["packet_fnv"], struct.pack("<QI", n, ns))
    r["page_fnv"] = "%016x" % r["page_fnv"]
    r["packet_fnv"] = "%016x" % r["packet_fnv"]
    return r


def wav_frames(path):
    b = open(path, "rb").read()
    ch = struct.unpack_from("<H", b, 22)[0]
    bits = struct.unpack_from("<H", b, 34)[0]
    pos = 12
    while pos + 8 <= len(b):
        cid, ln = b[pos:pos + 4], struct.unpack_from("<I", b, pos + 4)[0]
        if cid == b"data":
            ln = min(ln, len(b) - pos - 8)
            return ln // (ch * bits // 8), ch
        pos += 8 + ln + (ln & 1)
    return None, ch


def opusinfo(path):
    p = subprocess.run(["opusinfo", path], capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    f = dict(exit=p.returncode, warnings=sum(1 for l in out.splitlines() if re.search(r"warn|error|hole|corrupt|invalid", l, re.I)))
    for key, pat in [("pre_skip", r"Pre-skip: (\d+)"), ("gain", r"Playback gain: (.*)"), ("channels", r"Channels: (\d+)"),
                     ("input_rate", r"Original sample rate: (\d+) Hz"), ("playback_length", r"Playback length: (\S+)"),
                     ("data_length", r"Total data length: (\d+) bytes"), ("vendor", r"Encoded with (.*)"),
                     ("packet_duration", r"Packet duration: (.*)")]:
        m = re.search(pat, out)
        f[key] = (int(m.group(1)) if m.group(1).isdigit() else m.group(1).strip()) if m else None
    m = re.search(r"User comments section follows\.\.\.\n((?:\t.*\n)*)", out)
    f["comments"] = [l[1:] for l in m.group(1).splitlines()] if m else []
    f["warning_lines"] = [l.strip() for l in out.splitlines() if re.search(r"warn|error|hole|corrupt|invalid", l, re.I)][:6]
    return f


def opusdec(path):
    wav = path + ".dec.wav"
    p = subprocess.run(["opusdec", "--quiet", "--rate", "48000", path, wav], capture_output=True, text=True, errors="replace")
    frames, ch = (wav_frames(wav) if os.path.exists(wav) else (None, None))
    if os.path.exists(wav):
        os.remove(wav)
    return dict(exit=p.returncode, frames=frames, channels=ch, stderr=p.stderr.strip()[:300])


def main():
    for path in sys.argv[1:]:
        d = open(path, "rb").read()
        rec = dict(file=os.path.basename(path), sha256=hashlib.sha256(d).hexdigest(), bytes=len(d),
                   opusinfo=opusinfo(path), opusdec=opusdec(path), libogg=libogg_walk(path))
        print(json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--versions"]:
        print(json.dumps(dict(libopus=opus.opus_get_version_string().decode(),
                              opus_tools=subprocess.run(["opusinfo", "-V"], capture_output=True, text=True).stdout.strip())))
    else:
        main()
