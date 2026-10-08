// SPDX-License-Identifier: MIT OR Apache-2.0
//
// resample_oracle — records the samples whisper-cli's own audio reader produces (0.0.5), so voaice.rs's resampler
// can be compared to it bit for bit. It links the pinned build's examples/libcommon.a — the very object whisper-cli is
// linked with (common-whisper.cpp compiled -O3 -DNDEBUG -fPIC, miniaudio 0.11.24 inside it) — and calls
// read_audio_data(path, pcmf32, pcmf32s, /*stereo=*/false), as whisper-cli does without --diarize. Nothing here
// re-implements miniaudio.
//
//   resample_oracle <outdir> <wav ...>          per wav: <outdir>/<stem>.f32 (the vector, little-endian f32) and a
//                                               line "<stem> <samples>" on stdout
//   resample_oracle --bench <wav>               one measurement, the same keys as `voaice bench-resample`
//
// --bench, in a fresh process: the heap bytes live at the first call's peak (malloc/calloc/realloc/free interposed
// below, because miniaudio allocates with malloc, not operator new), the peak RSS of the first call (VmHWM after
// resetting it through /proc/self/clear_refs, minus VmRSS before), the best wall time of 10 calls, and CPU time per
// call (/proc/self/stat utime + stime) over a loop of at least 1 s. Each call is the whole read_audio_data: open,
// parse, convert, mix, resample, the vector out.
#include "common-whisper.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>
#include <vector>

// ---- heap accounting: glibc's allocator wrapped, sizes from malloc_usable_size (what the block really holds)
extern "C" void * __libc_malloc(size_t);
extern "C" void * __libc_calloc(size_t, size_t);
extern "C" void * __libc_realloc(void *, size_t);
extern "C" void   __libc_free(void *);
static std::atomic<size_t> g_live{0}, g_peak{0};
static void count_up(void * p) {
    if (!p) return;
    const size_t now = g_live.fetch_add(malloc_usable_size(p)) + malloc_usable_size(p);
    size_t pk = g_peak.load();
    while (now > pk && !g_peak.compare_exchange_weak(pk, now)) {}
}
extern "C" void * malloc(size_t n) { void * p = __libc_malloc(n); count_up(p); return p; }
extern "C" void * calloc(size_t a, size_t b) { void * p = __libc_calloc(a, b); count_up(p); return p; }
extern "C" void free(void * p) { if (p) g_live.fetch_sub(malloc_usable_size(p)); __libc_free(p); }
extern "C" void * realloc(void * p, size_t n) {
    const size_t old = p ? malloc_usable_size(p) : 0;
    void * q = __libc_realloc(p, n);
    if (q) { g_live.fetch_sub(old); count_up(q); } // on failure the old block is still live, unchanged
    return q;
}

static double cpu_seconds() {
    FILE * f = std::fopen("/proc/self/stat", "r");
    if (!f) return -1;
    char buf[1024];
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char * p = std::strrchr(buf, ')');
    if (!p) return -1;
    p += 2;
    unsigned long long ut = 0, st = 0;
    for (int field = 3; field <= 15 && *p; field++) {
        if (field == 14) ut = std::strtoull(p, nullptr, 10);
        if (field == 15) st = std::strtoull(p, nullptr, 10);
        p = std::strchr(p, ' ');
        if (!p) break;
        p++;
    }
    return (double)(ut + st) / 100.0;
}

static long status_kb(const char * key) {
    FILE * f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, std::strlen(key)) == 0) { v = std::atol(line + std::strlen(key)); break; }
    }
    std::fclose(f);
    return v;
}

static bool reset_peak_rss() {
    FILE * f = std::fopen("/proc/self/clear_refs", "w");
    if (!f) return false;
    bool ok = std::fputs("5", f) >= 0;
    return std::fclose(f) == 0 && ok;
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool read_one(const char * path, std::vector<float> & pcm) {
    std::vector<std::vector<float>> pcms;
    return read_audio_data(std::string(path), pcm, pcms, false);
}

static int bench(const char * wav) {
    size_t samples = 0;
    auto call = [&]() {
        std::vector<float> pcm;
        if (!read_one(wav, pcm)) { std::fprintf(stderr, "read_audio_data failed: %s\n", wav); std::exit(1); }
        samples = pcm.size();
    };
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    const size_t live0 = g_live.load();
    g_peak.store(live0);
    call();
    const size_t heap_peak = g_peak.load() - live0;
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        const double t0 = now_ms();
        call();
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { call(); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-resample-reference samples %zu heap_peak_kb %zu wall_best_ms %.3f cpu_ms_per_call %.3f cpu_reps %d "
                "rss_peak_delta_kb %ld rss_peak_kb %ld\n",
                samples, (heap_peak + 1023) / 1024, best, (c1 - c0) * 1000.0 / reps, reps, peak >= 0 ? peak - before : -1, peak);
    return 0;
}

int main(int argc, char ** argv) {
    // read_audio_data narrates on stderr; the record keeps stdout
    if (argc == 3 && std::strcmp(argv[1], "--bench") == 0) {
        std::freopen("/dev/null", "w", stderr);
        return bench(argv[2]);
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: resample_oracle <outdir> <wav ...> | --bench <wav>\n");
        return 2;
    }
    std::freopen("/dev/null", "w", stderr);
    const std::string out = argv[1];
    for (int i = 2; i < argc; i++) {
        std::vector<float> pcm;
        std::string stem = argv[i];
        stem = stem.substr(stem.find_last_of('/') + 1);
        stem = stem.substr(0, stem.rfind('.'));
        if (!read_one(argv[i], pcm)) { std::printf("%s FAILED\n", stem.c_str()); return 1; }
        FILE * f = std::fopen((out + "/" + stem + ".f32").c_str(), "wb");
        if (!f || std::fwrite(pcm.data(), sizeof(float), pcm.size(), f) != pcm.size() || std::fclose(f) != 0) {
            std::printf("%s WRITE FAILED\n", stem.c_str());
            return 1;
        }
        std::printf("%s %zu\n", stem.c_str(), pcm.size());
    }
    return 0;
}
