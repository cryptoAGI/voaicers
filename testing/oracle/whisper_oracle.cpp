// SPDX-License-Identifier: MIT OR Apache-2.0
//
// whisper_oracle — records what the SHIPPED whisper.cpp library computes, so voaice.rs can be compared to it bit
// for bit. It links the pinned libwhisper.so / libggml*.so (upstream/whisper.cpp/build/bin) and calls them
// in-process; nothing here re-implements whisper.cpp arithmetic.
//
//   whisper_oracle <model.bin> <outdir> [wav ...]
//   whisper_oracle --bench-mel <model.bin> <wav> <threads>
//   whisper_oracle --f16 <outdir>                 (0.0.3) the f32<->f16 conversions and GELU of the shipped libggml*
//   whisper_oracle --bench-f16 <what>             (0.0.3) one measurement of them, see bench_f16
//
// --bench-mel measures the reference's whisper_pcm_to_mel_with_state the way `voaice bench-mel` measures voaice's,
// in a fresh process each: the heap bytes live at the first call's peak (operator new counted, below), peak RSS of
// the first call (VmHWM after resetting it through /proc/self/clear_refs, minus
// VmRSS before), the best wall time of 10 calls, and CPU time per call (/proc/self/stat utime + stime, all threads)
// over a loop of at least 1 s. One line, the same keys as voaice's.
//
// writes
//   <outdir>/model.tsv     hparams (public getters), then one line per tensor the loader holds:
//                          name, ggml type, ne0..ne3, nbytes, sha256 of the bytes in tensor->data (what the loader
//                          actually put in memory, read back through ggml_backend_tensor_get)
//   <outdir>/vocab.tsv     id, hex bytes of whisper_token_to_str(id) for every id < n_vocab
//   <outdir>/filters.f32   the mel filterbank whisper_model::filters holds (n_mel*n_fft little-endian f32)
//   per wav, <outdir>/<stem>/
//     pcm.f32              the f32 samples fed to whisper (s16 * 2^-15, as miniaudio converts in whisper-cli)
//     mel.f32              whisper_state::mel.data after whisper_pcm_to_mel_with_state, layout [n_mel][n_len]
//     mel.tsv              n_mel, n_len, n_len_org, the threads compared and whether they agreed bit for bit
//     transcript.tsv       whisper_full (greedy, see params below) — per token: segment, index, id, t0, t1,
//                          p as f32 bits, text as hex; per segment: t0, t1, text
//
// --f16 writes (0.0.3; nothing of whisper is loaded, only libggml-base / libggml-cpu are called):
//   f16_to_f32.{base,table,cpu_row,cpu_tail}.u32  all 65,536 f16 patterns widened by ggml_fp16_to_fp32 (libggml-base),
//                          read from ggml_table_f32_f16 (exported; filled by ggml_cpu_init), by ggml_cpu_fp16_to_fp32
//                          over the whole array (its F16C blocks) and one value per call (its scalar tail)
//   f32_inputs.u32         the boundary set: every f16 value, its f32 neighbours, every halfway point between adjacent
//                          f16 values and its neighbours, the overflow and underflow edges, subnormals, infinities,
//                          NaN payloads, +-10 and its neighbours, then 2^20 xorshift patterns (a multiple of 8)
//   f32_to_f16.{base,cpu_row,cpu_tail}.u16  those inputs narrowed by ggml_fp32_to_fp16 (libggml-base), by
//                          ggml_cpu_fp32_to_fp16 over the whole array (F16C blocks) and three values per call (the
//                          scalar tail, i.e. ggml-cpu's own inlined GGML_CPU_FP32_TO_FP16)
//   f32_to_f16.{base,cpu_row,cpu_tail}.digest  ALL 2^32 f32 patterns, narrowed the same three ways: per chunk of
//                          65,536 (chunk c = the high 16 bits), a 64-bit FNV-1a over the outputs packed four to a
//                          u64 (65,536 u64 little-endian per file)
//   gelu_table.u16         ggml_table_gelu_f16, all 65,536 entries (exported)
//   gelu.u32               ggml_gelu on f32_inputs then all 65,536 f16 values widened, through a ggml graph computed
//                          by the shipped CPU backend (ggml_graph_compute_with_ctx, 1 thread)
//   gelu.digest            ggml_gelu on ALL 2^32 f32 patterns, per chunk of 65,536, outputs packed two to a u64
//   f16.tsv                counts, F16C present, from_float of F16 == ggml_cpu_fp32_to_fp16, gelu 1 vs 4 threads
//
// The internals (mel, filters, tensor map) are read at offsets the layout probe computed from the same source with
// the same compiler (layout.h); each offset is self-checked against a public getter before use.
#include "whisper.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "layout.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <atomic>
#include <malloc.h>
#include <new>
#include <chrono>
#include <algorithm>

// exported data symbols of libggml-cpu.so (nm -D: B), filled by ggml_cpu_init
extern "C" ggml_fp16_t ggml_table_gelu_f16[1 << 16];
extern "C" float ggml_table_f32_f16[1 << 16];

// ---- mirrors of the internal structs at the probed offsets (standard-layout prefixes) --------------------------
struct mel_mirror     { int n_len; int n_len_org; int n_mel; std::vector<float> data; };
struct filters_mirror { int32_t n_mel; int32_t n_fft; std::vector<float> data; };
static_assert(sizeof(mel_mirror) == VOAICE_SIZEOF_MEL, "whisper_mel layout changed");
static_assert(sizeof(filters_mirror) == VOAICE_SIZEOF_FILTERS, "whisper_filters layout changed");

[[noreturn]] static void die(const char * msg) { std::fprintf(stderr, "whisper_oracle: %s\n", msg); std::exit(2); }

// ---- sha256 (FIPS 180-4), for tensor bytes ----------------------------------------------------------------------
namespace sha {
static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,
    0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
    0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,
    0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hh + (rotr(e,6)^rotr(e,11)^rotr(e,25)) + ((e&f)^(~e&g)) + K[i] + w[i];
        uint32_t t2 = (rotr(a,2)^rotr(a,13)^rotr(a,22)) + ((a&b)^(a&c)^(b&c));
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}
static std::string hex(const uint8_t * data, size_t n) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) block(h, data + i);
    uint8_t tail[128] = {0};
    size_t r = n - i;
    std::memcpy(tail, data + i, r);
    tail[r] = 0x80;
    size_t tl = (r + 9 <= 64) ? 64 : 128;
    uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) tail[tl - 1 - k] = (uint8_t)(bits >> (8*k));
    block(h, tail);
    if (tl == 128) block(h, tail + 64);
    char out[65];
    for (int k = 0; k < 8; k++) std::snprintf(out + 8*k, 9, "%08x", h[k]);
    return std::string(out, 64);
}
}

static std::string hexbytes(const char * s) {
    std::string out;
    char b[3];
    for (const unsigned char * p = (const unsigned char *)s; *p; p++) { std::snprintf(b, 3, "%02x", *p); out += b; }
    return out;
}

static void write_f32(const std::string & path, const float * data, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) die(("cannot write " + path).c_str());
    if (std::fwrite(data, sizeof(float), n, f) != n) die("short write");
    std::fclose(f);
}

static uint32_t f32_bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

// ---- 16-bit PCM, mono, 16 kHz WAV (the only input this stage accepts; anything else is refused, not converted) -
static std::vector<float> read_wav(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) die("cannot open wav");
    std::vector<uint8_t> b;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    auto u16 = [&](size_t o) { return (uint32_t)b[o] | (uint32_t)b[o+1] << 8; };
    auto u32 = [&](size_t o) { return u16(o) | u16(o+2) << 16; };
    if (b.size() < 12 || std::memcmp(&b[0], "RIFF", 4) || std::memcmp(&b[8], "WAVE", 4)) die("not a RIFF/WAVE file");
    size_t pos = 12; bool fmt = false; std::vector<float> out;
    while (pos + 8 <= b.size()) {
        uint32_t len = u32(pos + 4);
        size_t body = pos + 8;
        if (body + len > b.size()) die("truncated chunk");
        if (!std::memcmp(&b[pos], "fmt ", 4)) {
            if (u16(body) != 1 || u16(body+2) != 1 || u32(body+4) != 16000 || u16(body+14) != 16)
                die("wav must be PCM, 1 channel, 16000 Hz, 16-bit");
            fmt = true;
        } else if (!std::memcmp(&b[pos], "data", 4)) {
            if (!fmt) die("data before fmt");
            for (uint32_t i = 0; i + 1 < len; i += 2) {
                int16_t s = (int16_t)u16(body + i);
                // miniaudio's ma_pcm_s16_to_f32 (whisper-cli's decoder): x * 0.00003051757812f, which is 2^-15 exactly
                out.push_back((float)s * 0.00003051757812f);
            }
            return out;
        }
        pos = body + len + (len & 1);
    }
    die("no data chunk");
}

static void check(bool ok, const char * what) { if (!ok) die(what); }

// ---- heap accounting: operator new/delete replaced in this executable, which the shared libraries' std::vector
// allocations bind to (ELF interposition), so the bytes a call holds live at its peak can be read as voaice's
// counting allocator reads its own. malloc/calloc called directly (ggml's buffers) are not counted; the mel path
// allocates through std::vector only. Thread stacks are mapped, not allocated, here as in voaice.
static std::atomic<size_t> g_live{0}, g_peak{0};
static void * counted(size_t n) {
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    size_t now = g_live.fetch_add(malloc_usable_size(p)) + malloc_usable_size(p);
    size_t pk = g_peak.load();
    while (now > pk && !g_peak.compare_exchange_weak(pk, now)) {}
    return p;
}
static void uncounted(void * p) {
    if (!p) return;
    g_live.fetch_sub(malloc_usable_size(p));
    std::free(p);
}
void * operator new(size_t n) { return counted(n); }
void * operator new[](size_t n) { return counted(n); }
void operator delete(void * p) noexcept { uncounted(p); }
void operator delete[](void * p) noexcept { uncounted(p); }
void operator delete(void * p, size_t) noexcept { uncounted(p); }
void operator delete[](void * p, size_t) noexcept { uncounted(p); }

// ---- efficiency, from /proc (as src/measure.rs) -------------------------------------------------------------------
static double cpu_seconds() {
    FILE * f = std::fopen("/proc/self/stat", "r");
    if (!f) return -1;
    char buf[4096];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char * p = std::strrchr(buf, ')');
    if (!p) return -1;
    p += 2;  // field 3 (state)
    for (int field = 3; field < 14 && *p; field++) { p = std::strchr(p, ' '); if (!p) return -1; p++; }
    unsigned long long ut = 0, st = 0;
    if (std::sscanf(p, "%llu %llu", &ut, &st) != 2) return -1;
    return (double)(ut + st) / 100.0;  // USER_HZ
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

static int bench_mel(const char * model_path, const char * wav, int threads) {
    whisper_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    check(ctx != nullptr, "model failed to load");
    std::vector<float> pcm = read_wav(wav);
    whisper_state * st = whisper_init_state(ctx);
    check(st != nullptr, "whisper_init_state failed");
    auto call = [&] { check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), threads) == 0, "pcm_to_mel failed"); };
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    const size_t live0 = g_live.load();
    g_peak.store(live0);
    call();
    const size_t heap_peak = g_peak.load() - live0;
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        const int64_t t0 = ggml_time_us();
        call();
        const double ms = (ggml_time_us() - t0) / 1000.0;
        if (ms < best) best = ms;
    }
    const double c0 = cpu_seconds();
    const int64_t w0 = ggml_time_us();
    int reps = 0;
    while (reps < 10 || ggml_time_us() - w0 < 1000000) { call(); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-mel-reference threads %d samples %zu heap_peak_kb %zu wall_best_ms %.3f cpu_ms_per_call %.3f cpu_reps %d "
                "rss_peak_delta_kb %ld rss_peak_kb %ld\n",
                threads, pcm.size(), (heap_peak + 1023) / 1024, best, (c1 - c0) * 1000.0 / reps, reps, peak >= 0 ? peak - before : -1, peak);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}


// ---- 0.0.3: f32 <-> f16 and GELU, as the shipped libggml-base / libggml-cpu compute them ------------------------
template <class T> static void write_bin(const std::string & path, const T * data, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) die(("cannot write " + path).c_str());
    if (std::fwrite(data, sizeof(T), n, f) != n) die("short write");
    std::fclose(f);
}
static float bits_f32(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
// 64-bit FNV-1a over u64 words: the f16 outputs four to a word, the f32 outputs two (little-endian packing)
static uint64_t digest16(const uint16_t * y, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint64_t w = (uint64_t)y[i] | (uint64_t)y[i+1] << 16 | (uint64_t)y[i+2] << 32 | (uint64_t)y[i+3] << 48;
        h = (h ^ w) * 0x100000001b3ULL;
    }
    return h;
}
static uint64_t digest32(const float * y, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i + 2 <= n; i += 2) {
        uint64_t w = (uint64_t)f32_bits(y[i]) | (uint64_t)f32_bits(y[i+1]) << 32;
        h = (h ^ w) * 0x100000001b3ULL;
    }
    return h;
}

// the boundary set (see the header); built from integer arithmetic only, so no conversion under test shapes it
static std::vector<uint32_t> f32_inputs() {
    std::vector<uint32_t> v;
    auto with_neighbours = [&](uint32_t u) {
        v.push_back(u);
        if ((u & 0x7FFFFFFF) != 0) v.push_back(u - 1);                          // toward zero (stays the same sign)
        if (((u + 1) & 0x7F800000) != 0x7F800000) v.push_back(u + 1);         // away from zero, still finite
    };
    auto exact = [](uint32_t h) -> double {    // the exact value of a finite f16
        const int e = (h >> 10) & 31; const double m = h & 0x3FF;
        const double a = e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25);
        return (h & 0x8000) ? -a : a;
    };
    auto f32bits = [](double d) { float f = (float)d; if ((double)f != d) die("an input is not exact in f32"); return f32_bits(f); };
    for (uint32_t h = 0; h < 0x10000; h++) {
        if ((h & 0x7C00) == 0x7C00) continue;                                   // inf and NaN: below
        with_neighbours(f32bits(exact(h)));
        const uint32_t mag = h & 0x7FFF;
        const double next = mag == 0x7BFF ? ((h & 0x8000) ? -65536.0 : 65536.0) // past the largest: the overflow edge
                                          : exact(h + 1);
        with_neighbours(f32bits((exact(h) + next) / 2));                       // the tie (exact: 11 bits in 24)
    }
    const uint32_t specials[] = {
        0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, // subnormals
        0x33000000, 0x32FFFFFF, 0x33000001, 0x33800000, 0x33C00000, 0x2F800000, 0x0D800000,             // underflow
        0x477FEFFF, 0x477FF000, 0x477FF001, 0x47800000, 0x7F7FFFFF, 0x501502F9,                         // overflow
        0x7F800000,                                                                                      // inf
        0x7F800001, 0x7F800FFF, 0x7F801000, 0x7F802000, 0x7FA00000, 0x7FBFFFFF, 0x7FC00000, 0x7FC00001,  // NaNs
        0x7FDFE000, 0x7FFFE000, 0x7FFFFFFF, 0x7F8FFFFF, 0x7FB00001,
    };
    for (uint32_t u : specials) { v.push_back(u); v.push_back(u | 0x80000000); }
    for (int k = -20; k <= 20; k++) {                                           // +-10, the GELU clamps
        v.push_back((uint32_t)((int64_t)0x41200000 + k)); v.push_back((uint32_t)((int64_t)0xC1200000 + k));
    }
    uint64_t x = 0x9E3779B97F4A7C15ULL;                                         // xorshift64
    for (int i = 0; i < (1 << 20); i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v.push_back((uint32_t)x); }
    while (v.size() % 8) v.push_back(0);
    return v;
}

// a ggml graph of one op, gelu, over n f32 values: what the encoder's ggml_gelu runs, by the shipped CPU backend
struct gelu_graph {
    ggml_context * ctx; ggml_tensor * in; ggml_tensor * out; ggml_cgraph * gf;
    explicit gelu_graph(int64_t n) {
        ggml_init_params p = { (size_t)n * 8 + ggml_graph_overhead() + (16u << 20), nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        out = ggml_gelu(ctx, in);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    void run(const float * x, float * y, int threads) {
        std::memcpy(in->data, x, ggml_nbytes(in));
        check(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "graph compute failed");
        std::memcpy(y, out->data, ggml_nbytes(out));
    }
    ~gelu_graph() { ggml_free(ctx); }
};

static int record_f16(const std::string & outdir) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    ggml_cpu_init();
    const int n16 = 1 << 16;

    // f16 -> f32, all 65,536 patterns, four ways
    std::vector<uint16_t> all(n16);
    for (int i = 0; i < n16; i++) all[i] = (uint16_t)i;
    std::vector<float> w(n16);
    for (int i = 0; i < n16; i++) w[i] = ggml_fp16_to_fp32(all[i]);
    write_bin(outdir + "/f16_to_f32.base.u32", w.data(), n16);
    write_bin(outdir + "/f16_to_f32.table.u32", ggml_table_f32_f16, n16);
    ggml_cpu_fp16_to_fp32(all.data(), w.data(), n16);
    write_bin(outdir + "/f16_to_f32.cpu_row.u32", w.data(), n16);
    for (int i = 0; i < n16; i++) ggml_cpu_fp16_to_fp32(&all[i], &w[i], 1);
    write_bin(outdir + "/f16_to_f32.cpu_tail.u32", w.data(), n16);

    // f32 -> f16 on the boundary set, three ways
    const std::vector<uint32_t> in_bits = f32_inputs();
    const size_t n = in_bits.size();
    std::vector<float> in(n);
    for (size_t i = 0; i < n; i++) in[i] = bits_f32(in_bits[i]);
    write_bin(outdir + "/f32_inputs.u32", in_bits.data(), n);
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = ggml_fp32_to_fp16(in[i]);
    write_bin(outdir + "/f32_to_f16.base.u16", h.data(), n);
    ggml_cpu_fp32_to_fp16(in.data(), h.data(), (int64_t)n);
    write_bin(outdir + "/f32_to_f16.cpu_row.u16", h.data(), n);
    for (size_t i = 0; i < n; i += 3) ggml_cpu_fp32_to_fp16(&in[i], &h[i], (int64_t)std::min<size_t>(3, n - i));
    write_bin(outdir + "/f32_to_f16.cpu_tail.u16", h.data(), n);

    // the GELU table, and the op on the boundary set followed by every f16 value widened
    write_bin(outdir + "/gelu_table.u16", ggml_table_gelu_f16, n16);
    std::vector<float> gx(in);
    for (int i = 0; i < n16; i++) gx.push_back(ggml_table_f32_f16[i]);
    std::vector<float> g1(gx.size()), g4(gx.size());
    {
        gelu_graph g((int64_t)gx.size());
        g.run(gx.data(), g1.data(), 1);
        g.run(gx.data(), g4.data(), 4);
    }
    write_bin(outdir + "/gelu.u32", g1.data(), g1.size());
    const bool gelu_threads_agree = std::memcmp(g1.data(), g4.data(), g1.size() * 4) == 0;

    // every f32 pattern: 65,536 chunks of 65,536, digested per chunk
    std::vector<uint64_t> d_base(n16), d_row(n16), d_tail(n16), d_gelu(n16);
    {
        const int per = 16;                         // gelu in graphs of 16 chunks (2^20 values) to amortize the setup
        gelu_graph g((int64_t)n16 * per);
        std::vector<float> x((size_t)n16 * per), y((size_t)n16 * per);
        std::vector<uint16_t> o(n16);
        for (int c0 = 0; c0 < n16; c0 += per) {
            for (int c = c0; c < c0 + per; c++) {
                float * xc = &x[(size_t)(c - c0) * n16];
                for (int i = 0; i < n16; i++) xc[i] = bits_f32((uint32_t)c << 16 | (uint32_t)i);
                for (int i = 0; i < n16; i++) o[i] = ggml_fp32_to_fp16(xc[i]);
                d_base[c] = digest16(o.data(), n16);
                ggml_cpu_fp32_to_fp16(xc, o.data(), n16);
                d_row[c] = digest16(o.data(), n16);
                for (int i = 0; i < n16; i += 3) ggml_cpu_fp32_to_fp16(xc + i, o.data() + i, std::min(3, n16 - i));
                d_tail[c] = digest16(o.data(), n16);
            }
            g.run(x.data(), y.data(), 1);
            for (int c = c0; c < c0 + per; c++) d_gelu[c] = digest32(&y[(size_t)(c - c0) * n16], n16);
        }
    }
    write_bin(outdir + "/f32_to_f16.base.digest", d_base.data(), n16);
    write_bin(outdir + "/f32_to_f16.cpu_row.digest", d_row.data(), n16);
    write_bin(outdir + "/f32_to_f16.cpu_tail.digest", d_tail.data(), n16);
    write_bin(outdir + "/gelu.digest", d_gelu.data(), n16);

    const bool from_float = ggml_get_type_traits_cpu(GGML_TYPE_F16)->from_float == (ggml_from_float_t)ggml_cpu_fp32_to_fp16;
    int same_base_tail = 0, same_base_row = 0;
    for (int c = 0; c < n16; c++) { same_base_tail += d_base[c] == d_tail[c]; same_base_row += d_base[c] == d_row[c]; }
    FILE * f = std::fopen((outdir + "/f16.tsv").c_str(), "w");
    std::fprintf(f, "f32_inputs\t%zu\ngelu_inputs\t%zu\nf16c\t%s\nfrom_float_f16_is_ggml_cpu_fp32_to_fp16\t%s\n"
                    "gelu_threads_1_vs_4_bit_identical\t%s\nchunks_base_eq_cpu_tail\t%d\nchunks_base_eq_cpu_row\t%d\n",
                 n, gx.size(), __builtin_cpu_supports("f16c") ? "yes" : "no", from_float ? "yes" : "NO",
                 gelu_threads_agree ? "yes" : "NO", same_base_tail, same_base_row);
    std::fclose(f);
    std::fprintf(stderr, "whisper_oracle: f16: 65536 f16 widened 4 ways; %zu f32 narrowed 3 ways; all 2^32 f32 digested "
                         "(chunks where libggml-base == ggml-cpu tail: %d, == ggml-cpu row: %d of 65536); gelu table; "
                         "gelu op on %zu values (1 vs 4 threads identical: %s) and on all 2^32\n",
                 n, same_base_tail, same_base_row, gx.size(), gelu_threads_agree ? "yes" : "NO");
    return 0;
}

// --bench-f16 <what>, one line, in a fresh process:
//   init   the wall time of the first ggml_cpu_init (it builds the GELU, quick-GELU and f32<-f16 tables, and two
//          256-entry ones); nothing before it initializes ggml-cpu
//   rows   ggml_cpu_fp32_to_fp16 and ggml_cpu_fp16_to_fp32 on 384 x 1500 values (one encoder activation), and the
//          GELU op on 1536 x 1500 (the encoder's MLP) through a graph at 1 thread; best of 10 each
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int bench_f16(const char * what) {
    if (std::strcmp(what, "init") == 0) {
        const double t0 = now_ms();
        ggml_cpu_init();
        std::printf("bench-f16-reference init_ms %.3f\n", now_ms() - t0);
        return 0;
    }
    if (std::strcmp(what, "rows") != 0) die("--bench-f16 init | rows");
    ggml_cpu_init();
    const int n = 384 * 1500, ng = 1536 * 1500;
    std::vector<float> x(ng), y(ng);
    std::vector<uint16_t> h(n);
    uint64_t s = 1;
    for (int i = 0; i < ng; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; x[i] = (float)((int64_t)(s >> 11) % 2000001 - 1000000) * 1e-5f; }
    auto best = [](auto && f) { f(); double b = 1e30; for (int r = 0; r < 10; r++) { const double t = now_ms(); f(); b = std::min(b, now_ms() - t); } return b; };
    const double to16 = best([&] { ggml_cpu_fp32_to_fp16(x.data(), h.data(), n); });
    const double to32 = best([&] { ggml_cpu_fp16_to_fp32(h.data(), y.data(), n); });
    gelu_graph g(ng);
    const double gelu = best([&] { g.run(x.data(), y.data(), 1); });
    // the graph run includes two memcpy of the tensor (in and out); measured alone so it can be taken off
    const double copies = best([&] { std::memcpy(g.in->data, x.data(), (size_t)ng * 4); std::memcpy(y.data(), g.out->data, (size_t)ng * 4); });
    std::printf("bench-f16-reference fp32_to_fp16_row_ms %.3f fp16_to_fp32_row_ms %.3f gelu_ms %.3f gelu_copies_ms %.3f n_row %d n_gelu %d\n",
                to16, to32, gelu, copies, n, ng);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--f16") == 0) return record_f16(argv[2]);
    if (argc == 3 && std::strcmp(argv[1], "--bench-f16") == 0) return bench_f16(argv[2]);
    if (argc == 5 && std::strcmp(argv[1], "--bench-mel") == 0) return bench_mel(argv[2], argv[3], std::atoi(argv[4]));
    if (argc < 3) die("usage: whisper_oracle <model.bin> <outdir> [wav ...]");
    const std::string outdir = argv[2];
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");

    whisper_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    whisper_context_params cparams = whisper_context_default_params();   // as whisper-cli: flash_attn = true
    cparams.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(argv[1], cparams);
    check(ctx != nullptr, "model failed to load");

    // ---- the model as the loader holds it ----------------------------------------------------------------------
    char * model = (char *)ctx + VOAICE_OFF_CTX_MODEL;
    auto & tensors = *reinterpret_cast<std::map<std::string, ggml_tensor *> *>(model + VOAICE_OFF_MODEL_TENSORS);
    const int n_loaded = *reinterpret_cast<int *>(model + VOAICE_OFF_MODEL_N_LOADED);
    auto & filters = *reinterpret_cast<filters_mirror *>(model + VOAICE_OFF_MODEL_FILTERS);
    // self-checks: the tensor map must hold exactly what the loader counted, the filterbank must be n_mels wide
    check((int)tensors.size() == n_loaded, "layout check failed: tensor map size != n_loaded");
    check(filters.n_mel == whisper_model_n_mels(ctx), "layout check failed: filters.n_mel != n_mels");
    check((size_t)filters.n_mel * filters.n_fft == filters.data.size(), "layout check failed: filters size");

    {
        FILE * f = std::fopen((outdir + "/model.tsv").c_str(), "w");
        std::fprintf(f, "n_vocab\t%d\nn_audio_ctx\t%d\nn_audio_state\t%d\nn_audio_head\t%d\nn_audio_layer\t%d\n"
                        "n_text_ctx\t%d\nn_text_state\t%d\nn_text_head\t%d\nn_text_layer\t%d\nn_mels\t%d\nftype\t%d\n"
                        "model_type\t%s\nfilters\t%d\t%d\nn_tensors\t%d\n",
                     whisper_model_n_vocab(ctx), whisper_model_n_audio_ctx(ctx), whisper_model_n_audio_state(ctx),
                     whisper_model_n_audio_head(ctx), whisper_model_n_audio_layer(ctx), whisper_model_n_text_ctx(ctx),
                     whisper_model_n_text_state(ctx), whisper_model_n_text_head(ctx), whisper_model_n_text_layer(ctx),
                     whisper_model_n_mels(ctx), whisper_model_ftype(ctx), whisper_model_type_readable(ctx),
                     filters.n_mel, filters.n_fft, n_loaded);
        std::vector<uint8_t> bytes;
        for (auto & kv : tensors) {          // std::map: sorted by name
            ggml_tensor * t = kv.second;
            size_t nb = ggml_nbytes(t);
            bytes.resize(nb);
            ggml_backend_tensor_get(t, bytes.data(), 0, nb);
            std::fprintf(f, "tensor\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%s\n", kv.first.c_str(), ggml_type_name(t->type),
                         (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], nb,
                         sha::hex(bytes.data(), nb).c_str());
        }
        std::fclose(f);
        write_f32(outdir + "/filters.f32", filters.data.data(), filters.data.size());

        FILE * v = std::fopen((outdir + "/vocab.tsv").c_str(), "w");
        for (int i = 0; i < whisper_model_n_vocab(ctx); i++)
            std::fprintf(v, "%d\t%s\n", i, hexbytes(whisper_token_to_str(ctx, i)).c_str());
        std::fprintf(v, "special\teot=%d sot=%d prev=%d solm=%d not=%d beg=%d\n", whisper_token_eot(ctx),
                     whisper_token_sot(ctx), whisper_token_prev(ctx), whisper_token_solm(ctx), whisper_token_not(ctx),
                     whisper_token_beg(ctx));
        std::fclose(v);
    }

    // ---- per WAV: the mel, then the transcript -----------------------------------------------------------------
    for (int a = 3; a < argc; a++) {
        std::string path = argv[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);

        std::vector<float> pcm = read_wav(path.c_str());
        write_f32(dir + "/pcm.f32", pcm.data(), pcm.size());

        // the mel at 1 thread and at 4: whisper splits frames between threads (frame i to thread i % n), each frame
        // computed whole by one thread, so the bits should not depend on the count; this records whether they do
        std::vector<float> mel1;
        int n_len = 0, n_len_org = 0, n_mel = 0;
        bool threads_agree = true;
        double best_us = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            whisper_state * st = whisper_init_state(ctx);
            const int64_t t0 = ggml_time_us();
            whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1);
            const double us = (double)(ggml_time_us() - t0);
            if (us < best_us) best_us = us;
            whisper_free_state(st);
        }
        for (int threads : {1, 4}) {
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), threads) == 0, "pcm_to_mel failed");
            auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
            // whisper_n_len_from_state returns mel.n_len_org (src/whisper.cpp), so that is the field it checks
            check(mel.n_len_org == whisper_n_len_from_state(st), "layout check failed: mel.n_len_org != whisper_n_len_from_state");
            check(mel.n_mel == whisper_model_n_mels(ctx), "layout check failed: mel.n_mel");
            check(mel.data.size() == (size_t)mel.n_mel * mel.n_len, "layout check failed: mel.data size");
            if (threads == 1) {
                mel1 = mel.data; n_len = mel.n_len; n_len_org = mel.n_len_org; n_mel = mel.n_mel;
            } else {
                threads_agree = mel.data.size() == mel1.size() &&
                                std::memcmp(mel.data.data(), mel1.data(), mel1.size() * 4) == 0;
            }
            whisper_free_state(st);
        }
        write_f32(dir + "/mel.f32", mel1.data(), mel1.size());
        FILE * m = std::fopen((dir + "/mel.tsv").c_str(), "w");
        std::fprintf(m, "n_samples\t%zu\nn_mel\t%d\nn_len\t%d\nn_len_org\t%d\nthreads_1_vs_4_bit_identical\t%s\n"
                        "reference_ms_1_thread_best_of_5\t%.2f\n",
                     pcm.size(), n_mel, n_len, n_len_org, threads_agree ? "yes" : "NO", best_us / 1000.0);
        std::fclose(m);

        // the transcript: greedy, one thread, everything else at whisper-cli's defaults except the printing.
        // temperature_inc = 0 turns off the temperature fallback (which would sample), so the run is one greedy pass.
        whisper_full_params fp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        // VOAICE_FULL_THREADS (default 1) sets whisper_full's threads, to record whether the transcript depends on it
        const char * ft = std::getenv("VOAICE_FULL_THREADS");
        fp.n_threads = ft ? std::atoi(ft) : 1;
        fp.print_progress = false; fp.print_realtime = false; fp.print_timestamps = false; fp.print_special = false;
        fp.token_timestamps = true;
        fp.temperature_inc = 0.0f;
        fp.language = "en";
        whisper_state * st = whisper_init_state(ctx);
        check(whisper_full_with_state(ctx, st, fp, pcm.data(), (int)pcm.size()) == 0, "whisper_full failed");
        FILE * t = std::fopen((dir + "/transcript.tsv").c_str(), "w");
        std::fprintf(t, "# params: greedy best_of=%d n_threads=%d flash_attn=%d token_timestamps=1 temperature=%g "
                        "temperature_inc=0 language=en no_context=%d\n",
                     fp.greedy.best_of, fp.n_threads, (int)cparams.flash_attn, fp.temperature, (int)fp.no_context);
        const int ns = whisper_full_n_segments_from_state(st);
        for (int s = 0; s < ns; s++) {
            std::fprintf(t, "segment\t%d\t%lld\t%lld\t%s\n", s, (long long)whisper_full_get_segment_t0_from_state(st, s),
                         (long long)whisper_full_get_segment_t1_from_state(st, s),
                         hexbytes(whisper_full_get_segment_text_from_state(st, s)).c_str());
            const int nt = whisper_full_n_tokens_from_state(st, s);
            for (int k = 0; k < nt; k++) {
                whisper_token_data d = whisper_full_get_token_data_from_state(st, s, k);
                std::fprintf(t, "token\t%d\t%d\t%d\t%lld\t%lld\t%08x\t%s\n", s, k, d.id, (long long)d.t0, (long long)d.t1,
                             f32_bits(d.p), hexbytes(whisper_full_get_token_text_from_state(ctx, st, s, k)).c_str());
            }
        }
        std::fclose(t);
        // also the readable text, for people
        FILE * tx = std::fopen((dir + "/transcript.txt").c_str(), "w");
        for (int s = 0; s < ns; s++) std::fprintf(tx, "%s\n", whisper_full_get_segment_text_from_state(st, s));
        std::fclose(tx);
        whisper_free_state(st);
        std::fprintf(stderr, "whisper_oracle: %s: %zu samples, mel %dx%d, threads agree: %s, %d segments\n",
                     stem.c_str(), pcm.size(), n_mel, n_len, threads_agree ? "yes" : "NO", ns);
    }
    whisper_free(ctx);
    return 0;
}
