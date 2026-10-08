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
//   whisper_oracle --conv1 <model.bin> <outdir> <wav ...>   (0.0.6) the conv graph's nodes and ggml_vec_dot_f16, see record_conv1
//   whisper_oracle --bench-conv1 <model.bin> <wav> <threads> conv1|gelu   (0.0.6) conv1's time, see bench_conv1
//   whisper_oracle --conv2 <model.bin> <outdir> <wav ...>   (0.0.7) conv2's nodes and the positional add, see record_conv2
//   whisper_oracle --bench-conv2 <model.bin> <wav> <threads> conv2|stage   (0.0.7) their time, see bench_conv2
//   whisper_oracle --norm <model.bin> <outdir> <wav ...>    (0.0.8) the encoder's nine norm -> mul -> add chains, see record_norm
//   whisper_oracle --bench-norm <model.bin> <wav> <threads> norm|chain   (0.0.8) their time, see bench_norm
//   whisper_oracle --matmul <model.bin> <outdir> <wav ...>  (0.0.9) every block's products, biases, GELU, residuals, see record_matmul
//   whisper_oracle --mm-nan <model.bin> <outdir>            (0.0.9) from_float on NaN-bearing rows at 1..8 threads, see record_mm_nan
//   whisper_oracle --bench-mm <model.bin> <wav> <threads> q|fc1|fc2|qkv|mlp|block   (0.0.9) their time, see bench_mm
//   whisper_oracle --encoder <model.bin> <outdir> <wav ...> (v0.1.0) flash attention and the whole encoder, see record_encoder
//   whisper_oracle --bench-attn <model.bin> <wav> <threads>    (v0.1.0) block 0's attention, see bench_attn
//   whisper_oracle --bench-encode <model.bin> <wav> <threads>  (v0.1.0) whisper_encode_with_state, mel -> embd_enc (and,
//                                                              found in 0.1.1, the cross graph's kv_cross: it runs that too)
//   whisper_oracle --cross <model.bin> <outdir> <wav ...>   (0.1.1) the cross graph (sched_cross) and kv_cross, see record_cross
//   whisper_oracle --bench-cross <model.bin> <wav> <threads>   (0.1.1) the cross graph's time, see bench_cross
//   whisper_oracle --decin <model.bin> <outdir> <wav ...>   (0.1.2) the decoder's input on every decoder call, see record_decin
//   whisper_oracle --bench-decin <model.bin> <threads> <n_tokens>   (0.1.2) those three nodes' time, see bench_decin
//   whisper_oracle --selfkv <model.bin> <outdir> <wav ...>  (0.1.3) the self-attention products, the mask, kv_self, see record_selfkv
//   whisper_oracle --bench-selfkv <model.bin> <threads> <n_tokens> block|call   (0.1.3) their time, see bench_selfkv
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
#include <set>
#include <cstddef>
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


// ---- 0.0.6: encoder conv1 and ggml_vec_dot_f16, observed in the shipped library -----------------------------------
// --conv1 <model.bin> <outdir> <wav ...> writes, per wav, <outdir>/<stem>/:
//   nodes.tsv        every node of the conv graph as the scheduler computes it: index, name, op, type, ne0..ne3
//   im2col.u16       the first IM2COL node (conv1's: f16 [240, 3000], row t = the 240 inputs of output frame t)
//   conv1.f32        the MUL_MAT after it (conv1 without its bias: f32, ne0 = 3000 frames, ne1 = 384 channels)
//   conv1_bias.f32   the ADD after it (+ encoder.conv1.bias)
//   conv1_gelu.f32   the GELU after it (what conv2's im2col reads)
//   conv1.tsv        threads compared (1 and 4: all four nodes bit-identical?), embd_conv observed vs not, the
//                    standalone ggml graph (the one --bench-conv1 times) vs the scheduler's node
// The nodes are read through ggml_backend_sched_set_eval_callback on whisper_state::sched_conv (layout probe); the
// shipped library does all the arithmetic, the callback only copies each node's output when the node is done.
// and <outdir>/vecdot.bin (+ vecdot.tsv): ggml_vec_dot_f16 through ggml_get_type_traits_cpu(F16)->vec_dot on real
// rows of every f16 tensor of the model, every length 1..300 on real conv2 rows, and random finite f16 patterns;
// records of (u32 n, u16 x[n], u16 y[n], u32 result bits), little-endian.
extern "C" void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);

struct node_capture {
    std::vector<std::string> lines;              // nodes.tsv
    std::vector<uint8_t> im2col, mm, add, gelu;  // the first IM2COL and the MUL_MAT / ADD / GELU that follow it
    int stage = 0;                               // 0: want IM2COL, 1: MUL_MAT, 2: ADD, 3: GELU, 4: done
    std::vector<uint8_t> last;                   // the last node's output (the graph's result, embd_conv)
};
static bool conv_cb(ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;                        // observe every node
    auto & c = *static_cast<node_capture *>(ud);
    char line[256];
    std::snprintf(line, sizeof line, "%zu\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld", c.lines.size(), t->name, ggml_op_desc(t),
                  ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    c.lines.push_back(line);
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    const ggml_op want[4] = {GGML_OP_IM2COL, GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_UNARY};
    if (c.stage < 4 && t->op == want[c.stage] && (c.stage != 3 || ggml_get_unary_op(t) == GGML_UNARY_OP_GELU)) {
        std::vector<uint8_t> * dst[4] = {&c.im2col, &c.mm, &c.add, &c.gelu};
        *dst[c.stage] = b;
        c.stage++;
    }
    c.last.swap(b);
    return true;
}

// conv1 alone as a ggml graph, computed by the shipped CPU backend (what --bench-conv1 times): ggml_conv_1d_ph on a
// copy of encoder.conv1.weight and the first 3000 mel frames, optionally + bias and GELU as whisper's graph does
struct conv1_graph {
    ggml_context * ctx; ggml_tensor * w; ggml_tensor * b; ggml_tensor * mel; ggml_tensor * out; ggml_cgraph * gf;
    conv1_graph(ggml_tensor * w_src, ggml_tensor * b_src, int n_frames, int n_mel, bool bias_gelu) {
        ggml_init_params p = { (size_t)64 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        w = ggml_new_tensor_3d(ctx, w_src->type, w_src->ne[0], w_src->ne[1], w_src->ne[2]);
        ggml_backend_tensor_get(w_src, w->data, 0, ggml_nbytes(w));
        b = ggml_new_tensor_2d(ctx, b_src->type, b_src->ne[0], b_src->ne[1]);
        ggml_backend_tensor_get(b_src, b->data, 0, ggml_nbytes(b));
        mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_mel);
        out = ggml_conv_1d_ph(ctx, w, mel, 1, 1);
        if (bias_gelu) out = ggml_gelu(ctx, ggml_add(ctx, out, b));
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    void set_mel(const std::vector<float> & m, int n_len) {   // whisper's slice at offset 0: [n_mel][2*n_ctx]
        float * d = (float *)mel->data;
        const int64_t nf = mel->ne[0];
        std::memset(d, 0, ggml_nbytes(mel));
        for (int64_t j = 0; j < mel->ne[1]; j++)
            for (int64_t i = 0; i < std::min<int64_t>(nf, n_len); i++) d[j*nf + i] = m[j*n_len + i];
    }
    void run(int threads) { check(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "graph compute failed"); }
    ~conv1_graph() { ggml_free(ctx); }
};

static whisper_context * load_quiet(const char * model_path) {
    whisper_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    check(ctx != nullptr, "model failed to load");
    return ctx;
}
static std::map<std::string, ggml_tensor *> & model_tensors(whisper_context * ctx) {
    char * model = (char *)ctx + VOAICE_OFF_CTX_MODEL;
    auto & tensors = *reinterpret_cast<std::map<std::string, ggml_tensor *> *>(model + VOAICE_OFF_MODEL_TENSORS);
    check((int)tensors.size() == *reinterpret_cast<int *>(model + VOAICE_OFF_MODEL_N_LOADED), "layout check failed: tensor map size != n_loaded");
    return tensors;
}
static ggml_backend_sched_t state_sched(whisper_state * st, size_t off) {
    ggml_backend_sched_t s = *reinterpret_cast<ggml_backend_sched_t *>((char *)st + off + VOAICE_OFF_WSCHED_SCHED);
    check(s != nullptr && ggml_backend_sched_get_n_backends(s) >= 1, "layout check failed: whisper_state scheduler");
    for (int i = 0; i < ggml_backend_sched_get_n_backends(s); i++)   // CPU only: the build has no other backend
        check(std::strcmp(ggml_backend_name(ggml_backend_sched_get_backend(s, i)), "CPU") == 0, "a scheduler backend is not the CPU");
    return s;
}

static void record_vecdot(whisper_context * ctx, const std::string & outdir) {
    auto & tensors = model_tensors(ctx);
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_F16);
    FILE * f = std::fopen((outdir + "/vecdot.bin").c_str(), "wb");
    if (!f) die("cannot write vecdot.bin");
    size_t records = 0;
    auto put = [&](const uint16_t * x, const uint16_t * y, int n) {
        float s = 0.0f;
        tr->vec_dot(n, &s, 0, x, 0, y, 0, 1);
        const uint32_t un = (uint32_t)n, r = f32_bits(s);
        std::fwrite(&un, 4, 1, f); std::fwrite(x, 2, n, f); std::fwrite(y, 2, n, f); std::fwrite(&r, 4, 1, f);
        records++;
    };
    uint64_t r = 0x2545F4914F6CDD1DULL;
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return r; };
    std::vector<uint16_t> a, b, all;
    std::string per;
    for (auto & kv : tensors) {                    // sorted by name; every f16 tensor: its rows as mul_mat reads them
        ggml_tensor * t = kv.second;
        if (t->type != GGML_TYPE_F16) continue;
        const bool conv = kv.first.find(".conv") != std::string::npos;
        const int64_t len = conv ? t->ne[0] * t->ne[1] : t->ne[0];   // a conv weight is reshaped to [K*IC, OC]
        const int64_t rows = ggml_nelements(t) / len;
        all.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));
        const int pairs = 16;
        for (int k = 0; k < pairs; k++) {
            const int64_t i = (int64_t)(rnd() % rows), j = (int64_t)(rnd() % rows);
            put(&all[i * len], &all[j * len], (int)len);
        }
        per += kv.first + "\t" + std::to_string(len) + "\t" + std::to_string(pairs) + "\n";
    }
    {   // every length 1..300 (every tail 0..31, no main loop below 32) on two real conv2 rows
        ggml_tensor * t = tensors.at("encoder.conv2.weight");
        all.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));
        for (int n = 1; n <= 300; n++) put(&all[7 * 1152], &all[300 * 1152], n);
        per += "prefixes of encoder.conv2.weight rows 7 x 300\t1..300\t300\n";
    }
    {   // random finite f16 patterns (every exponent, both signs, subnormals): cancellation and rounding everywhere
        a.resize(1536); b.resize(1536);
        auto finite = [&]() { uint16_t h; do { h = (uint16_t)rnd(); } while ((h & 0x7C00) == 0x7C00); return h; };
        const int lens[] = {240, 1152, 384, 1536, 33, 63, 64, 95};
        for (int k = 0; k < 64; k++) {
            const int n = lens[k % 8];
            for (int i = 0; i < n; i++) { a[i] = finite(); b[i] = finite(); }
            put(a.data(), b.data(), n);
        }
        per += "random finite f16 patterns\t33..1536\t64\n";
    }
    std::fclose(f);
    FILE * m = std::fopen((outdir + "/vecdot.tsv").c_str(), "w");
    std::fprintf(m, "records\t%zu\nvec_dot_is_ggml_vec_dot_f16\t%s\nvec_dot_type\t%s\nnrows\t%lld\n"
                    "avx2\t%s\nfma\t%s\nf16c\t%s\navx512f\t%s\n%s",
                 records, tr->vec_dot == (ggml_vec_dot_t)ggml_vec_dot_f16 ? "yes" : "NO", ggml_type_name(tr->vec_dot_type),
                 (long long)tr->nrows, __builtin_cpu_supports("avx2") ? "yes" : "no", __builtin_cpu_supports("fma") ? "yes" : "no",
                 __builtin_cpu_supports("f16c") ? "yes" : "no", __builtin_cpu_supports("avx512f") ? "yes" : "no", per.c_str());
    std::fclose(m);
    std::fprintf(stderr, "whisper_oracle: vec_dot_f16: %zu records (real rows of every f16 tensor, lengths 1..300, random patterns)\n", records);
}

static int record_conv1(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    record_vecdot(ctx, outdir);
    auto & tensors = model_tensors(ctx);
    ggml_tensor * w1 = tensors.at("encoder.conv1.weight");
    ggml_tensor * b1 = tensors.at("encoder.conv1.bias");
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_mel = whisper_model_n_mels(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        node_capture caps[2];
        std::vector<uint8_t> unobserved;
        std::vector<float> mel_copy;
        int n_len = 0;
        const int threads[2] = {1, 4};
        for (int k = 0; k < 3; k++) {              // k = 0, 1: observed at 1 and 4 threads; k = 2: not observed
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            if (k == 0) {
                auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
                check(mel.n_len_org == whisper_n_len_from_state(st), "layout check failed: mel.n_len_org");
                mel_copy = mel.data; n_len = mel.n_len;
            }
            ggml_backend_sched_t sc = state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            if (k < 2) ggml_backend_sched_set_eval_callback(sc, conv_cb, &caps[k]);
            check(whisper_encode_with_state(ctx, st, 0, k < 2 ? threads[k] : 1) == 0, "whisper_encode failed");
            if (k < 2) check(caps[k].stage == 4, "the conv graph did not show IM2COL, MUL_MAT, ADD, GELU in order");
            ggml_tensor * ec = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_CONV);
            check(ec != nullptr && std::strcmp(ec->name, "embd_conv") == 0, "layout check failed: embd_conv");
            if (k == 2) {                          // read after the encoder ran: embd_conv is a graph output, kept
                unobserved.resize(ggml_nbytes(ec));
                ggml_backend_tensor_get(ec, unobserved.data(), 0, unobserved.size());
            }
            whisper_free_state(st);
        }
        node_capture & c = caps[0];
        check(c.im2col.size() == (size_t)240 * 2 * n_ctx * 2 && c.mm.size() == (size_t)2 * n_ctx * 384 * 4, "unexpected conv1 shapes");
        write_bin(dir + "/im2col.u16", (const uint16_t *)c.im2col.data(), c.im2col.size() / 2);
        write_bin(dir + "/conv1.f32", (const float *)c.mm.data(), c.mm.size() / 4);
        write_bin(dir + "/conv1_bias.f32", (const float *)c.add.data(), c.add.size() / 4);
        write_bin(dir + "/conv1_gelu.f32", (const float *)c.gelu.data(), c.gelu.size() / 4);
        FILE * nf = std::fopen((dir + "/nodes.tsv").c_str(), "w");
        for (auto & l : c.lines) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        const bool thr = c.im2col == caps[1].im2col && c.mm == caps[1].mm && c.add == caps[1].add && c.gelu == caps[1].gelu;
        const bool unobs = unobserved == c.last;
        bool alone = true;
        for (int bg = 0; bg < 2; bg++) {           // the standalone graph --bench-conv1 times, against the scheduler's nodes
            conv1_graph g(w1, b1, 2 * n_ctx, n_mel, bg == 1);
            g.set_mel(mel_copy, n_len);
            for (int th : {1, 4}) {
                g.run(th);
                const std::vector<uint8_t> & want = bg ? c.gelu : c.mm;
                alone = alone && ggml_nbytes(g.out) == want.size() && std::memcmp(g.out->data, want.data(), want.size()) == 0;
            }
        }
        FILE * m = std::fopen((dir + "/conv1.tsv").c_str(), "w");
        std::fprintf(m, "n_len\t%d\noffset\t0\nnodes\t%zu\nthreads_1_vs_4_bit_identical\t%s\nembd_conv_observed_eq_unobserved\t%s\n"
                        "standalone_graph_eq_sched\t%s\n", n_len, c.lines.size(), thr ? "yes" : "NO", unobs ? "yes" : "NO", alone ? "yes" : "NO");
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: conv graph %zu nodes; conv1 im2col + mul_mat + add + gelu recorded; 1 vs 4 threads "
                             "identical: %s; embd_conv with and without observing: %s; standalone graph = scheduler: %s\n",
                     stem.c_str(), c.lines.size(), thr ? "yes" : "NO", unobs ? "yes" : "NO", alone ? "yes" : "NO");
    }
    whisper_free(ctx);
    return 0;
}

// --bench-conv1 <model.bin> <wav> <threads> <what>: conv1 through the standalone graph above (what = conv1 | gelu:
// conv1 alone, or conv1 + bias + GELU), in a fresh process. wall = best of 10 computes, cpu = CPU ms per compute over
// >= 1 s, mem = the bytes ggml holds for the op's own tensors (im2col + the output, + the add and GELU outputs), from
// ggml_nbytes (ggml_graph_compute_with_ctx needs no work buffer here: src1 is already f16), rss = VmHWM delta of the
// first compute
static int bench_conv1(const char * model_path, const char * wav, int threads, const char * what) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
    const bool bg = std::strcmp(what, "gelu") == 0;
    conv1_graph g(tensors.at("encoder.conv1.weight"), tensors.at("encoder.conv1.bias"), 2 * whisper_model_n_audio_ctx(ctx),
                  whisper_model_n_mels(ctx), bg);
    g.set_mel(mel.data, mel.n_len);
    size_t op_bytes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
        ggml_tensor * t = ggml_graph_node(g.gf, i);
        if (t->view_src == nullptr) op_bytes += ggml_nbytes(t);
    }
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-conv1-reference what %s threads %d wall_best_ms %.3f cpu_ms_per_call %.3f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld\n",
                what, threads, best, (c1 - c0) * 1000.0 / reps, reps, (op_bytes + 1023) / 1024, peak >= 0 ? peak - before : -1);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// ---- 0.0.7: encoder conv2 and the positional embedding, observed in the shipped library ----------------------------
// --conv2 <model.bin> <outdir> <wav ...> writes, per wav, <outdir>/<stem>/:
//   nodes.tsv        every node of the conv graph, then the encoder graph's nodes up to its first ADD (the eval
//                    callback stops observing there: the rest of the encoder runs unobserved)
//   conv1_gelu.f32   conv1's GELU node (conv2's input: f32, ne0 = 3000 frames, ne1 = 384)
//   im2col2.u16      the second IM2COL node (conv2's: f16 [1152, 1500], row t = the 1152 inputs of output frame t)
//   conv2.f32        the MUL_MAT after it (f32, ne0 = 1500 frames, ne1 = 384 channels), conv2_bias.f32 the ADD,
//   conv2_gelu.f32   the GELU (the graph's last node: embd_conv)
//   cont.f32         the encoder graph's CONT of TRANSPOSE(embd_conv) (f32, ne0 = 384 channels, ne1 = 1500 frames)
//   pe_add.f32       the ADD after it: e_pe view + cont, the encoder's input (inpL)
//   conv2.tsv        threads compared (1 and 4: every node above bit-identical?), the scheduler's last conv node ==
//                    whisper_state::embd_conv read after an unobserved run, embd_enc (the whole encoder's output)
//                    with the observation vs without, the standalone graphs (what --bench-conv2 times) vs the nodes
struct conv2_capture {
    std::vector<std::string> lines;
    std::vector<uint8_t> gelu1, im2col, mm, add, gelu, cont, pe_add;
    int stage = 0;          // conv graph: 0 want GELU (conv1's), 1 IM2COL, 2 MUL_MAT, 3 ADD, 4 GELU, 5 done
    int enc = 0;            // encoder graph: 0 want CONT, 1 ADD, 2 done (stop observing)
    std::vector<uint8_t> last;
};
static bool is_gelu(const ggml_tensor * t) { return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_GELU; }
static void cap_line(std::vector<std::string> & lines, const char * graph, ggml_tensor * t) {
    char line[256];
    std::snprintf(line, sizeof line, "%s\t%zu\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld", graph, lines.size(), t->name, ggml_op_desc(t),
                  ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    lines.push_back(line);
}
static bool conv2_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<conv2_capture *>(ud);
    if (ask) return true;
    cap_line(c.lines, "conv", t);
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    std::vector<uint8_t> * dst = nullptr;
    if (c.stage == 0 && is_gelu(t)) dst = &c.gelu1;
    else if (c.stage == 1 && t->op == GGML_OP_IM2COL) dst = &c.im2col;
    else if (c.stage == 2 && t->op == GGML_OP_MUL_MAT) dst = &c.mm;
    else if (c.stage == 3 && t->op == GGML_OP_ADD) dst = &c.add;
    else if (c.stage == 4 && is_gelu(t)) dst = &c.gelu;
    if (dst) { *dst = b; c.stage++; }
    c.last.swap(b);
    return true;
}
static bool enc_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<conv2_capture *>(ud);
    if (ask) return c.enc < 2;                    // observe up to the first ADD, then let the encoder run whole
    if (c.enc >= 2) return true;
    cap_line(c.lines, "encode", t);
    if ((c.enc == 0 && t->op == GGML_OP_CONT) || (c.enc == 1 && t->op == GGML_OP_ADD)) {
        std::vector<uint8_t> & d = c.enc == 0 ? c.cont : c.pe_add;
        d.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, d.data(), 0, d.size());
        c.enc++;
    }
    return true;
}

// conv2, or the whole conv stage, as a standalone ggml graph on the shipped CPU backend (what --bench-conv2 times):
//   conv2: ggml_conv_1d_ph(w2, x, 2, 1) + b2, GELU, on a copy of conv1's GELU output x [3000, 384]
//   stage: mel [3000, 80] -> conv1 + b1, GELU -> conv2 + b2, GELU -> add(e_pe view, cont(transpose(.))), as
//          whisper_build_graph_conv then whisper_build_graph_encoder's first op
struct conv2_graph {
    ggml_context * ctx; ggml_tensor * in; ggml_tensor * out; ggml_tensor * conv; ggml_cgraph * gf;
    static ggml_tensor * copy(ggml_context * ctx, ggml_tensor * s) {
        ggml_tensor * t = ggml_new_tensor(ctx, s->type, ggml_n_dims(s), s->ne);
        ggml_backend_tensor_get(s, t->data, 0, ggml_nbytes(t));
        return t;
    }
    conv2_graph(std::map<std::string, ggml_tensor *> & m, int n_frames, int n_mel, bool stage) {
        ggml_init_params p = { (size_t)96 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        ggml_tensor * w2 = copy(ctx, m.at("encoder.conv2.weight")), * b2 = copy(ctx, m.at("encoder.conv2.bias"));
        ggml_tensor * cur;
        if (stage) {
            ggml_tensor * w1 = copy(ctx, m.at("encoder.conv1.weight")), * b1 = copy(ctx, m.at("encoder.conv1.bias"));
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_mel);
            cur = ggml_gelu(ctx, ggml_add(ctx, ggml_conv_1d_ph(ctx, w1, in, 1, 1), b1));
        } else {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, w2->ne[1]);
            cur = in;
        }
        conv = ggml_gelu(ctx, ggml_add(ctx, ggml_conv_1d_ph(ctx, w2, cur, 2, 1), b2));
        out = conv;
        if (stage) {
            ggml_tensor * pe = copy(ctx, m.at("encoder.positional_embedding"));
            const int n_ctx = (int)conv->ne[0];
            ggml_tensor * e_pe = ggml_view_2d(ctx, pe, pe->ne[0], n_ctx, pe->ne[0] * ggml_element_size(pe), 0);
            out = ggml_add(ctx, e_pe, ggml_cont(ctx, ggml_transpose(ctx, conv)));
        }
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    void set(const std::vector<float> & x, int n_len) {   // [rows][n_len] into [rows][ne0], zero past n_len (whisper's slice)
        float * d = (float *)in->data;
        const int64_t nf = in->ne[0];
        std::memset(d, 0, ggml_nbytes(in));
        for (int64_t j = 0; j < in->ne[1]; j++)
            for (int64_t i = 0; i < std::min<int64_t>(nf, n_len); i++) d[j*nf + i] = x[j*n_len + i];
    }
    void run(int threads) { check(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "graph compute failed"); }
    ~conv2_graph() { ggml_free(ctx); }
};
static bool same(const ggml_tensor * t, const std::vector<uint8_t> & want) {
    return ggml_nbytes(t) == want.size() && std::memcmp(t->data, want.data(), want.size()) == 0;
}

static int record_conv2(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_mel = whisper_model_n_mels(ctx), n_state = whisper_model_n_audio_state(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        conv2_capture caps[2];
        std::vector<uint8_t> unobs_conv, unobs_enc, obs_enc;
        std::vector<float> mel_copy;
        int n_len = 0;
        const int threads[2] = {1, 4};
        for (int k = 0; k < 3; k++) {              // k = 0, 1: observed at 1 and 4 threads; k = 2: not observed, 1 thread
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            if (k == 0) {
                auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
                check(mel.n_len_org == whisper_n_len_from_state(st), "layout check failed: mel.n_len_org");
                mel_copy = mel.data; n_len = mel.n_len;
            }
            ggml_backend_sched_t sc = state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            ggml_backend_sched_t se = state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            if (k < 2) {
                ggml_backend_sched_set_eval_callback(sc, conv2_cb, &caps[k]);
                ggml_backend_sched_set_eval_callback(se, enc_cb, &caps[k]);
            }
            check(whisper_encode_with_state(ctx, st, 0, k < 2 ? threads[k] : 1) == 0, "whisper_encode failed");
            if (k < 2) check(caps[k].stage == 5 && caps[k].enc == 2, "the graphs did not show conv2's IM2COL, MUL_MAT, ADD, GELU, then CONT, ADD");
            ggml_tensor * ec = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_CONV);
            ggml_tensor * ee = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_ENC);
            check(ec != nullptr && std::strcmp(ec->name, "embd_conv") == 0, "layout check failed: embd_conv");
            check(ee != nullptr && ee->type == GGML_TYPE_F32 && ee->ne[0] == n_state, "layout check failed: embd_enc");
            std::vector<uint8_t> enc(ggml_nbytes(ee));
            ggml_backend_tensor_get(ee, enc.data(), 0, enc.size());
            if (k == 0) obs_enc = enc;
            if (k == 2) {                          // read after the encoder ran: both are graph outputs, kept
                unobs_enc = enc;
                unobs_conv.resize(ggml_nbytes(ec));
                ggml_backend_tensor_get(ec, unobs_conv.data(), 0, unobs_conv.size());
            }
            whisper_free_state(st);
        }
        conv2_capture & c = caps[0];
        check(c.im2col.size() == (size_t)3 * n_state * n_ctx * 2 && c.mm.size() == (size_t)n_ctx * n_state * 4 &&
              c.pe_add.size() == (size_t)n_ctx * n_state * 4, "unexpected conv2 shapes");
        write_bin(dir + "/conv1_gelu.f32", (const float *)c.gelu1.data(), c.gelu1.size() / 4);
        write_bin(dir + "/im2col2.u16", (const uint16_t *)c.im2col.data(), c.im2col.size() / 2);
        write_bin(dir + "/conv2.f32", (const float *)c.mm.data(), c.mm.size() / 4);
        write_bin(dir + "/conv2_bias.f32", (const float *)c.add.data(), c.add.size() / 4);
        write_bin(dir + "/conv2_gelu.f32", (const float *)c.gelu.data(), c.gelu.size() / 4);
        write_bin(dir + "/cont.f32", (const float *)c.cont.data(), c.cont.size() / 4);
        write_bin(dir + "/pe_add.f32", (const float *)c.pe_add.data(), c.pe_add.size() / 4);
        FILE * nf = std::fopen((dir + "/nodes.tsv").c_str(), "w");
        for (auto & l : c.lines) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        const conv2_capture & d = caps[1];
        const bool thr = c.gelu1 == d.gelu1 && c.im2col == d.im2col && c.mm == d.mm && c.add == d.add && c.gelu == d.gelu &&
                         c.cont == d.cont && c.pe_add == d.pe_add;
        const bool last = c.gelu == c.last && unobs_conv == c.last;
        const bool enc_eq = obs_enc == unobs_enc;
        bool alone_conv2 = true, alone_stage = true;
        {
            conv2_graph g(tensors, 2 * n_ctx, n_mel, false);
            std::vector<float> x(c.gelu1.size() / 4);
            std::memcpy(x.data(), c.gelu1.data(), c.gelu1.size());
            g.set(x, 2 * n_ctx);
            for (int th : {1, 4}) { g.run(th); alone_conv2 = alone_conv2 && same(g.out, c.gelu); }
        }
        {
            conv2_graph g(tensors, 2 * n_ctx, n_mel, true);
            g.set(mel_copy, n_len);
            for (int th : {1, 4}) { g.run(th); alone_stage = alone_stage && same(g.conv, c.gelu) && same(g.out, c.pe_add); }
        }
        FILE * m = std::fopen((dir + "/conv2.tsv").c_str(), "w");
        std::fprintf(m, "n_len\t%d\noffset\t0\nn_ctx\t%d\nnodes\t%zu\nthreads_1_vs_4_bit_identical\t%s\n"
                        "last_conv_node_eq_embd_conv_unobserved\t%s\nembd_enc_observed_eq_unobserved\t%s\n"
                        "standalone_conv2_eq_sched\t%s\nstandalone_stage_eq_sched\t%s\n", n_len, n_ctx, c.lines.size(),
                     thr ? "yes" : "NO", last ? "yes" : "NO", enc_eq ? "yes" : "NO", alone_conv2 ? "yes" : "NO", alone_stage ? "yes" : "NO");
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: %zu nodes observed; conv2 im2col + mul_mat + add + gelu, cont + pe add recorded; 1 vs 4 "
                             "threads identical: %s; embd_conv = last node, unobserved: %s; embd_enc observed = unobserved: %s; "
                             "standalone conv2 / stage = scheduler: %s / %s\n", stem.c_str(), c.lines.size(), thr ? "yes" : "NO",
                     last ? "yes" : "NO", enc_eq ? "yes" : "NO", alone_conv2 ? "yes" : "NO", alone_stage ? "yes" : "NO");
    }
    whisper_free(ctx);
    return 0;
}

// --bench-conv2 <model.bin> <wav> <threads> <what>: what = conv2 (conv1's GELU output -> conv2 + bias + GELU: embd_conv)
// or stage (the mel -> conv1 -> conv2 -> + positions: the encoder's input), through the standalone graph above, in a
// fresh process. wall = best of 10, cpu = CPU ms per compute over >= 1 s, mem = the bytes of the graph's own tensors
// (everything but the copied weights, the embedding and the input), rss = VmHWM delta of the first compute.
static int bench_conv2(const char * model_path, const char * wav, int threads, const char * what) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
    const bool stage = std::strcmp(what, "stage") == 0;
    check(stage || std::strcmp(what, "conv2") == 0, "--bench-conv2 ... conv2|stage");
    const int n_frames = 2 * whisper_model_n_audio_ctx(ctx), n_mel = whisper_model_n_mels(ctx);
    std::vector<float> x;
    if (!stage) {                                  // conv2's input: conv1's GELU, computed by the shipped backend
        conv1_graph g1(tensors.at("encoder.conv1.weight"), tensors.at("encoder.conv1.bias"), n_frames, n_mel, true);
        g1.set_mel(mel.data, mel.n_len);
        g1.run(threads);
        x.assign((const float *)g1.out->data, (const float *)g1.out->data + ggml_nelements(g1.out));
    }
    conv2_graph g(tensors, n_frames, n_mel, stage);
    if (stage) g.set(mel.data, mel.n_len); else g.set(x, n_frames);
    size_t op_bytes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
        ggml_tensor * t = ggml_graph_node(g.gf, i);
        if (t->view_src == nullptr) op_bytes += ggml_nbytes(t);
    }
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-conv2-reference what %s threads %d wall_best_ms %.3f cpu_ms_per_call %.3f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld\n",
                what, threads, best, (c1 - c0) * 1000.0 / reps, reps, (op_bytes + 1023) / 1024, peak >= 0 ? peak - before : -1);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// ---- 0.0.8: the encoder's layer norms, observed in the shipped library ------------------------------------------------
// --norm <model.bin> <outdir> <wav ...> writes, per wav, <outdir>/<stem>/:
//   nodes.tsv        every node of the encoder graph (the eval callback on sched_encode observes them all, one at a time)
//   <chain>.in.f32   the NORM node's input (src[0]), read when the scheduler asks about the NORM node, i.e. after every
//                    node before it was computed and before the NORM ran (so an in-place NORM cannot have overwritten it)
//   <chain>.norm.f32 the NORM node; <chain>.mul.f32 the MUL after it (src[0] = the NORM, src[1] = the weight);
//   <chain>.add.f32  the ADD after that (src[0] = the MUL, src[1] = the bias). All f32 [384, 1500] (ne0 = channels).
//   chains: attn_ln_0, mlp_ln_0, ..., attn_ln_3, mlp_ln_3, ln_post (the encoder's nine norms, in graph order)
//   norm.tsv         eps (from the node's op_params), each chain's weight / bias tensor (by pointer identity with the
//                    model's tensor map), and the self-checks: 1 vs 4 threads bit-identical; embd_enc observed ==
//                    unobserved; ln_post's ADD == embd_enc; the standalone graphs (what --bench-norm times) == the nodes
struct norm_chain {
    std::string name, w, b;
    std::vector<uint8_t> in, norm, mul, add;
    const ggml_tensor * nt = nullptr, * mt = nullptr;
    float eps = -1.0f;
    bool contiguous = false;
};
struct norm_capture {
    std::vector<std::string> lines;
    std::vector<norm_chain> ch;
    int state = 0;          // 0: want NORM, 1: its MUL, 2: that MUL's ADD
    std::map<const ggml_tensor *, std::string> names;   // the model's tensors, by pointer
    int n_layer = 0;
};
static std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}
static bool norm_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<norm_capture *>(ud);
    if (ask) {
        if (t->op == GGML_OP_NORM) {                // the input, before the NORM runs
            norm_chain n;
            const int k = (int)c.ch.size();
            n.name = k == 2 * c.n_layer ? "ln_post" : std::string(k % 2 ? "mlp_ln_" : "attn_ln_") + std::to_string(k / 2);
            n.in = tensor_bytes(t->src[0]);
            n.contiguous = ggml_is_contiguous(t->src[0]) && t->src[0]->type == GGML_TYPE_F32;
            std::memcpy(&n.eps, t->op_params, sizeof(float));
            c.ch.push_back(std::move(n));
        }
        return true;                                // observe every node
    }
    cap_line(c.lines, "encode", t);
    if (t->op == GGML_OP_NORM) {
        check(c.state == 0 && !c.ch.empty(), "a NORM before the last one's MUL and ADD");
        c.ch.back().norm = tensor_bytes(t); c.ch.back().nt = t; c.state = 1;
    } else if (c.state == 1 && t->op == GGML_OP_MUL) {
        check(t->src[0] == c.ch.back().nt, "the MUL after a NORM does not read it");
        auto it = c.names.find(t->src[1]);
        c.ch.back().w = it == c.names.end() ? "?" : it->second;
        c.ch.back().mul = tensor_bytes(t); c.ch.back().mt = t; c.state = 2;
    } else if (c.state == 2 && t->op == GGML_OP_ADD) {
        check(t->src[0] == c.ch.back().mt, "the ADD after a NORM's MUL does not read it");
        auto it = c.names.find(t->src[1]);
        c.ch.back().b = it == c.names.end() ? "?" : it->second;
        c.ch.back().add = tensor_bytes(t); c.state = 0;
    } else {
        check(c.state == 0, "a NORM not followed by its MUL then its ADD");
    }
    return true;
}

// norm alone, or norm -> mul(w) -> add(b), as a standalone ggml graph on the shipped CPU backend (what --bench-norm times)
struct norm_graph {
    ggml_context * ctx; ggml_tensor * in; ggml_tensor * nrm; ggml_tensor * out; ggml_cgraph * gf;
    norm_graph(ggml_tensor * w_src, ggml_tensor * b_src, int n_state, int n_ctx, float eps, bool chain) {
        ggml_init_params p = { (size_t)32 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
        nrm = ggml_norm(ctx, in, eps);
        out = nrm;
        if (chain) {
            ggml_tensor * w = conv2_graph::copy(ctx, w_src), * b = conv2_graph::copy(ctx, b_src);
            out = ggml_add(ctx, ggml_mul(ctx, nrm, w), b);
        }
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    void set(const std::vector<uint8_t> & x) { check(x.size() == ggml_nbytes(in), "norm input size"); std::memcpy(in->data, x.data(), x.size()); }
    void run(int threads) { check(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "graph compute failed"); }
    ~norm_graph() { ggml_free(ctx); }
};
static std::string chain_prefix(const std::string & name) {   // attn_ln_2 -> encoder.blocks.2.attn_ln
    if (name == "ln_post") return "encoder.ln_post";
    const size_t u = name.find_last_of('_');
    return "encoder.blocks." + name.substr(u + 1) + "." + name.substr(0, u);
}

static int record_norm(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx), n_layer = whisper_model_n_audio_layer(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        norm_capture caps[2];
        std::vector<uint8_t> obs_enc, unobs_enc;
        const int threads[2] = {1, 4};
        for (int k = 0; k < 3; k++) {              // k = 0, 1: observed at 1 and 4 threads; k = 2: not observed, 1 thread
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            ggml_backend_sched_t se = state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            if (k < 2) {
                for (auto & kv : tensors) caps[k].names[kv.second] = kv.first;
                caps[k].n_layer = n_layer;
                ggml_backend_sched_set_eval_callback(se, norm_cb, &caps[k]);
            }
            check(whisper_encode_with_state(ctx, st, 0, k < 2 ? threads[k] : 1) == 0, "whisper_encode failed");
            if (k < 2) check((int)caps[k].ch.size() == 2 * n_layer + 1 && caps[k].state == 0, "the encoder graph did not show 2 x n_layer + 1 NORM -> MUL -> ADD chains");
            ggml_tensor * ee = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_ENC);
            check(ee != nullptr && ee->type == GGML_TYPE_F32 && ee->ne[0] == n_state, "layout check failed: embd_enc");
            if (k != 1) (k == 0 ? obs_enc : unobs_enc) = tensor_bytes(ee);
            whisper_free_state(st);
        }
        norm_capture & c = caps[0];
        const norm_capture & d = caps[1];
        bool thr = true, names_ok = true, eps_ok = true, contig = true;
        for (size_t i = 0; i < c.ch.size(); i++) {
            const norm_chain & x = c.ch[i], & y = d.ch[i];
            check(x.in.size() == (size_t)n_state * n_ctx * 4 && x.add.size() == x.in.size(), "unexpected norm shapes");
            thr = thr && x.in == y.in && x.norm == y.norm && x.mul == y.mul && x.add == y.add;
            names_ok = names_ok && x.w == chain_prefix(x.name) + ".weight" && x.b == chain_prefix(x.name) + ".bias";
            eps_ok = eps_ok && x.eps == 1e-5f;
            contig = contig && x.contiguous;
            write_bin(dir + "/" + x.name + ".in.f32", (const float *)x.in.data(), x.in.size() / 4);
            write_bin(dir + "/" + x.name + ".norm.f32", (const float *)x.norm.data(), x.norm.size() / 4);
            write_bin(dir + "/" + x.name + ".mul.f32", (const float *)x.mul.data(), x.mul.size() / 4);
            write_bin(dir + "/" + x.name + ".add.f32", (const float *)x.add.data(), x.add.size() / 4);
        }
        FILE * nf = std::fopen((dir + "/nodes.tsv").c_str(), "w");
        for (auto & l : c.lines) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        const bool enc_eq = obs_enc == unobs_enc;
        const bool post_eq = c.ch.back().add == obs_enc;
        bool alone = true;
        for (const norm_chain & x : c.ch) {        // the standalone graphs --bench-norm times, against the scheduler's nodes
            const std::string p = chain_prefix(x.name);
            for (int ch = 0; ch < 2; ch++) {
                norm_graph g(tensors.at(p + ".weight"), tensors.at(p + ".bias"), n_state, n_ctx, 1e-5f, ch == 1);
                g.set(x.in);
                for (int th : {1, 4}) { g.run(th); alone = alone && same(g.nrm, x.norm) && (ch == 0 || same(g.out, x.add)); }
            }
        }
        FILE * m = std::fopen((dir + "/norm.tsv").c_str(), "w");
        std::fprintf(m, "n_ctx\t%d\nn_state\t%d\nchains\t%zu\neps\t%a\nnodes\t%zu\neps_is_1e-5f\t%s\ninputs_contiguous_f32\t%s\n"
                        "weights_are_the_named_tensors\t%s\nthreads_1_vs_4_bit_identical\t%s\nembd_enc_observed_eq_unobserved\t%s\n"
                        "ln_post_add_eq_embd_enc\t%s\nstandalone_eq_sched\t%s\n", n_ctx, n_state, c.ch.size(), (double)c.ch[0].eps,
                     c.lines.size(), eps_ok ? "yes" : "NO", contig ? "yes" : "NO", names_ok ? "yes" : "NO", thr ? "yes" : "NO",
                     enc_eq ? "yes" : "NO", post_eq ? "yes" : "NO", alone ? "yes" : "NO");
        for (const norm_chain & x : c.ch) std::fprintf(m, "chain\t%s\t%s\t%s\n", x.name.c_str(), x.w.c_str(), x.b.c_str());
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: %zu encoder nodes observed; %zu norm -> mul -> add chains recorded; eps 1e-5f: %s; "
                             "the named weights: %s; 1 vs 4 threads identical: %s; embd_enc observed = unobserved: %s; ln_post = embd_enc: %s; "
                             "standalone = scheduler: %s\n", stem.c_str(), c.lines.size(), c.ch.size(), eps_ok ? "yes" : "NO",
                     names_ok ? "yes" : "NO", thr ? "yes" : "NO", enc_eq ? "yes" : "NO", post_eq ? "yes" : "NO", alone ? "yes" : "NO");
    }
    whisper_free(ctx);
    return 0;
}

// --bench-norm <model.bin> <wav> <threads> <what>: what = norm (the NORM node alone) or chain (norm -> * w + b, block 0's
// attn_ln weights), on the encoder's input of <wav> (the conv stage computed beforehand by the standalone graph above),
// through the standalone graph, in a fresh process. wall = best of 10, cpu = CPU ms per compute over >= 1 s, mem = the
// bytes of the graph's own non-view nodes (the norm, mul and add outputs), rss = VmHWM delta of the first compute.
static int bench_norm(const char * model_path, const char * wav, int threads, const char * what) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
    const bool chain = std::strcmp(what, "chain") == 0;
    check(chain || std::strcmp(what, "norm") == 0, "--bench-norm ... norm|chain");
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx);
    std::vector<uint8_t> x;
    {
        conv2_graph s(tensors, 2 * n_ctx, whisper_model_n_mels(ctx), true);
        s.set(mel.data, mel.n_len);
        s.run(threads);
        x.assign((const uint8_t *)s.out->data, (const uint8_t *)s.out->data + ggml_nbytes(s.out));
    }
    norm_graph g(tensors.at("encoder.blocks.0.attn_ln.weight"), tensors.at("encoder.blocks.0.attn_ln.bias"), n_state, n_ctx, 1e-5f, chain);
    g.set(x);
    size_t op_bytes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
        ggml_tensor * t = ggml_graph_node(g.gf, i);
        if (t->view_src == nullptr) op_bytes += ggml_nbytes(t);
    }
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-norm-reference what %s threads %d wall_best_ms %.4f cpu_ms_per_call %.4f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld\n",
                what, threads, best, (c1 - c0) * 1000.0 / reps, reps, (op_bytes + 1023) / 1024, peak >= 0 ? peak - before : -1);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// ---- 0.0.9: the matrix products on activations, observed in the shipped library --------------------------------------
// --matmul <model.bin> <outdir> <wav ...> writes, per wav, <outdir>/<stem>/ (compact: per-row digests, not tensors):
//   nodes.tsv          every node of the encoder graph (the eval callback on sched_encode observes them all)
//   b<il>.<key>.d64    for each block il and each node below, one 64-bit FNV-1a digest per row (digest32 / digest16 of
//                      the row's bits: 384 or 1536 f32, or 384 f16), u64 little-endian, 1500 rows:
//                        k_mm (MUL_MAT key.weight; K has no bias), k_cpy (CPY of k_mm into kv_pad.k, f16),
//                        v_mm, v_add (+ value.bias), v_cpy (CPY of v_add into kv_pad.v, f16), q_mm, q_add (+ query.bias),
//                        o_mm (MUL_MAT attn.out.weight on the attention's output), o_add (+ out.bias), o_res (+ the
//                        block's input), fc1_mm (mlp.0), fc1_add, gelu, fc2_mm (mlp.2), fc2_add, mlp_res (+ o_res)
//   b<il>.<key>.in.d64 each MUL_MAT's src1 (its f32 activations), digested per row when the scheduler asks about the
//                      MUL_MAT (every earlier node computed): q, k, v (attn_ln's ADD), o (the attention), fc1 (mlp_ln's
//                      ADD), fc2 (the GELU)
//   b<il>.fa.f32       the FLASH_ATTN_EXT node (f32 [64, 6, 1500] = [384, 1500] frame-major): o_mm's input, kept whole
//                      because voaice.rs does not compute attention yet
//   matmul.tsv         the self-checks: every block shows the 17 nodes; the MUL_MATs read the named f16 weights from a
//                      plain CPU buffer (not a repack buffer); src1 contiguous f32; no ADD reads K's MUL_MAT; the ADDs
//                      read the named biases; 1 vs 4 threads bit-identical (every digest and the attention output);
//                      embd_enc observed == unobserved; the standalone graphs (what --bench-mm times) == the nodes
static const char * MM_KEYS[17] = {"k_mm", "k_cpy", "v_mm", "v_add", "v_cpy", "q_mm", "q_add", "o_mm", "o_add", "o_res",
                                   "fc1_mm", "fc1_add", "gelu", "fc2_mm", "fc2_add", "mlp_res", "fa"};
static std::vector<uint64_t> row_digests(const ggml_tensor * t, int64_t row) {
    // a scheduler's tensor lives in a backend buffer; a standalone graph's in its context
    std::vector<uint8_t> b = t->buffer ? tensor_bytes(t) : std::vector<uint8_t>((const uint8_t *)t->data, (const uint8_t *)t->data + ggml_nbytes(t));
    const size_t es = ggml_type_size(t->type), rb = (size_t)row * es, n = b.size() / rb;
    check(b.size() % rb == 0, "row digests: not whole rows");
    std::vector<uint64_t> d(n);
    for (size_t r = 0; r < n; r++)
        d[r] = es == 4 ? digest32((const float *)(b.data() + r * rb), row) : digest16((const uint16_t *)(b.data() + r * rb), row);
    return d;
}
struct mm_capture {
    std::vector<std::string> lines;
    std::map<const ggml_tensor *, std::string> names;   // the model's tensors, by pointer
    std::map<const ggml_tensor *, std::string> key;     // node -> "<il>.<key>"
    std::map<std::string, std::vector<uint64_t>> d;     // "b<il>.<key>" -> row digests (and "...in")
    std::map<std::string, std::vector<uint8_t>> keep;   // "b<il>.fa", "b<il>.inp" (the block's input), kept whole
    int n_layer = 0, n_state = 0, fa_seen = 0;
    bool weights_named = true, plain_buffers = true, src1_f32 = true, biases_named = true, k_unbiased = true;
};
static std::string mm_weight_key(const std::string & w, int & il) {   // encoder.blocks.2.attn.query.weight -> q_mm, il 2
    const std::string p = "encoder.blocks.";
    if (w.compare(0, p.size(), p) != 0) return "";
    il = std::atoi(w.c_str() + p.size());
    const std::string rest = w.substr(w.find('.', p.size()) + 1);
    if (rest == "attn.query.weight") return "q_mm";
    if (rest == "attn.key.weight") return "k_mm";
    if (rest == "attn.value.weight") return "v_mm";
    if (rest == "attn.out.weight") return "o_mm";
    if (rest == "mlp.0.weight") return "fc1_mm";
    if (rest == "mlp.2.weight") return "fc2_mm";
    return "";
}
static bool mm_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<mm_capture *>(ud);
    auto src_key = [&](int i) -> std::string {
        auto it = t->src[i] ? c.key.find(t->src[i]) : c.key.end();
        return it == c.key.end() ? "" : it->second;
    };
    auto put = [&](const std::string & k) { c.key[t] = k; };
    if (ask) {   // inputs, before the node runs
        if (t->op == GGML_OP_MUL_MAT) {
            auto it = c.names.find(t->src[0]);
            int il = -1;
            const std::string k = it == c.names.end() ? "" : mm_weight_key(it->second, il);
            if (!k.empty()) {
                c.d["b" + std::to_string(il) + "." + k.substr(0, k.size() - 3) + ".in"] = row_digests(t->src[1], t->src[1]->ne[0]);
                c.src1_f32 = c.src1_f32 && t->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(t->src[1]);
            }
        } else if (t->op == GGML_OP_ADD) {
            const std::string s = src_key(0);
            if (s.size() > 6 && s.compare(s.size() - 6, 6, ".o_add") == 0)   // the residual: src[1] is the block's input
                c.keep["b" + s.substr(0, s.find('.')) + ".inp"] = tensor_bytes(t->src[1]);
        }
        return true;
    }
    cap_line(c.lines, "encode", t);
    std::string k;   // "<il>.<key>" of this node, if it is one of ours
    if (t->op == GGML_OP_MUL_MAT) {
        auto it = c.names.find(t->src[0]);
        int il = -1;
        const std::string w = it == c.names.end() ? "" : mm_weight_key(it->second, il);
        if (!w.empty()) {
            k = std::to_string(il) + "." + w;
            c.plain_buffers = c.plain_buffers && t->src[0]->buffer && std::strcmp(ggml_backend_buffer_name(t->src[0]->buffer), "CPU") == 0;
            c.weights_named = c.weights_named && t->src[0]->type == GGML_TYPE_F16;
        }
    } else if (t->op == GGML_OP_ADD) {
        const std::string s = src_key(0);
        const std::string il = s.substr(0, s.find('.')), sk = s.empty() ? "" : s.substr(s.find('.') + 1);
        auto bias = [&](const char * name) {
            auto it = c.names.find(t->src[1]);
            c.biases_named = c.biases_named && it != c.names.end() && it->second == "encoder.blocks." + il + "." + name;
        };
        if (sk == "k_mm") c.k_unbiased = false;
        else if (sk == "q_mm") { k = il + ".q_add"; bias("attn.query.bias"); }
        else if (sk == "v_mm") { k = il + ".v_add"; bias("attn.value.bias"); }
        else if (sk == "o_mm") { k = il + ".o_add"; bias("attn.out.bias"); }
        else if (sk == "o_add") k = il + ".o_res";
        else if (sk == "fc1_mm") { k = il + ".fc1_add"; bias("mlp.0.bias"); }
        else if (sk == "fc2_mm") { k = il + ".fc2_add"; bias("mlp.2.bias"); }
        else if (sk == "fc2_add") { k = il + ".mlp_res"; check(src_key(1) == il + ".o_res", "the MLP's residual does not read o_res"); }
    } else if (t->op == GGML_OP_CPY) {
        const std::string s = src_key(0), il = s.substr(0, s.find('.'));
        if (!s.empty() && s.substr(s.find('.') + 1) == "k_mm") k = il + ".k_cpy";
        if (!s.empty() && s.substr(s.find('.') + 1) == "v_add") k = il + ".v_cpy";
    } else if (is_gelu(t)) {
        const std::string s = src_key(0);
        if (!s.empty() && s.substr(s.find('.') + 1) == "fc1_add") k = s.substr(0, s.find('.')) + ".gelu";
    } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        k = std::to_string(c.fa_seen++) + ".fa";
        c.keep["b" + k] = tensor_bytes(t);
    }
    if (k.empty()) return true;
    put(k);
    const std::string file = "b" + k;
    if (t->op != GGML_OP_FLASH_ATTN_EXT) c.d[file] = row_digests(t, t->op == GGML_OP_CPY ? c.n_state : t->ne[0]);
    else c.d[file] = row_digests(t, c.n_state);
    return true;
}

// the block's products as a standalone ggml graph on the shipped CPU backend (what --bench-mm times), the same ops in
// the same order as whisper_build_graph_encoder builds them (attention left out):
//   q    in = attn_ln's output [384, n]     -> q_mm, q_add
//   fc1  in = mlp_ln's output              -> fc1_mm, fc1_add, gelu
//   fc2  in = the GELU [1536, n]           -> fc2_mm, fc2_add
//   qkv  in = the block's input            -> norm * w + b, then k_mm, k_cpy, v_mm, v_add, v_cpy, q_mm, q_add
//   mlp  in = the block's input, att = the attention's output -> o_mm, o_add, o_res, norm * w + b, fc1.., fc2.., mlp_res
//   block = qkv then mlp with att = v_add (V standing in for the attention, which is v0.1.0's)
struct mm_graph {
    ggml_context * ctx; ggml_tensor * in = nullptr; ggml_tensor * att = nullptr; ggml_cgraph * gf;
    std::map<std::string, ggml_tensor *> node;
    mm_graph(std::map<std::string, ggml_tensor *> & m, int il, const std::string & what, int n_state, int n_ctx) {
        ggml_init_params p = { (size_t)160 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        const std::string pre = "encoder.blocks." + std::to_string(il) + ".";
        auto W = [&](const char * n) { return conv2_graph::copy(ctx, m.at(pre + n)); };
        gf = ggml_new_graph(ctx);
        auto ln = [&](ggml_tensor * x, const char * w) {
            return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), W((std::string(w) + ".weight").c_str())), W((std::string(w) + ".bias").c_str()));
        };
        auto qkv = [&](ggml_tensor * cur) {
            node["k_mm"] = ggml_mul_mat(ctx, W("attn.key.weight"), cur);
            node["k_cpy"] = ggml_cpy(ctx, node["k_mm"], ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_ctx));
            ggml_build_forward_expand(gf, node["k_cpy"]);
            node["v_mm"] = ggml_mul_mat(ctx, W("attn.value.weight"), cur);
            node["v_add"] = ggml_add(ctx, node["v_mm"], W("attn.value.bias"));
            node["v_cpy"] = ggml_cpy(ctx, node["v_add"], ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_ctx));
            ggml_build_forward_expand(gf, node["v_cpy"]);
            node["q_mm"] = ggml_mul_mat(ctx, W("attn.query.weight"), cur);
            node["q_add"] = ggml_add(ctx, node["q_mm"], W("attn.query.bias"));
            ggml_build_forward_expand(gf, node["q_add"]);
        };
        auto fc1 = [&](ggml_tensor * cur) {
            node["fc1_mm"] = ggml_mul_mat(ctx, W("mlp.0.weight"), cur);
            node["fc1_add"] = ggml_add(ctx, node["fc1_mm"], W("mlp.0.bias"));
            node["gelu"] = ggml_gelu(ctx, node["fc1_add"]);
            return node["gelu"];
        };
        auto fc2 = [&](ggml_tensor * cur) {
            node["fc2_mm"] = ggml_mul_mat(ctx, W("mlp.2.weight"), cur);
            node["fc2_add"] = ggml_add(ctx, node["fc2_mm"], W("mlp.2.bias"));
            return node["fc2_add"];
        };
        auto mlp = [&](ggml_tensor * a, ggml_tensor * inp) {
            node["o_mm"] = ggml_mul_mat(ctx, W("attn.out.weight"), a);
            node["o_add"] = ggml_add(ctx, node["o_mm"], W("attn.out.bias"));
            node["o_res"] = ggml_add(ctx, node["o_add"], inp);
            node["mlp_res"] = ggml_add(ctx, fc2(fc1(ln(node["o_res"], "mlp_ln"))), node["o_res"]);
            ggml_build_forward_expand(gf, node["mlp_res"]);
        };
        if (what == "q") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            node["q_mm"] = ggml_mul_mat(ctx, W("attn.query.weight"), in);
            node["q_add"] = ggml_add(ctx, node["q_mm"], W("attn.query.bias"));
            ggml_build_forward_expand(gf, node["q_add"]);
        } else if (what == "fc1") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            ggml_build_forward_expand(gf, fc1(in));
        } else if (what == "fc2") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * n_state, n_ctx);
            ggml_build_forward_expand(gf, fc2(in));
        } else if (what == "qkv") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            qkv(ln(in, "attn_ln"));
        } else if (what == "mlp") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            att = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            mlp(att, in);
        } else if (what == "block") {
            in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
            qkv(ln(in, "attn_ln"));
            mlp(node["v_add"], in);
        } else {
            die("--bench-mm ... q|fc1|fc2|qkv|mlp|block");
        }
    }
    static void set(ggml_tensor * t, const std::vector<uint8_t> & x) { check(x.size() == ggml_nbytes(t), "mm graph input size"); std::memcpy(t->data, x.data(), x.size()); }
    // ggml_graph_compute_with_ctx would take a new work buffer from ctx on every call: the same plan and compute, the
    // work buffer (the converted activations) allocated once and reused
    std::vector<uint8_t> work;
    void run(int threads) {
        ggml_cplan cp = ggml_graph_plan(gf, threads, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
    }
    size_t op_bytes() const {   // the graph's own non-view nodes and the work buffer
        size_t b = work.size();
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->view_src == nullptr) b += ggml_nbytes(t);
        }
        return b;
    }
    ~mm_graph() { ggml_free(ctx); }
};

static int record_matmul(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx), n_layer = whisper_model_n_audio_layer(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        mm_capture caps[2];
        std::vector<uint8_t> obs_enc, unobs_enc;
        const int threads[2] = {1, 4};
        for (int k = 0; k < 3; k++) {              // k = 0, 1: observed at 1 and 4 threads; k = 2: not observed, 1 thread
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            ggml_backend_sched_t se = state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            if (k < 2) {
                for (auto & kv : tensors) caps[k].names[kv.second] = kv.first;
                caps[k].n_layer = n_layer;
                caps[k].n_state = n_state;
                ggml_backend_sched_set_eval_callback(se, mm_cb, &caps[k]);
            }
            check(whisper_encode_with_state(ctx, st, 0, k < 2 ? threads[k] : 1) == 0, "whisper_encode failed");
            ggml_tensor * ee = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_ENC);
            check(ee != nullptr && ee->type == GGML_TYPE_F32 && ee->ne[0] == n_state, "layout check failed: embd_enc");
            if (k != 1) (k == 0 ? obs_enc : unobs_enc) = tensor_bytes(ee);
            whisper_free_state(st);
        }
        mm_capture & c = caps[0];
        const mm_capture & d = caps[1];
        bool all17 = c.fa_seen == n_layer, thr = c.d == d.d && c.keep == d.keep;
        for (int il = 0; il < n_layer; il++)
            for (const char * key : MM_KEYS) all17 = all17 && c.d.count("b" + std::to_string(il) + "." + key) == 1;
        for (const char * in : {"q", "k", "v", "o", "fc1", "fc2"})
            for (int il = 0; il < n_layer; il++) all17 = all17 && c.d.count("b" + std::to_string(il) + "." + in + ".in") == 1;
        check(all17, "the encoder graph did not show every block's 17 nodes and 6 products");
        for (auto & kv : c.d) {
            check(kv.second.size() == (size_t)n_ctx, "unexpected row count");
            write_bin(dir + "/" + kv.first + ".d64", kv.second.data(), kv.second.size());
        }
        for (int il = 0; il < n_layer; il++) {
            const std::vector<uint8_t> & fa = c.keep.at("b" + std::to_string(il) + ".fa");
            write_bin(dir + "/b" + std::to_string(il) + ".fa.f32", (const float *)fa.data(), fa.size() / 4);
        }
        FILE * nf = std::fopen((dir + "/nodes.tsv").c_str(), "w");
        for (auto & l : c.lines) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        const bool enc_eq = obs_enc == unobs_enc;
        // the standalone graphs --bench-mm times, against the scheduler's nodes: qkv from the block's input, mlp from the
        // block's input and the recorded attention output, at 1 and 4 threads, every node's row digests
        bool alone = true;
        for (int il = 0; il < n_layer; il++) {
            const std::string b = "b" + std::to_string(il) + ".";
            for (const char * what : {"qkv", "mlp"}) {
                mm_graph g(tensors, il, what, n_state, n_ctx);
                mm_graph::set(g.in, c.keep.at(b + "inp"));
                if (g.att) mm_graph::set(g.att, c.keep.at(b + "fa"));
                for (int th : {1, 4}) {
                    g.run(th);
                    for (auto & kv : g.node)
                        alone = alone && row_digests(kv.second, kv.second->op == GGML_OP_CPY ? n_state : kv.second->ne[0]) == c.d.at(b + kv.first);
                }
            }
        }
        FILE * m = std::fopen((dir + "/matmul.tsv").c_str(), "w");
        std::fprintf(m, "n_ctx\t%d\nn_state\t%d\nn_layer\t%d\nnodes\t%zu\nevery_block_17_nodes\tyes\nweights_f16_named\t%s\n"
                        "weights_in_plain_cpu_buffers\t%s\nsrc1_contiguous_f32\t%s\nbiases_named\t%s\nk_has_no_bias\t%s\n"
                        "threads_1_vs_4_bit_identical\t%s\nembd_enc_observed_eq_unobserved\t%s\nstandalone_eq_sched\t%s\n",
                     n_ctx, n_state, n_layer, c.lines.size(), c.weights_named ? "yes" : "NO", c.plain_buffers ? "yes" : "NO",
                     c.src1_f32 ? "yes" : "NO", c.biases_named ? "yes" : "NO", c.k_unbiased ? "yes" : "NO", thr ? "yes" : "NO",
                     enc_eq ? "yes" : "NO", alone ? "yes" : "NO");
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: %zu encoder nodes observed; %d blocks x 17 nodes digested; f16 weights, plain CPU "
                             "buffers: %s; K unbiased: %s; biases named: %s; 1 vs 4 threads identical: %s; embd_enc observed = "
                             "unobserved: %s; standalone = scheduler: %s\n", stem.c_str(), c.lines.size(), n_layer,
                     c.weights_named && c.plain_buffers ? "yes" : "NO", c.k_unbiased ? "yes" : "NO", c.biases_named ? "yes" : "NO",
                     thr ? "yes" : "NO", enc_eq ? "yes" : "NO", alone ? "yes" : "NO");
    }
    whisper_free(ctx);
    return 0;
}

// --mm-nan <model.bin> <outdir>: from_float on NaN-bearing activation rows. The real inputs carry no NaN, and on every
// other f32 the row converter (vcvtps2ph) and the portable bit trick agree (0.0.3), so this is the only place the
// converter can be told apart. Rows of encoder.positional_embedding (finite), each with one NaN at a position chosen so
// that some thread counts convert it in an 8-block, a 4-block or the scalar tail of their element range; through the
// standalone graph mul_mat(block 0's query.weight, x) + query.bias at 1..8 threads. Writes nan.in.f32 ([384, R]),
// nan.mm.t<n>.f32 and nan.add.t<n>.f32 ([384, R] each), nan.tsv (the positions and payloads).
static int record_mm_nan(const char * model_path, const std::string & outdir) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_state = whisper_model_n_audio_state(ctx);
    struct nanrow { int pos; uint32_t bits; };
    // 152, 229, 306, 383: the scalar tail of thread 1..4's range at 5 threads; 148..151: a 4-block there; 52, 53: the
    // scalar tail of thread 0 at 7 threads; 8: an 8-block at every count. Payloads: high bits (vcvtps2ph keeps them, the
    // trick does not), negative, signalling with a high payload, signalling with only low bits (both give 0x7E00).
    const nanrow rows[] = {{152, 0x7FC12345u}, {152, 0xFFD0F000u}, {229, 0x7FBFFFFFu}, {306, 0x7FC7E000u}, {383, 0xFFFFE000u},
                           {150, 0x7FC12345u}, {53, 0x7FD5A000u}, {52, 0xFFC40000u}, {8, 0x7FC12345u}, {8, 0x7F801234u},
                           {-1, 0}, {-1, 0}};
    const int R = (int)(sizeof rows / sizeof rows[0]);
    std::vector<float> x((size_t)n_state * R);
    ggml_tensor * pe = tensors.at("encoder.positional_embedding");
    std::vector<uint8_t> peb = tensor_bytes(pe);
    for (int r = 0; r < R; r++) {
        std::memcpy(&x[(size_t)r * n_state], peb.data() + (size_t)(17 * r + 3) * n_state * 4, (size_t)n_state * 4);
        if (rows[r].pos >= 0) std::memcpy(&x[(size_t)r * n_state + rows[r].pos], &rows[r].bits, 4);
    }
    write_bin(outdir + "/nan.in.f32", x.data(), x.size());
    ggml_init_params p = { (size_t)16 << 20, nullptr, false };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, n_state, R);
    std::memcpy(in->data, x.data(), x.size() * 4);
    ggml_tensor * mm = ggml_mul_mat(gc, conv2_graph::copy(gc, tensors.at("encoder.blocks.0.attn.query.weight")), in);
    ggml_tensor * add = ggml_add(gc, mm, conv2_graph::copy(gc, tensors.at("encoder.blocks.0.attn.query.bias")));
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, add);
    FILE * m = std::fopen((outdir + "/nan.tsv").c_str(), "w");
    std::vector<uint8_t> work;
    std::fprintf(m, "rows\t%d\nn_state\t%d\n", R, n_state);
    for (int r = 0; r < R; r++) std::fprintf(m, "row\t%d\t%d\t%08x\n", r, rows[r].pos, rows[r].bits);
    for (int th = 1; th <= 8; th++) {
        ggml_cplan cp = ggml_graph_plan(gf, th, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
        check(cp.n_threads == th, "the plan did not take the thread count");
        write_bin(outdir + "/nan.mm.t" + std::to_string(th) + ".f32", (const float *)mm->data, (size_t)n_state * R);
        write_bin(outdir + "/nan.add.t" + std::to_string(th) + ".f32", (const float *)add->data, (size_t)n_state * R);
        std::fprintf(m, "threads\t%d\n", th);
    }
    std::fclose(m);
    std::fprintf(stderr, "whisper_oracle: mm-nan: %d rows (%d with a NaN), query.weight x rows + bias at 1..8 threads recorded\n",
                 R, R - 2);
    ggml_free(gc);
    whisper_free(ctx);
    return 0;
}

// --bench-mm <model.bin> <wav> <threads> <what>: block 0's products through the standalone graph above, in a fresh
// process (what = q | fc1 | fc2 | qkv | mlp | block). The inputs are computed beforehand from <wav> by the standalone
// graphs of 0.0.7 / 0.0.8 (X = the encoder's input): q, fc1 read attn_ln_0(X) (fc1's stand-in for mlp_ln's output, the
// same shape); fc2 reads gelu(fc1(attn_ln_0(X))); qkv reads X; mlp reads X and V's output (v_add, standing in for
// the attention, which voaice.rs does not compute yet); block reads X. wall = best of 10, cpu = CPU ms per compute over
// >= 1 s, mem = the bytes of the graph's own non-view nodes, rss = VmHWM delta of the first compute.
static int bench_mm(const char * model_path, const char * wav, int threads, const char * what) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx);
    const std::string w = what;
    std::vector<uint8_t> X, LN, V, G;
    {
        conv2_graph s(tensors, 2 * n_ctx, whisper_model_n_mels(ctx), true);
        s.set(mel.data, mel.n_len);
        s.run(threads);
        X.assign((const uint8_t *)s.out->data, (const uint8_t *)s.out->data + ggml_nbytes(s.out));
    }
    {
        norm_graph g(tensors.at("encoder.blocks.0.attn_ln.weight"), tensors.at("encoder.blocks.0.attn_ln.bias"), n_state, n_ctx, 1e-5f, true);
        g.set(X);
        g.run(threads);
        LN.assign((const uint8_t *)g.out->data, (const uint8_t *)g.out->data + ggml_nbytes(g.out));
    }
    if (w == "fc2") {
        mm_graph g(tensors, 0, "fc1", n_state, n_ctx);
        mm_graph::set(g.in, LN);
        g.run(threads);
        ggml_tensor * t = g.node.at("gelu");
        G.assign((const uint8_t *)t->data, (const uint8_t *)t->data + ggml_nbytes(t));
    }
    if (w == "mlp") {
        mm_graph g(tensors, 0, "qkv", n_state, n_ctx);
        mm_graph::set(g.in, X);
        g.run(threads);
        ggml_tensor * t = g.node.at("v_add");
        V.assign((const uint8_t *)t->data, (const uint8_t *)t->data + ggml_nbytes(t));
    }
    mm_graph g(tensors, 0, w, n_state, n_ctx);
    mm_graph::set(g.in, w == "q" || w == "fc1" ? LN : w == "fc2" ? G : X);
    if (g.att) mm_graph::set(g.att, V);
    g.run(threads);                                // the work buffer sized before the measurement (as voaice's first call)
    const size_t op_bytes = g.op_bytes();
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-mm-reference what %s threads %d wall_best_ms %.4f cpu_ms_per_call %.4f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld\n",
                what, threads, best, (c1 - c0) * 1000.0 / reps, reps, (op_bytes + 1023) / 1024, peak >= 0 ? peak - before : -1);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// ---- v0.1.0: flash attention and the whole encoder, observed in the shipped library ----------------------------------
// --encoder <model.bin> <outdir> <wav ...> writes, per wav, <outdir>/<stem>/ (compact: per-row digests, not tensors):
//   embd_enc.f32       whisper_state::embd_enc (layout probe) after an unobserved whisper_encode_with_state at 1 thread:
//                      f32 [384, 1500] frame-major, the encoder's output; also computed at 2 and 4 threads and compared
//   enc.d64            for EVERY computed node of the encoder graph (views, reshapes, permutes and transposes are not
//                      computed and are left out), in the order the scheduler ran them, one 64-bit FNV-1a digest per
//                      row (row = ne0 f32 values; a CPY into kv_pad: 384 f16; FLASH_ATTN_EXT: 384 = 64 x 6 heads)
//   enc.tsv            one line per digested node: index in the graph, key, op, type, row length, rows, offset (in
//                      u64) into enc.d64. Keys: pe_add, attn_ln_<il>.{norm,mul,add}, mlp_ln_<il>.*, ln_post.*, and
//                      0.0.9's b<il>.<key> (k_mm, k_cpy, v_mm, v_add, v_cpy, q_mm, q_add, fa, o_mm, ..., mlp_res)
//   b<il>.fa_ref.d64   the same FLASH_ATTN_EXT recomputed by the shipped CPU backend with cplan.use_ref = true (the
//                      one-chunk path: Q converted to f16, ggml_vec_dot_f16 scores, V accumulated in f16), row digests:
//                      the reference's own other reading, which whisper's graph does not take
//   encoder.tsv        the self-checks: every FLASH_ATTN_EXT reads Q f32 (a permuted view of q_add), K and V f16 views
//                      of kv_pad with ne [64, 1536, 6] and strides [2, 768, 128], no mask, no sinks, scale 0.125, max_bias
//                      and softcap 0, default precision; kv_pad's rows 1500..1535 all +0 (K and V, every block, read
//                      when the scheduler asks about the node); 1 vs 4 threads every node identical; embd_enc observed ==
//                      unobserved, and 1 = 2 = 4 threads unobserved; the last node == embd_enc; the standalone
//                      flash-attention graph (what --bench-attn times) == the scheduler's node at 1..8 threads; how
//                      many values the use_ref path changes
static const char * FA_REF = "fa_ref";
struct encoder_capture {
    mm_capture mm;                                       // 0.0.9's node keys (and the products' input digests)
    std::vector<std::string> tsv;                        // enc.tsv
    std::vector<uint64_t> digests;                       // enc.d64
    std::map<std::string, std::vector<uint64_t>> by_key; // key -> row digests (for the 1 vs 4 comparison)
    int norm_seen = 0, norm_state = 0, n_layer = 0, n_state = 0, n_ctx = 0, idx = 0;
    std::string chain;
    bool fa_shape = true, pad_zero = true;
    std::vector<std::vector<uint8_t>> fa_q, fa_k, fa_v, fa_out;   // per block: q_add, kv_pad.k and .v whole, the node
    std::vector<uint8_t> last;
};
static bool encoder_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<encoder_capture *>(ud);
    mm_cb(t, ask, &c.mm);
    if (ask) {
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {               // its inputs, every earlier node computed
            const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2];
            float op[3];
            std::memcpy(op, t->op_params, sizeof op);
            const int32_t prec = t->op_params[3];
            bool ok = q->type == GGML_TYPE_F32 && q->view_src && q->view_src->type == GGML_TYPE_F32 &&
                      q->ne[0] == 64 && q->ne[1] == c.n_ctx && q->ne[2] == c.n_state / 64 && q->nb[1] == (size_t)c.n_state * 4 && q->nb[2] == 64 * 4;
            for (const ggml_tensor * x : {k, v})
                ok = ok && x->type == GGML_TYPE_F16 && x->view_src && x->ne[0] == 64 && x->ne[1] == 1536 && x->ne[2] == c.n_state / 64 &&
                     x->nb[0] == 2 && x->nb[1] == (size_t)c.n_state * 2 && x->nb[2] == 128 && x->view_offs == 0 &&
                     ggml_nelements(x->view_src) == (int64_t)1536 * c.n_state;
            ok = ok && t->src[3] == nullptr && t->src[4] == nullptr && op[0] == 0.125f && op[1] == 0.0f && op[2] == 0.0f && prec == GGML_PREC_DEFAULT;
            c.fa_shape = c.fa_shape && ok;
            c.fa_q.push_back(tensor_bytes(q->view_src));
            c.fa_k.push_back(tensor_bytes(k->view_src));
            c.fa_v.push_back(tensor_bytes(v->view_src));
            for (const std::vector<uint8_t> * b : {&c.fa_k.back(), &c.fa_v.back()})
                for (size_t i = (size_t)c.n_ctx * c.n_state * 2; i < b->size(); i++) c.pad_zero = c.pad_zero && (*b)[i] == 0;
        }
        return true;
    }
    const int idx = c.idx++;
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE)
        return true;
    // the key: a norm chain's node, 0.0.9's key, the positional add (the first ADD), or the node's index
    std::string key;
    auto it = c.mm.key.find(t);
    if (it != c.mm.key.end()) key = "b" + it->second;
    else if (t->op == GGML_OP_NORM) {
        const int k = c.norm_seen++;
        c.chain = k == 2 * c.n_layer ? "ln_post" : std::string(k % 2 ? "mlp_ln_" : "attn_ln_") + std::to_string(k / 2);
        key = c.chain + ".norm"; c.norm_state = 1;
    } else if (c.norm_state == 1 && t->op == GGML_OP_MUL) { key = c.chain + ".mul"; c.norm_state = 2; }
    else if (c.norm_state == 2 && t->op == GGML_OP_ADD) { key = c.chain + ".add"; c.norm_state = 0; }
    else if (t->op == GGML_OP_ADD && c.norm_seen == 0) key = "pe_add";
    else key = "n" + std::to_string(idx);
    const int64_t row = t->op == GGML_OP_CPY ? c.n_state : t->op == GGML_OP_FLASH_ATTN_EXT ? t->ne[0] * t->ne[1] : t->ne[0];
    std::vector<uint64_t> d = row_digests(t, row);
    if (t->op == GGML_OP_FLASH_ATTN_EXT) c.fa_out.push_back(tensor_bytes(t));
    char line[256];
    std::snprintf(line, sizeof line, "%d\t%s\t%s\t%s\t%lld\t%zu\t%zu", idx, key.c_str(), ggml_op_desc(t), ggml_type_name(t->type),
                  (long long)row, d.size(), c.digests.size());
    c.tsv.push_back(line);
    c.digests.insert(c.digests.end(), d.begin(), d.end());
    c.by_key[key] = d;
    c.last = tensor_bytes(t);
    return true;
}

// flash attention alone as whisper builds it, as a standalone graph on the shipped CPU backend (what --bench-attn times):
// Q = permute(reshape_3d(q [384, n_ctx] f32, 64, 6, n_ctx), 0, 2, 1, 3); K, V = view_3d of a [1536 x 384] f16 buffer
struct fa_graph {
    ggml_context * ctx; ggml_tensor * q, * k, * v, * out; ggml_cgraph * gf;
    std::vector<uint8_t> work;
    fa_graph(int n_state, int n_ctx, int n_pad) {
        ggml_init_params p = { (size_t)32 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        const int nh = n_state / 64;
        q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
        k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_pad);
        v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_pad);
        ggml_tensor * Q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, 64, nh, n_ctx), 0, 2, 1, 3);
        ggml_tensor * K = ggml_view_3d(ctx, k, 64, n_pad, nh, 2 * n_state, 2 * 64, 0);
        ggml_tensor * V = ggml_view_3d(ctx, v, 64, n_pad, nh, 2 * n_state, 2 * 64, 0);
        out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f / sqrtf(64.0f), 0.0f, 0.0f);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    void set(const std::vector<uint8_t> & qb, const std::vector<uint8_t> & kb, const std::vector<uint8_t> & vb) {
        check(qb.size() == ggml_nbytes(q) && kb.size() == ggml_nbytes(k) && vb.size() == ggml_nbytes(v), "fa graph input size");
        std::memcpy(q->data, qb.data(), qb.size()); std::memcpy(k->data, kb.data(), kb.size()); std::memcpy(v->data, vb.data(), vb.size());
    }
    void run(int threads, bool use_ref = false) {
        ggml_cplan cp = ggml_graph_plan(gf, threads, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        cp.use_ref = use_ref;
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
        check(cp.n_threads == threads, "the plan did not take the thread count");
    }
    ~fa_graph() { ggml_free(ctx); }
};

static ggml_tensor * state_embd_enc(whisper_state * st, int n_state) {
    ggml_tensor * ee = *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_EMBD_ENC);
    check(ee != nullptr && ee->type == GGML_TYPE_F32 && ee->ne[0] == n_state, "layout check failed: embd_enc");
    return ee;
}

static int record_encoder(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx), n_layer = whisper_model_n_audio_layer(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        encoder_capture caps[2];
        std::vector<uint8_t> obs_enc, unobs[3];
        const int threads[5] = {1, 4, 1, 2, 4};
        for (int k = 0; k < 5; k++) {              // k = 0, 1: observed at 1 and 4 threads; k = 2..4: not observed, 1, 2, 4
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            ggml_backend_sched_t se = state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            if (k < 2) {
                for (auto & kv : tensors) caps[k].mm.names[kv.second] = kv.first;
                caps[k].mm.n_layer = caps[k].n_layer = n_layer;
                caps[k].mm.n_state = caps[k].n_state = n_state;
                caps[k].n_ctx = n_ctx;
                ggml_backend_sched_set_eval_callback(se, encoder_cb, &caps[k]);
            }
            check(whisper_encode_with_state(ctx, st, 0, threads[k]) == 0, "whisper_encode failed");
            std::vector<uint8_t> e = tensor_bytes(state_embd_enc(st, n_state));
            if (k == 0) obs_enc = e; else if (k >= 2) unobs[k - 2] = e;
            whisper_free_state(st);
        }
        encoder_capture & c = caps[0];
        const encoder_capture & d = caps[1];
        check((int)c.fa_out.size() == n_layer && c.norm_seen == 2 * n_layer + 1, "the encoder graph did not show n_layer attentions and 2 x n_layer + 1 norms");
        const bool thr = c.digests == d.digests && c.tsv == d.tsv && c.fa_out == d.fa_out;
        const bool enc_eq = obs_enc == unobs[0], enc_thr = unobs[0] == unobs[1] && unobs[0] == unobs[2], last_eq = c.last == obs_enc;
        write_bin(dir + "/embd_enc.f32", (const float *)unobs[0].data(), unobs[0].size() / 4);
        write_bin(dir + "/enc.d64", c.digests.data(), c.digests.size());
        FILE * nf = std::fopen((dir + "/enc.tsv").c_str(), "w");
        for (auto & l : c.tsv) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        // the standalone graph against the scheduler's node at 1..8 threads; then the use_ref path, 1 thread
        bool alone = true;
        size_t ref_values = 0, ref_rows = 0;
        for (int il = 0; il < n_layer; il++) {
            fa_graph g(n_state, n_ctx, 1536);
            g.set(c.fa_q[il], c.fa_k[il], c.fa_v[il]);
            for (int th = 1; th <= 8; th++) { g.run(th); alone = alone && same(g.out, c.fa_out[il]); }
            g.run(1, true);
            std::vector<uint64_t> rd = row_digests(g.out, n_state);
            write_bin(dir + "/b" + std::to_string(il) + "." + FA_REF + ".d64", rd.data(), rd.size());
            const float * x = (const float *)g.out->data, * y = (const float *)c.fa_out[il].data();
            for (size_t i = 0; i < (size_t)n_state * n_ctx; i++) ref_values += std::memcmp(x + i, y + i, 4) != 0;
            const std::vector<uint64_t> & sd = c.by_key.at("b" + std::to_string(il) + ".fa");
            for (size_t r = 0; r < rd.size(); r++) ref_rows += rd[r] != sd[r];
        }
        FILE * m = std::fopen((dir + "/encoder.tsv").c_str(), "w");
        std::fprintf(m, "n_ctx\t%d\nn_state\t%d\nn_layer\t%d\nnodes_digested\t%zu\nfa_inputs_as_whisper_builds_them\t%s\n"
                        "kv_pad_rows_1500_1535_all_zero\t%s\nthreads_1_vs_4_bit_identical\t%s\nembd_enc_observed_eq_unobserved\t%s\n"
                        "embd_enc_1_2_4_threads_identical\t%s\nlast_node_eq_embd_enc\t%s\nstandalone_fa_eq_sched_1_to_8_threads\t%s\n"
                        "use_ref_values_differing\t%zu\nuse_ref_rows_differing\t%zu\n",
                     n_ctx, n_state, n_layer, c.tsv.size(), c.fa_shape ? "yes" : "NO", c.pad_zero ? "yes" : "NO", thr ? "yes" : "NO",
                     enc_eq ? "yes" : "NO", enc_thr ? "yes" : "NO", last_eq ? "yes" : "NO", alone ? "yes" : "NO", ref_values, ref_rows);
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: %zu encoder nodes digested; attention inputs as built: %s; kv_pad padding +0: %s; "
                             "1 vs 4 threads identical: %s; embd_enc observed = unobserved: %s, 1 = 2 = 4 threads: %s, = last node: %s; "
                             "standalone attention = scheduler at 1..8 threads: %s; use_ref changes %zu values (%zu rows)\n",
                     stem.c_str(), c.tsv.size(), c.fa_shape ? "yes" : "NO", c.pad_zero ? "yes" : "NO", thr ? "yes" : "NO",
                     enc_eq ? "yes" : "NO", enc_thr ? "yes" : "NO", last_eq ? "yes" : "NO", alone ? "yes" : "NO", ref_values, ref_rows);
    }
    whisper_free(ctx);
    return 0;
}

// --bench-attn <model.bin> <wav> <threads>: block 0's flash attention through the standalone graph above, its Q, K, V
// computed beforehand from <wav> by the standalone graphs of 0.0.7 - 0.0.9 (K and V into a zeroed 1536-row f16 buffer, as
// kv_pad), in a fresh process. wall = best of 10, cpu = CPU ms per compute over >= 1 s, mem = the node's output + the
// K/V buffers + the work buffer, rss = VmHWM delta of the first compute.
static int bench_attn(const char * model_path, const char * wav, int threads) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    auto & mel = *reinterpret_cast<mel_mirror *>((char *)st + VOAICE_OFF_STATE_MEL);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_audio_state(ctx);
    std::vector<uint8_t> X, Q, K((size_t)1536 * n_state * 2, 0), V((size_t)1536 * n_state * 2, 0);
    {
        conv2_graph s(tensors, 2 * n_ctx, whisper_model_n_mels(ctx), true);
        s.set(mel.data, mel.n_len);
        s.run(threads);
        X.assign((const uint8_t *)s.out->data, (const uint8_t *)s.out->data + ggml_nbytes(s.out));
    }
    {
        mm_graph g(tensors, 0, "qkv", n_state, n_ctx);
        mm_graph::set(g.in, X);
        g.run(threads);
        ggml_tensor * q = g.node.at("q_add"), * k = g.node.at("k_cpy"), * v = g.node.at("v_cpy");
        Q.assign((const uint8_t *)q->data, (const uint8_t *)q->data + ggml_nbytes(q));
        std::memcpy(K.data(), k->data, ggml_nbytes(k));
        std::memcpy(V.data(), v->data, ggml_nbytes(v));
    }
    fa_graph g(n_state, n_ctx, 1536);
    g.set(Q, K, V);
    g.run(threads);
    const size_t op_bytes = ggml_nbytes(g.out) + K.size() + V.size() + g.work.size();
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::printf("bench-attn-reference threads %d wall_best_ms %.4f cpu_ms_per_call %.4f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld\n",
                threads, best, (c1 - c0) * 1000.0 / reps, reps, (op_bytes + 1023) / 1024, peak >= 0 ? peak - before : -1);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// --bench-encode <model.bin> <wav> <threads>: the WHOLE encoder as whisper runs it, whisper_encode_with_state (the conv
// graph, then the encoder graph: mel -> embd_enc), on the state after whisper_pcm_to_mel_with_state, in a fresh process.
// wall = best of 10, cpu = CPU ms per call over >= 1 s; mem = the two schedulers' compute buffers + kv_pad (the bytes
// the state holds for the encoder besides the model; allocated by whisper_init_state); rss = VmHWM delta of the first call.
static int bench_encode(const char * model_path, const char * wav, int threads) {
    whisper_context * ctx = load_quiet(model_path);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    const int n_state = whisper_model_n_audio_state(ctx);
    size_t bytes = (size_t)2 * 1536 * n_state * 2;   // kv_pad.k and .v (f16)
    for (size_t off : {(size_t)VOAICE_OFF_STATE_SCHED_CONV, (size_t)VOAICE_OFF_STATE_SCHED_ENCODE}) {
        ggml_backend_sched_t s = state_sched(st, off);
        for (int i = 0; i < ggml_backend_sched_get_n_backends(s); i++) bytes += ggml_backend_sched_get_buffer_size(s, ggml_backend_sched_get_backend(s, i));
    }
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    check(whisper_encode_with_state(ctx, st, 0, threads) == 0, "whisper_encode failed");
    const long peak = reset ? status_kb("VmHWM:") : -1;
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        const double t = now_ms();
        check(whisper_encode_with_state(ctx, st, 0, threads) == 0, "whisper_encode failed");
        best = std::min(best, now_ms() - t);
    }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 5 || now_ms() - w0 < 2000.0) { check(whisper_encode_with_state(ctx, st, 0, threads) == 0, "whisper_encode failed"); reps++; }
    const double c1 = cpu_seconds();
    const std::vector<uint8_t> e = tensor_bytes(state_embd_enc(st, n_state));
    // (0.1.1) whisper_encode_with_state also ran the cross graph: kv_cross's digest (k then v, whole), comparable with
    // `voaice bench-cross ... whole`; op_mem_kb above does not count sched_cross's buffer or kv_cross (as v0.1.0 printed it)
    std::vector<uint8_t> kv;
    for (size_t off : {(size_t)VOAICE_OFF_KV_K, (size_t)VOAICE_OFF_KV_V}) {
        const std::vector<uint8_t> b = tensor_bytes(*reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_KV_CROSS + off));
        kv.insert(kv.end(), b.begin(), b.end());
    }
    size_t cross_bytes = kv.size();   // kv_cross + sched_cross's compute buffer: what the call also uses for the cross graph
    {
        ggml_backend_sched_t s = state_sched(st, VOAICE_OFF_STATE_SCHED_CROSS);
        for (int i = 0; i < ggml_backend_sched_get_n_backends(s); i++) cross_bytes += ggml_backend_sched_get_buffer_size(s, ggml_backend_sched_get_backend(s, i));
    }
    std::printf("bench-encode-reference threads %d wall_best_ms %.4f cpu_ms_per_call %.4f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld embd_enc_digest %016llx kv_cross_digest %016llx cross_mem_kb %zu\n",
                threads, best, (c1 - c0) * 1000.0 / reps, reps, (bytes + 1023) / 1024, peak >= 0 ? peak - before : -1,
                (unsigned long long)digest32((const float *)e.data(), e.size() / 4),
                (unsigned long long)digest16((const uint16_t *)kv.data(), kv.size() / 2), (cross_bytes + 1023) / 1024);
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// ---- (0.1.1) the cross graph: cross-attention K and V of every decoder layer, into kv_cross --------------------------
// --cross <model.bin> <outdir> <wav ...>: each input encoded five times (whisper_encode_with_state runs the conv graph,
// the encoder graph and then the CROSS graph on sched_cross) — the cross graph observed through its eval callback at 1
// and 4 threads, not observed at 1, 2, 4 — and after each run the state's kv_cross.k / .v read whole (the layout
// probe's offsets, self-checked against the CPY nodes' destinations). Writes per input:
//   cross_nodes.tsv   idx, key (b<il>.{k_mm,k_scale,k_cpy,v_mm,v_add,v_cpy}), op, type, row width, rows, offset in cross.d64
//   cross.d64         one 64-bit FNV-1a digest per row (frame) of every computed node, 1-thread run
//   kv_cross.d64      one digest per row (n_state f16) of kv_cross.k then of kv_cross.v, ALL rows: n_layer x n_pad each,
//                     the padding rows included
//   cross.tsv         the self-checks and the facts: Kscale's bits, the cache's shape, embd_enc's digest (the input,
//                     equal to the encoder record's embd_enc.f32), 1 vs 4 threads, observed vs not, padding +0, the
//                     standalone graph (what --bench-cross times) = the scheduler's kv_cross at 1 and 4 threads
static const char * CROSS_KEYS[6] = {"k_mm", "k_scale", "k_cpy", "v_mm", "v_add", "v_cpy"};
struct cross_capture {
    std::map<const ggml_tensor *, std::string> names, key;
    std::vector<std::string> tsv, other;
    std::vector<uint64_t> digests;
    std::map<std::string, std::vector<uint64_t>> by_key;
    const ggml_tensor * embd_enc = nullptr, * kc = nullptr, * vc = nullptr;
    int n_state = 0, n_pad = 0, idx = 0, scale_inplace = 0;
    uint32_t scale_bits = 0;
    bool weights_f16_plain = true, src1_embd_enc = true, biases_named = true, k_unbiased = true, scale_b_zero = true,
         scale_same = true, cpy_into_cache = true;
};
static bool cross_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<cross_capture *>(ud);
    if (ask) return true;
    const int idx = c.idx++;
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE)
        return true;
    auto src_key = [&](int i) -> std::string {
        auto it = t->src[i] ? c.key.find(t->src[i]) : c.key.end();
        return it == c.key.end() ? "" : it->second;
    };
    auto split = [](const std::string & s, std::string & il) { il = s.substr(0, s.find('.')); return s.empty() ? s : s.substr(s.find('.') + 1); };
    std::string k, il;
    if (t->op == GGML_OP_MUL_MAT) {
        auto it = c.names.find(t->src[0]);
        const std::string p = "decoder.blocks.", w = it == c.names.end() ? "" : it->second;
        if (w.compare(0, p.size(), p) == 0) {
            il = std::to_string(std::atoi(w.c_str() + p.size()));
            const std::string rest = w.substr(w.find('.', p.size()) + 1);
            if (rest == "cross_attn.key.weight") k = il + ".k_mm";
            if (rest == "cross_attn.value.weight") k = il + ".v_mm";
        }
        c.weights_f16_plain = c.weights_f16_plain && t->src[0]->type == GGML_TYPE_F16 && t->src[0]->buffer &&
                              std::strcmp(ggml_backend_buffer_name(t->src[0]->buffer), "CPU") == 0;
        const ggml_tensor * s1 = t->src[1];
        c.src1_embd_enc = c.src1_embd_enc && s1->type == GGML_TYPE_F32 && ggml_is_contiguous(s1) &&
                          (s1 == c.embd_enc || s1->view_src == c.embd_enc) && s1->data == c.embd_enc->data;
    } else if (t->op == GGML_OP_SCALE) {
        const std::string sk = split(src_key(0), il);
        if (sk == "k_mm") k = il + ".k_scale";
        float sb[2];
        std::memcpy(sb, t->op_params, sizeof sb);
        if (c.scale_bits == 0) c.scale_bits = f32_bits(sb[0]);
        c.scale_same = c.scale_same && f32_bits(sb[0]) == c.scale_bits;
        c.scale_b_zero = c.scale_b_zero && sb[1] == 0.0f;
        c.scale_inplace += t->data == t->src[0]->data;
    } else if (t->op == GGML_OP_ADD) {
        const std::string sk = split(src_key(0), il);
        if (sk == "k_mm" || sk == "k_scale") c.k_unbiased = false;
        if (sk == "v_mm") {
            k = il + ".v_add";
            auto it = c.names.find(t->src[1]);
            c.biases_named = c.biases_named && it != c.names.end() && it->second == "decoder.blocks." + il + ".cross_attn.value.bias";
        }
    } else if (t->op == GGML_OP_CPY) {
        const std::string sk = split(src_key(0), il);
        const ggml_tensor * cache = sk == "k_scale" ? c.kc : sk == "v_add" ? c.vc : nullptr;
        if (cache) {
            k = il + (sk == "k_scale" ? ".k_cpy" : ".v_cpy");
            c.cpy_into_cache = c.cpy_into_cache && t->type == GGML_TYPE_F16 && t->view_src == cache &&
                               t->view_offs == (size_t)2 * c.n_state * std::atoi(il.c_str()) * c.n_pad &&
                               ggml_nelements(t) == ggml_nelements(t->src[0]);
        }
    }
    if (k.empty()) { c.other.push_back(std::string(ggml_op_desc(t)) + "@" + std::to_string(idx)); return true; }
    c.key[t] = k;
    std::vector<uint64_t> d = row_digests(t, c.n_state);
    char line[256];
    std::snprintf(line, sizeof line, "%d\tb%s\t%s\t%s\t%d\t%zu\t%zu", idx, k.c_str(), ggml_op_desc(t), ggml_type_name(t->type),
                  c.n_state, d.size(), c.digests.size());
    c.tsv.push_back(line);
    c.digests.insert(c.digests.end(), d.begin(), d.end());
    c.by_key["b" + k] = d;
    return true;
}
static ggml_tensor * state_kv_cross(whisper_state * st, bool v) {
    return *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_KV_CROSS + (v ? VOAICE_OFF_KV_V : VOAICE_OFF_KV_K));
}
// the cross graph as whisper_build_graph_cross builds it (flash_attn: view_1d at il * n_pad rows), as a standalone graph
// on the shipped CPU backend (what --bench-cross times): in = embd_enc [n_state, n_ctx]; k, v = [n_state * n_layer * n_pad] f16
struct cross_graph {
    ggml_context * ctx; ggml_tensor * in, * k, * v; ggml_cgraph * gf;
    std::vector<uint8_t> work;
    cross_graph(std::map<std::string, ggml_tensor *> & m, int n_state, int n_ctx, int n_pad, int n_layer) {
        ggml_init_params p = { (size_t)96 << 20, nullptr, false };
        ctx = ggml_init(p);
        check(ctx != nullptr, "ggml_init failed");
        in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_state, n_ctx);
        k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_layer * n_pad);
        v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t)n_state * n_layer * n_pad);
        std::memset(k->data, 0, ggml_nbytes(k));
        std::memset(v->data, 0, ggml_nbytes(v));
        gf = ggml_new_graph(ctx);
        const float Kscale = pow(float(n_state / (n_state / 64)), -0.25);   // whisper.cpp:2298, the same expression
        for (int il = 0; il < n_layer; il++) {
            const std::string pre = "decoder.blocks." + std::to_string(il) + ".cross_attn.";
            ggml_tensor * K = ggml_scale(ctx, ggml_mul_mat(ctx, conv2_graph::copy(ctx, m.at(pre + "key.weight")), in), Kscale);
            ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, conv2_graph::copy(ctx, m.at(pre + "value.weight")), in), conv2_graph::copy(ctx, m.at(pre + "value.bias")));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, ggml_view_1d(ctx, k, (int64_t)n_state * n_ctx, (size_t)2 * n_state * il * n_pad)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, ggml_view_1d(ctx, v, (int64_t)n_state * n_ctx, (size_t)2 * n_state * il * n_pad)));
        }
    }
    void run(int threads) {
        ggml_cplan cp = ggml_graph_plan(gf, threads, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
        check(cp.n_threads == threads, "the plan did not take the thread count");
    }
    ~cross_graph() { ggml_free(ctx); }
};
static std::vector<uint64_t> kv_rows(const std::vector<uint8_t> & kb, const std::vector<uint8_t> & vb, int n_state) {
    std::vector<uint64_t> d;
    for (const std::vector<uint8_t> * b : {&kb, &vb})
        for (size_t r = 0; r < b->size() / (2 * (size_t)n_state); r++) d.push_back(digest16((const uint16_t *)(b->data() + r * 2 * n_state), n_state));
    return d;
}
static int record_cross(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_text_state(ctx), n_layer = whisper_model_n_text_layer(ctx);
    const int n_pad = (n_ctx + 255) / 256 * 256;
    check(n_state == whisper_model_n_audio_state(ctx), "n_text_state != n_audio_state");
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        cross_capture caps[2];
        std::vector<uint8_t> kb[5], vb[5], enc[5];
        bool shape = true, zero_at_init = true;
        const int threads[5] = {1, 4, 1, 2, 4};
        for (int r = 0; r < 5; r++) {              // r = 0, 1: observed at 1 and 4 threads; r = 2..4: not observed, 1, 2, 4
            whisper_state * st = whisper_init_state(ctx);
            check(st != nullptr, "whisper_init_state failed");
            ggml_tensor * kc = state_kv_cross(st, false), * vc = state_kv_cross(st, true);
            for (ggml_tensor * x : {kc, vc})
                shape = shape && x && x->type == GGML_TYPE_F16 && ggml_n_dims(x) == 1 && x->ne[0] == (int64_t)n_state * n_layer * n_pad;
            check(shape, "layout check failed: kv_cross is not [n_state * n_layer * n_pad] f16");
            for (ggml_tensor * x : {kc, vc}) { std::vector<uint8_t> b = tensor_bytes(x); for (uint8_t y : b) zero_at_init = zero_at_init && y == 0; }
            check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
            state_sched(st, VOAICE_OFF_STATE_SCHED_CONV);
            state_sched(st, VOAICE_OFF_STATE_SCHED_ENCODE);
            ggml_backend_sched_t sx = state_sched(st, VOAICE_OFF_STATE_SCHED_CROSS);
            if (r < 2) {
                for (auto & kv : tensors) caps[r].names[kv.second] = kv.first;
                caps[r].n_state = n_state; caps[r].n_pad = n_pad;
                caps[r].embd_enc = state_embd_enc(st, n_state); caps[r].kc = kc; caps[r].vc = vc;
                ggml_backend_sched_set_eval_callback(sx, cross_cb, &caps[r]);
            }
            check(whisper_encode_with_state(ctx, st, 0, threads[r]) == 0, "whisper_encode failed");
            kb[r] = tensor_bytes(kc); vb[r] = tensor_bytes(vc); enc[r] = tensor_bytes(state_embd_enc(st, n_state));
            whisper_free_state(st);
        }
        cross_capture & c = caps[0];
        check((int)c.tsv.size() == 6 * n_layer, "the cross graph did not show 6 nodes per decoder layer");
        bool keys = true;
        for (int il = 0; il < n_layer; il++)
            for (const char * k : CROSS_KEYS) keys = keys && c.by_key.count("b" + std::to_string(il) + "." + k);
        const bool thr = c.digests == caps[1].digests && c.tsv == caps[1].tsv && kb[0] == kb[1] && vb[0] == vb[1];
        const bool obs = kb[0] == kb[2] && vb[0] == vb[2];
        const bool thr_unobs = kb[2] == kb[3] && kb[2] == kb[4] && vb[2] == vb[3] && vb[2] == vb[4];
        const bool enc_same = enc[0] == enc[1] && enc[0] == enc[2] && enc[0] == enc[3] && enc[0] == enc[4];
        bool pad_zero = true;
        size_t pad_rows = 0;
        for (const std::vector<uint8_t> * b : {&kb[2], &vb[2]})
            for (int il = 0; il < n_layer; il++)
                for (int row = n_ctx; row < n_pad; row++, pad_rows++)
                    for (int e = 0; e < 2 * n_state; e++) pad_zero = pad_zero && (*b)[((size_t)il * n_pad + row) * 2 * n_state + e] == 0;
        std::vector<uint64_t> kv = kv_rows(kb[2], vb[2], n_state);
        bool cpy_eq_buffer = true;   // each CPY node's rows = the buffer's rows of its layer
        for (int il = 0; il < n_layer; il++)
            for (int which = 0; which < 2; which++) {
                const std::vector<uint64_t> & d = c.by_key.at("b" + std::to_string(il) + (which ? ".v_cpy" : ".k_cpy"));
                for (int row = 0; row < n_ctx; row++) cpy_eq_buffer = cpy_eq_buffer && d[row] == kv[(size_t)which * n_layer * n_pad + (size_t)il * n_pad + row];
            }
        // the standalone graph (what --bench-cross times) from the state's own embd_enc, at 1 and 4 threads
        bool alone = true;
        {
            cross_graph g(tensors, n_state, n_ctx, n_pad, n_layer);
            check(enc[0].size() == ggml_nbytes(g.in), "embd_enc size");
            for (int th : {1, 4}) {
                std::memcpy(g.in->data, enc[0].data(), enc[0].size());
                g.run(th);
                alone = alone && std::memcmp(g.k->data, kb[2].data(), kb[2].size()) == 0 && std::memcmp(g.v->data, vb[2].data(), vb[2].size()) == 0;
            }
        }
        write_bin(dir + "/cross.d64", c.digests.data(), c.digests.size());
        write_bin(dir + "/kv_cross.d64", kv.data(), kv.size());
        FILE * nf = std::fopen((dir + "/cross_nodes.tsv").c_str(), "w");
        for (auto & l : c.tsv) std::fprintf(nf, "%s\n", l.c_str());
        std::fclose(nf);
        std::string other;
        for (auto & o : c.other) other += (other.empty() ? "" : ",") + o;
        const double kscale_d = pow(float(n_state / (n_state / 64)), -0.25);
        const auto yn = [](bool b) { return b ? "yes" : "NO"; };
        FILE * m = std::fopen((dir + "/cross.tsv").c_str(), "w");
        std::fprintf(m, "n_ctx\t%d\nn_pad\t%d\nn_state\t%d\nn_layer\t%d\nnodes_digested\t%zu\nflash_attn_layout\tyes\n"
                        "kscale_bits\t%08x\nkscale_pow_double\t%.17g\nscale_b_zero\t%s\nscale_same_every_layer\t%s\nscale_in_place\t%d\n"
                        "six_nodes_per_layer\t%s\nother_computed_nodes\t%s\nweights_f16_plain_cpu\t%s\nsrc1_is_embd_enc\t%s\n"
                        "v_bias_named\t%s\nk_unbiased\t%s\ncpy_into_kv_cross_at_il_n_pad\t%s\nkv_cross_zero_at_init\t%s\n"
                        "kv_cross_padding_rows\t%zu\nkv_cross_padding_all_zero\t%s\ncpy_nodes_eq_kv_cross_rows\t%s\n"
                        "threads_1_vs_4_bit_identical\t%s\nkv_cross_observed_eq_unobserved\t%s\nkv_cross_1_2_4_threads_identical\t%s\n"
                        "embd_enc_every_run_identical\t%s\nembd_enc_digest\t%016llx\nstandalone_eq_sched_1_4_threads\t%s\n",
                     n_ctx, n_pad, n_state, n_layer, c.tsv.size(), c.scale_bits, kscale_d, yn(c.scale_b_zero), yn(c.scale_same),
                     c.scale_inplace, yn(keys), other.empty() ? "none" : other.c_str(), yn(c.weights_f16_plain), yn(c.src1_embd_enc),
                     yn(c.biases_named), yn(c.k_unbiased), yn(c.cpy_into_cache), yn(zero_at_init), pad_rows, yn(pad_zero), yn(cpy_eq_buffer),
                     yn(thr), yn(obs), yn(thr_unobs), yn(enc_same),
                     (unsigned long long)digest32((const float *)enc[0].data(), enc[0].size() / 4), yn(alone));
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: %zu cross nodes digested (6 per layer: %s; others computed: %s); Kscale %08x (b = 0: %s), "
                             "in place %d of %d; weights f16 plain: %s; src1 = embd_enc: %s; V's bias named: %s; K unbiased: %s; "
                             "CPYs into kv_cross at il*%d: %s; kv_cross +0 at init: %s, %zu padding rows +0 after: %s; CPY nodes = buffer: %s; "
                             "1 vs 4 threads identical: %s; observed = unobserved: %s; 1 = 2 = 4 threads: %s; standalone = scheduler: %s\n",
                     stem.c_str(), c.tsv.size(), yn(keys), other.empty() ? "none" : other.c_str(), c.scale_bits, yn(c.scale_b_zero),
                     c.scale_inplace, n_layer, yn(c.weights_f16_plain), yn(c.src1_embd_enc), yn(c.biases_named), yn(c.k_unbiased), n_pad,
                     yn(c.cpy_into_cache), yn(zero_at_init), pad_rows, yn(pad_zero), yn(cpy_eq_buffer), yn(thr), yn(obs), yn(thr_unobs), yn(alone));
    }
    whisper_free(ctx);
    return 0;
}

// --bench-cross <model.bin> <wav> <threads>: the cross graph through the standalone graph above (the record shows it equal
// to the scheduler's), its input embd_enc computed beforehand from <wav> by whisper_encode_with_state, in a fresh process.
// wall = best of 10, cpu = CPU ms per compute over >= 1 s; mem = what whisper holds for it: sched_cross's compute buffer +
// kv_cross (allocated by whisper_init_state); rss = VmHWM delta of the first compute (the standalone graph's tensors were
// allocated before it). kv_cross_digest = digest16 over k then v, whole (padding included).
static int bench_cross(const char * model_path, const char * wav, int threads) {
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    whisper_state * st = whisper_init_state(ctx);
    std::vector<float> pcm = read_wav(wav);
    check(whisper_pcm_to_mel_with_state(ctx, st, pcm.data(), (int)pcm.size(), 1) == 0, "pcm_to_mel failed");
    check(whisper_encode_with_state(ctx, st, 0, threads) == 0, "whisper_encode failed");
    const int n_ctx = whisper_model_n_audio_ctx(ctx), n_state = whisper_model_n_text_state(ctx), n_layer = whisper_model_n_text_layer(ctx);
    const int n_pad = (n_ctx + 255) / 256 * 256;
    const std::vector<uint8_t> e = tensor_bytes(state_embd_enc(st, n_state));
    const std::vector<uint8_t> want_k = tensor_bytes(state_kv_cross(st, false)), want_v = tensor_bytes(state_kv_cross(st, true));
    size_t bytes = want_k.size() + want_v.size();
    ggml_backend_sched_t s = state_sched(st, VOAICE_OFF_STATE_SCHED_CROSS);
    for (int i = 0; i < ggml_backend_sched_get_n_backends(s); i++) bytes += ggml_backend_sched_get_buffer_size(s, ggml_backend_sched_get_backend(s, i));
    cross_graph g(tensors, n_state, n_ctx, n_pad, n_layer);
    std::memcpy(g.in->data, e.data(), e.size());
    const long before = status_kb("VmRSS:");
    const bool reset = reset_peak_rss();
    g.run(threads);
    const long peak = reset ? status_kb("VmHWM:") : -1;
    check(std::memcmp(g.k->data, want_k.data(), want_k.size()) == 0 && std::memcmp(g.v->data, want_v.data(), want_v.size()) == 0,
          "the standalone cross graph != the state's kv_cross");
    double best = 1e30;
    for (int r = 0; r < 10; r++) { const double t = now_ms(); g.run(threads); best = std::min(best, now_ms() - t); }
    const double c0 = cpu_seconds(), w0 = now_ms();
    int reps = 0;
    while (reps < 10 || now_ms() - w0 < 1000.0) { g.run(threads); reps++; }
    const double c1 = cpu_seconds();
    std::vector<uint16_t> kv(want_k.size() / 2 + want_v.size() / 2);
    std::memcpy(kv.data(), want_k.data(), want_k.size());
    std::memcpy((uint8_t *)kv.data() + want_k.size(), want_v.data(), want_v.size());
    std::printf("bench-cross-reference threads %d wall_best_ms %.4f cpu_ms_per_call %.4f cpu_reps %d op_mem_kb %zu rss_peak_delta_kb %ld kv_cross_digest %016llx\n",
                threads, best, (c1 - c0) * 1000.0 / reps, reps, (bytes + 1023) / 1024, peak >= 0 ? peak - before : -1,
                (unsigned long long)digest16(kv.data(), kv.size()));
    whisper_free_state(st);
    whisper_free(ctx);
    return 0;
}

// --decin <model.bin> <outdir> <wav ...> (0.1.2): the decoder's input. whisper_full_with_state on each input with the
// params of the 0.0.1 transcript record (config A: greedy, temperature_inc 0, language en, token timestamps) and with a
// 300-token prompt and no timestamps (config B, see decin_params), sched_decode observed through its eval callback at 1
// and 4 threads (whisper_full's n_threads), then run again unobserved (A at 1 and 4, B at 1).
// For EVERY decoder call (the prompt of each window and every one-token step) the callback sees the graph's first two
// GET_ROWS (d_te by `embd`, d_pe by `position`) and their ADD, and at the first of them reads the state's whisper_batch
// (the layout probe's offset, self-checked against the graph's own input tensors). Writes per input:
//   decin_calls.tsv   per call: config (A, B), threads, call index, n_tokens, the batch's token / pos / seq_id[i][0] / n_seq_id / logits
//                     (comma lists), the offset of its rows in decin.d64
//   decin.d64         per call, one 64-bit FNV-1a digest per row: the token rows (GET_ROWS d_te), the position rows
//                     (GET_ROWS d_pe), the sum (ADD) — n_tokens each, in that order
//   decin_result.tsv  the observed 1-thread run's result tokens (segment, index, id, p bits): the record's transcript
//   decin.tsv         the self-checks and the facts
struct batch_mirror { int32_t n_tokens; int32_t * token; int32_t * pos; int32_t * n_seq_id; int32_t ** seq_id; int8_t * logits; };
static_assert(sizeof(batch_mirror) == VOAICE_SIZEOF_BATCH, "whisper_batch layout changed");
struct decin_call {
    int n = 0;
    std::vector<int32_t> tok, pos, seq, nseq;
    std::vector<int8_t> logits;
    std::vector<uint64_t> te, pe, add;
    int seen = 0;   // 1 = te, 2 = te + pe, 3 = all three, in that order
};
struct decin_capture {
    whisper_state * st = nullptr;
    const ggml_tensor * d_te = nullptr, * d_pe = nullptr;
    int n_state = 0;
    std::vector<decin_call> calls;
    bool batch_eq_inputs = true, types = true, order = true, add_src0_te = true, other_get_rows = false;
    int add_inplace = 0;
};
static bool is_add_te_pe(const ggml_tensor * t, const decin_capture & c) {
    if (t->op != GGML_OP_ADD || !t->src[0] || !t->src[1]) return false;
    const ggml_tensor * a = t->src[0], * b = t->src[1];
    if (a->op != GGML_OP_GET_ROWS || b->op != GGML_OP_GET_ROWS) return false;
    return (a->src[0] == c.d_te && b->src[0] == c.d_pe) || (a->src[0] == c.d_pe && b->src[0] == c.d_te);
}
static bool decin_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<decin_capture *>(ud);
    const bool te = t->op == GGML_OP_GET_ROWS && t->src[0] == c.d_te, pe = t->op == GGML_OP_GET_ROWS && t->src[0] == c.d_pe;
    const bool add = is_add_te_pe(t, c);
    if (ask) {
        if (t->op == GGML_OP_GET_ROWS && !te && !pe) c.other_get_rows = true;
        return te || pe || add;
    }
    const std::vector<uint64_t> d = row_digests(t, c.n_state);
    c.types = c.types && t->type == GGML_TYPE_F32 && t->ne[0] == c.n_state;
    if (te) {
        decin_call k;
        const auto & b = *reinterpret_cast<const batch_mirror *>((const char *)c.st + VOAICE_OFF_STATE_BATCH);
        k.n = b.n_tokens;
        const std::vector<uint8_t> in = tensor_bytes(t->src[1]);
        c.batch_eq_inputs = c.batch_eq_inputs && t->src[1]->type == GGML_TYPE_I32 && (int64_t)k.n == ggml_nelements(t->src[1]) &&
                            (int64_t)k.n == t->ne[1] && in.size() == 4 * (size_t)k.n;
        for (int i = 0; i < k.n; i++) {
            k.tok.push_back(b.token[i]); k.pos.push_back(b.pos[i]); k.nseq.push_back(b.n_seq_id[i]);
            k.seq.push_back(b.seq_id[i][0]); k.logits.push_back(b.logits[i]);
            int32_t v; std::memcpy(&v, in.data() + 4 * i, 4);
            c.batch_eq_inputs = c.batch_eq_inputs && v == b.token[i];
        }
        c.types = c.types && c.d_te->type == GGML_TYPE_F16;
        k.te = d; k.seen = 1;
        c.calls.push_back(k);
        return true;
    }
    check(!c.calls.empty(), "decin: a position row or a sum before any token row");
    decin_call & k = c.calls.back();
    if (pe) {
        const std::vector<uint8_t> in = tensor_bytes(t->src[1]);
        c.batch_eq_inputs = c.batch_eq_inputs && t->src[1]->type == GGML_TYPE_I32 && in.size() == 4 * (size_t)k.n;
        for (int i = 0; i < k.n && c.batch_eq_inputs; i++) { int32_t v; std::memcpy(&v, in.data() + 4 * i, 4); c.batch_eq_inputs = v == k.pos[i]; }
        c.types = c.types && c.d_pe->type == GGML_TYPE_F32;
        c.order = c.order && k.seen == 1;
        k.pe = d; k.seen = 2;
    } else {
        c.order = c.order && k.seen == 2;
        c.add_src0_te = c.add_src0_te && t->src[0]->src[0] == c.d_te;
        c.add_inplace += t->data == t->src[0]->data || t->data == t->src[1]->data;
        k.add = d; k.seen = 3;
    }
    return true;
}
static std::string csv(const std::vector<int32_t> & v) { std::string s; for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]); return s; }
// config A = the 0.0.1 transcript record's params (see main); config B = the same with no_timestamps and DECIN_PROMPT as
// prompt_tokens (not carried), so each window's first prompt is [PREV, the last 223 of them, SOT, NOT]: 226 tokens, the
// many-row batch and the positions past 225 that config A (whose prompts are all [SOT]) never reaches
static std::vector<whisper_token> decin_prompt() {
    static const whisper_token words[23] = {843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, 460, 466, 329, 345, 1265,
                                            644, 345, 460, 466, 329, 534, 1499, 13};   // jfk's recorded text tokens
    std::vector<whisper_token> v(300);
    for (size_t i = 0; i < v.size(); i++) v[i] = words[i % 23];
    return v;
}
static whisper_full_params decin_params(int threads, bool prompted, const std::vector<whisper_token> & prompt) {
    whisper_full_params fp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    fp.n_threads = threads;
    fp.print_progress = false; fp.print_realtime = false; fp.print_timestamps = false; fp.print_special = false;
    fp.token_timestamps = true;
    fp.temperature_inc = 0.0f;
    fp.language = "en";
    if (prompted) { fp.no_timestamps = true; fp.prompt_tokens = prompt.data(); fp.prompt_n_tokens = (int)prompt.size(); }
    return fp;
}
static std::vector<std::string> result_lines(whisper_context * ctx, whisper_state * st) {
    (void)ctx;
    std::vector<std::string> r;
    for (int s = 0; s < whisper_full_n_segments_from_state(st); s++)
        for (int k = 0; k < whisper_full_n_tokens_from_state(st, s); k++) {
            whisper_token_data d = whisper_full_get_token_data_from_state(st, s, k);
            char line[96];
            std::snprintf(line, sizeof line, "token\t%d\t%d\t%d\t%08x", s, k, d.id, f32_bits(d.p));
            r.push_back(line);
        }
    return r;
}
static int record_decin(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_state = whisper_model_n_text_state(ctx);
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        const std::vector<whisper_token> prompt = decin_prompt();
        decin_capture caps[2][2];
        std::vector<std::string> res[2][4];
        const int threads[4] = {1, 4, 1, 4};
        for (int cfg = 0; cfg < 2; cfg++)
            for (int r = 0; r < (cfg ? 3 : 4); r++) {   // r = 0, 1: observed at 1 and 4 threads; r = 2, 3: not observed
                whisper_state * st = whisper_init_state(ctx);
                check(st != nullptr, "whisper_init_state failed");
                ggml_backend_sched_t sd = state_sched(st, VOAICE_OFF_STATE_SCHED_DECODE);
                if (r < 2) {
                    decin_capture & c = caps[cfg][r];
                    c.st = st; c.n_state = n_state;
                    c.d_te = tensors.at("decoder.token_embedding.weight");
                    c.d_pe = tensors.at("decoder.positional_embedding");
                    ggml_backend_sched_set_eval_callback(sd, decin_cb, &c);
                }
                check(whisper_full_with_state(ctx, st, decin_params(threads[r], cfg == 1, prompt), pcm.data(), (int)pcm.size()) == 0, "whisper_full failed");
                res[cfg][r] = result_lines(ctx, st);
                whisper_free_state(st);
            }
        std::vector<uint64_t> d64;
        FILE * cf = std::fopen((dir + "/decin_calls.tsv").c_str(), "w");
        int prompts[2] = {0, 0}, steps[2] = {0, 0}, rows = 0, max_n = 0, max_pos = 0;
        bool complete = true, thr[2], obs1[2], obs4[2], res14[2], all = true;
        int first_diff[2] = {-1, -1};
        for (int cfg = 0; cfg < 2; cfg++)
            for (int r = 0; r < 2; r++) {
                const decin_capture & c = caps[cfg][r];
                all = all && c.batch_eq_inputs && c.types && c.order && c.add_src0_te && !c.other_get_rows;
                for (size_t k = 0; k < c.calls.size(); k++) {
                    const decin_call & x = c.calls[k];
                    complete = complete && x.seen == 3 && (int)x.te.size() == x.n && (int)x.pe.size() == x.n && (int)x.add.size() == x.n;
                    if (r == 0) {
                        (x.pos[0] == 0 ? prompts[cfg] : steps[cfg])++;
                        rows += x.n; max_n = std::max(max_n, x.n); max_pos = std::max(max_pos, x.pos[x.n - 1]);
                    }
                    std::vector<int32_t> lg(x.logits.begin(), x.logits.end());
                    std::fprintf(cf, "call\t%c\t%d\t%zu\t%d\t%s\t%s\t%s\t%s\t%s\t%zu\n", "AB"[cfg], threads[r], k, x.n, csv(x.tok).c_str(),
                                 csv(x.pos).c_str(), csv(x.seq).c_str(), csv(x.nseq).c_str(), csv(lg).c_str(), d64.size());
                    for (const auto * v : {&x.te, &x.pe, &x.add}) d64.insert(d64.end(), v->begin(), v->end());
                }
            }
        std::fclose(cf);
        write_bin(dir + "/decin.d64", d64.data(), d64.size());
        FILE * rf = std::fopen((dir + "/decin_result.tsv").c_str(), "w");
        for (auto & l : res[0][0]) std::fprintf(rf, "%s\n", l.c_str());
        std::fclose(rf);
        for (int cfg = 0; cfg < 2; cfg++) {
            const auto & x1 = caps[cfg][0].calls, & x4 = caps[cfg][1].calls;
            thr[cfg] = x1.size() == x4.size();
            for (size_t k = 0; k < std::min(x1.size(), x4.size()); k++)
                if (!(x1[k].tok == x4[k].tok && x1[k].pos == x4[k].pos && x1[k].seq == x4[k].seq && x1[k].logits == x4[k].logits &&
                      x1[k].te == x4[k].te && x1[k].pe == x4[k].pe && x1[k].add == x4[k].add)) { thr[cfg] = false; first_diff[cfg] = (int)k; break; }
            obs1[cfg] = res[cfg][0] == res[cfg][2];
            obs4[cfg] = cfg == 1 || res[cfg][1] == res[cfg][3];
            res14[cfg] = res[cfg][0] == res[cfg][1];
        }
        int inplace = 0;
        for (int cfg = 0; cfg < 2; cfg++) inplace += caps[cfg][0].add_inplace;
        const auto yn = [](bool b) { return b ? "yes" : "NO"; };
        FILE * m = std::fopen((dir + "/decin.tsv").c_str(), "w");
        std::fprintf(m, "n_state\t%d\nd_te_type\t%s\nd_pe_type\t%s\nA_calls_1t\t%zu\nA_calls_4t\t%zu\nA_prompts\t%d\nA_steps\t%d\n"
                        "B_calls_1t\t%zu\nB_calls_4t\t%zu\nB_prompts\t%d\nB_steps\t%d\nrows_1t\t%d\nmax_n_tokens\t%d\nmax_pos\t%d\n"
                        "every_call_te_pe_add_observed\t%s\nchecks_batch_eq_inputs_types_order_add_src0_te_no_other_get_rows\t%s\n"
                        "add_in_place_1t\t%d\nA_calls_threads_1_vs_4_identical\t%s\t%d\nB_calls_threads_1_vs_4_identical\t%s\t%d\n"
                        "A_result_observed_eq_unobserved_1t\t%s\nA_result_observed_eq_unobserved_4t\t%s\nA_result_1t_eq_4t\t%s\n"
                        "B_result_observed_eq_unobserved_1t\t%s\nB_result_1t_eq_4t\t%s\nA_result_tokens\t%zu\nB_result_tokens\t%zu\n",
                     n_state, ggml_type_name(caps[0][0].d_te->type), ggml_type_name(caps[0][0].d_pe->type), caps[0][0].calls.size(),
                     caps[0][1].calls.size(), prompts[0], steps[0], caps[1][0].calls.size(), caps[1][1].calls.size(), prompts[1], steps[1],
                     rows, max_n, max_pos, yn(complete), yn(all), inplace, yn(thr[0]), first_diff[0], yn(thr[1]), first_diff[1],
                     yn(obs1[0]), yn(obs4[0]), yn(res14[0]), yn(obs1[1]), yn(res14[1]), res[0][0].size(), res[1][0].size());
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: A %zu calls (%d prompts, %d steps), B %zu calls (%d prompts, %d steps); %d rows, longest batch %d, "
                             "last position %d; all observed: %s; checks: %s; add in place %d; calls 1 vs 4 identical: %s / %s (first differing call %d / %d); "
                             "observed = unobserved: A %s %s, B %s; results 1 = 4 threads: %s / %s\n",
                     stem.c_str(), caps[0][0].calls.size(), prompts[0], steps[0], caps[1][0].calls.size(), prompts[1], steps[1], rows, max_n,
                     max_pos, yn(complete), yn(all), inplace, yn(thr[0]), yn(thr[1]), first_diff[0], first_diff[1], yn(obs1[0]), yn(obs4[0]),
                     yn(obs1[1]), yn(res14[0]), yn(res14[1]));
    }
    whisper_free(ctx);
    return 0;
}

// --bench-decin <model.bin> <threads> <n_tokens>: the decoder's input as a standalone graph on the shipped CPU backend —
// exactly the three nodes whisper builds, add(get_rows(d_te, embd), get_rows(d_pe, position)) — for n_tokens tokens
// (token i = (i * 7919 + 50257) % n_vocab, position i). One call = what whisper_decode_internal does for these nodes: set
// the two I32 inputs, plan, compute (the CPU backend plans every graph it computes). wall = mean over >= 1 s of calls
// (one call is microseconds: a best-of-10 is the timer's resolution), cpu = CPU per call over the same loop.
// out_digest = digest32 over the [n_tokens][n_state] sum.
static int bench_decin(const char * model_path, int threads, int n_tokens) {
    whisper_context * ctx = load_quiet(model_path);
    auto & m = model_tensors(ctx);
    const int n_vocab = whisper_model_n_vocab(ctx);
    ggml_init_params p = { (size_t)64 << 20, nullptr, false };
    ggml_context * g = ggml_init(p);
    check(g != nullptr, "ggml_init failed");
    ggml_tensor * te = conv2_graph::copy(g, m.at("decoder.token_embedding.weight")), * pe = conv2_graph::copy(g, m.at("decoder.positional_embedding"));
    ggml_tensor * embd = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens), * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
    ggml_tensor * out = ggml_add(g, ggml_get_rows(g, te, embd), ggml_get_rows(g, pe, pos));
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, out);
    std::vector<int32_t> tok(n_tokens), ps(n_tokens);
    for (int i = 0; i < n_tokens; i++) { tok[i] = (int32_t)(((int64_t)i * 7919 + 50257) % n_vocab); ps[i] = i; }
    std::vector<uint8_t> work;
    auto call = [&]() {
        std::memcpy(embd->data, tok.data(), 4 * (size_t)n_tokens);
        std::memcpy(pos->data, ps.data(), 4 * (size_t)n_tokens);
        ggml_cplan cp = ggml_graph_plan(gf, threads, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
    };
    call();
    const double c0 = cpu_seconds(), w0 = now_ms();
    long reps = 0;
    while (reps < 1000 || now_ms() - w0 < 1000.0) { call(); reps++; }
    const double w1 = now_ms(), c1 = cpu_seconds();
    std::printf("bench-decin-reference threads %d n_tokens %d wall_us_per_call %.4f cpu_us_per_call %.4f reps %ld out_digest %016llx\n",
                threads, n_tokens, (w1 - w0) * 1000.0 / reps, (c1 - c0) * 1e6 / reps, reps,
                (unsigned long long)digest32((const float *)out->data, (size_t)n_tokens * out->ne[0]));
    ggml_free(g);
    whisper_free(ctx);
    return 0;
}

// --selfkv <model.bin> <outdir> <wav ...> (0.1.3): the self-attention products and the f16 self KV cache. whisper_full on
// each input in 0.1.2's two configs (A: the transcript record's params; B: a 300-token prompt, no timestamps), sched_decode
// observed at 1 and 4 threads (then unobserved at 1 and 4, the results compared). On EVERY decoder call, for every
// decoder layer, the callback observes the nodes whisper_build_graph_decoder builds before self-attention itself:
//   0 norm (NORM, its input = the layer's input read whole), 1 ln_mul (MUL attn_ln.weight), 2 ln_add (ADD attn_ln.bias),
//   3 q_mm, 4 q_add (+ attn.query.bias), 5 q_scale (SCALE), 6 k_mm, 7 k_scale (SCALE; no bias), 8 v_mm, 9 v_add
//   (+ attn.value.bias), 10 k_cpy, 11 v_cpy (the CPYs into kv_self.k / kv_self.v at row il * size + head)
// identified by what they read (the model's tensors, structurally), plus the KQ_mask's cast to f16 (its f32 source read
// too), the kv_self metadata (head, n, size, each cell's pos and seq_id set) and, after the last layer's two CPYs,
// kv_self.k and kv_self.v whole. Writes per input:
//   selfkv_calls.tsv  per call: config, threads, call index, n_tokens, head, n_kv, size, tokens, positions, the cells'
//                     pos (cells 0..n_kv), whether every cell's seq_id set is {0} (used) or empty (unused), the offsets
//                     of its digests in selfkv.d64 and of its layer inputs in selfkv_in.f32
//   selfkv.d64        per call: [n_layer][12 nodes][n_tokens] row digests (digest32 / digest16 over n_state), the mask
//                     rows f32 then f16 (n_tokens each: mask_digest, every element its own word), then kv_self after the
//                     call: [k, v][n_layer][size] row digests (digest16 over n_state)
//   selfkv_in.f32     per call: [n_layer][n_tokens][n_state] — each layer's input (the attn_ln NORM's src0), whole
//   selfkv_result.tsv the observed 1-thread runs' result tokens (A, then B)
//   selfkv.tsv        the self-checks and the facts
struct kv_cell_mirror { int32_t pos; std::set<int32_t> seq_id; };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"   // the same compiler and libstdc++ as the probe's offsetof
static_assert(sizeof(kv_cell_mirror) == VOAICE_SIZEOF_KV_CELL && offsetof(kv_cell_mirror, seq_id) == VOAICE_OFF_CELL_SEQ &&
              offsetof(kv_cell_mirror, pos) == VOAICE_OFF_CELL_POS, "whisper_kv_cell layout changed");
#pragma GCC diagnostic pop
static const char * SELF_KEYS[12] = {"norm", "ln_mul", "ln_add", "q_mm", "q_add", "q_scale", "k_mm", "k_scale", "v_mm", "v_add", "k_cpy", "v_cpy"};
static uint64_t mask_digest(const uint8_t * p, size_t n, size_t es) {   // every element its own word: odd n_kv counts
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { uint64_t w = 0; std::memcpy(&w, p + i * es, es); h = (h ^ w) * 0x100000001b3ULL; }
    return h;
}
struct selfkv_call {
    int n = 0, head = 0, n_kv = 0, size = 0;
    std::vector<int32_t> tok, pos, cell_pos;
    bool seq_ok = true;
    std::vector<uint64_t> nodes;    // [n_layer][12][n]
    std::vector<float> in;          // [n_layer][n][n_state]
    std::vector<uint64_t> m32, m16, kv;
    std::vector<uint8_t> seen;      // [n_layer][12]: times observed
    bool mask_seen = false, done = false;
    int mask_other = 0, mask_ninf = 0, mask_zero = 0;
};
struct selfkv_capture {
    whisper_state * st = nullptr;
    int n_state = 0, n_layer = 0;
    std::map<const ggml_tensor *, std::pair<int, int>> w;   // model tensor -> (layer, kind): 0 ln.w 1 ln.b 2 q.w 3 q.b 4 k.w 5 v.w 6 v.b
    std::vector<selfkv_call> calls;
    const ggml_tensor * pend_norm = nullptr;
    std::vector<uint8_t> pend_in;
    std::vector<uint64_t> pend_d;
    uint32_t scale_bits = 0;
    int scale_inplace = 0, q_add_inplace = 0, cpy_n = 0;
    bool scale_b_zero = true, scale_same = true, cpy_into_cache = true, mm_src1_ln = true, weights_f16 = true, k_unbiased = true,
         mask_shape = true, mask_f16 = true, mask_once = true, kv_f16 = true;
};
static ggml_tensor * state_kv_self(whisper_state * st, bool v) {
    return *reinterpret_cast<ggml_tensor **>((char *)st + VOAICE_OFF_STATE_KV_SELF + (v ? VOAICE_OFF_KV_V : VOAICE_OFF_KV_K));
}
template <class T> static T kv_self_field(whisper_state * st, size_t off) { return *reinterpret_cast<T *>((char *)st + VOAICE_OFF_STATE_KV_SELF + off); }
// what a node is, by what it reads: (layer, key) or (-1, -1); key -2 = a NORM (whose layer its MUL tells), -3 = the mask cast
static std::pair<int, int> selfkv_kind(const ggml_tensor * t, const selfkv_capture & c) {
    auto wk = [&](const ggml_tensor * x) -> std::pair<int, int> { auto it = x ? c.w.find(x) : c.w.end(); return it == c.w.end() ? std::make_pair(-1, -1) : it->second; };
    switch (t->op) {
        case GGML_OP_NORM: return {-1, -2};
        case GGML_OP_MUL: { auto k = wk(t->src[1]); if (k.second == 0) return {k.first, 1}; break; }
        case GGML_OP_ADD: {
            auto k = wk(t->src[1]);
            if (k.second == 1) return {k.first, 2};
            if (k.second == 3) return {k.first, 4};
            if (k.second == 6) return {k.first, 9};
            break;
        }
        case GGML_OP_MUL_MAT: {
            auto k = wk(t->src[0]);
            if (k.second == 2) return {k.first, 3};
            if (k.second == 4) return {k.first, 6};
            if (k.second == 5) return {k.first, 8};
            break;
        }
        case GGML_OP_SCALE: {
            const ggml_tensor * s = t->src[0];
            if (s->op == GGML_OP_ADD && wk(s->src[1]).second == 3) return {wk(s->src[1]).first, 5};
            if (s->op == GGML_OP_MUL_MAT && wk(s->src[0]).second == 4) return {wk(s->src[0]).first, 7};
            break;
        }
        case GGML_OP_CPY: {
            const ggml_tensor * s = t->src[0];
            if (std::strcmp(s->name, "KQ_mask") == 0) return {-1, -3};
            if (s->op == GGML_OP_SCALE && s->src[0]->op == GGML_OP_MUL_MAT && wk(s->src[0]->src[0]).second == 4) return {wk(s->src[0]->src[0]).first, 10};
            if (s->op == GGML_OP_ADD && wk(s->src[1]).second == 6) return {wk(s->src[1]).first, 11};
            break;
        }
        default: break;
    }
    return {-1, -1};
}
// when every node of every layer and the mask are in: the cache after the call (nothing later in the graph writes it)
static void selfkv_finish(selfkv_capture & c, selfkv_call & k) {
    bool all = k.mask_seen;
    for (uint8_t s : k.seen) all = all && s == 1;
    if (!all) return;
    const std::vector<uint8_t> kb = tensor_bytes(state_kv_self(c.st, false)), vb = tensor_bytes(state_kv_self(c.st, true));
    for (const std::vector<uint8_t> * b : {&kb, &vb})
        for (size_t r = 0; r < b->size() / (2 * (size_t)c.n_state); r++) k.kv.push_back(digest16((const uint16_t *)(b->data() + r * 2 * c.n_state), c.n_state));
    k.done = true;
}
static bool selfkv_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & c = *static_cast<selfkv_capture *>(ud);
    const auto kind = selfkv_kind(t, c);
    if (ask) {
        // an ADD reading K (k_mm or its SCALE) would be a bias on K: not asked for, only noted
        if (t->op == GGML_OP_ADD && t->src[0] && ((t->src[0]->op == GGML_OP_MUL_MAT && c.w.count(t->src[0]->src[0]) && c.w.at(t->src[0]->src[0]).second == 4) ||
                                                  (t->src[0]->op == GGML_OP_SCALE && t->src[0]->src[0]->op == GGML_OP_MUL_MAT &&
                                                   c.w.count(t->src[0]->src[0]->src[0]) && c.w.at(t->src[0]->src[0]->src[0]).second == 4)))
            c.k_unbiased = false;
        return kind.second != -1;
    }
    // a call begins at layer 0's attn_ln NORM, the one that reads the decoder's input add(get_rows(d_te), get_rows(d_pe));
    // the NORMs after the last layer's attention (cross_attn_ln, mlp_ln) come after the call is complete and are ignored
    const ggml_tensor * s0 = t->src[0];
    const bool first = t->op == GGML_OP_NORM && s0->op == GGML_OP_ADD && s0->src[0]->op == GGML_OP_GET_ROWS && s0->src[1]->op == GGML_OP_GET_ROWS;
    if (!first && (c.calls.empty() || c.calls.back().done)) {
        check(kind.second == -2, "selfkv: a node of a call outside any call");
        return true;
    }
    if (first) {   // the first node of a call: the batch and the cache's metadata
        check(c.calls.empty() || c.calls.back().done, "selfkv: a call began before the last one was complete");
        selfkv_call k;
        const auto & b = *reinterpret_cast<const batch_mirror *>((const char *)c.st + VOAICE_OFF_STATE_BATCH);
        k.n = b.n_tokens;
        for (int i = 0; i < k.n; i++) { k.tok.push_back(b.token[i]); k.pos.push_back(b.pos[i]); }
        k.head = (int)kv_self_field<uint32_t>(c.st, VOAICE_OFF_KV_HEAD);
        k.n_kv = (int)kv_self_field<uint32_t>(c.st, VOAICE_OFF_KV_N);
        k.size = (int)kv_self_field<uint32_t>(c.st, VOAICE_OFF_KV_SIZE);
        const auto & cells = kv_self_field<std::vector<kv_cell_mirror>>(c.st, VOAICE_OFF_KV_CELLS);
        check((int)cells.size() == k.size, "selfkv: kv_self.cells.size() != kv_self.size");
        for (int i = 0; i < k.size; i++) {
            if (i < k.n_kv) k.cell_pos.push_back(cells[i].pos);
            const bool used = cells[i].pos >= 0;
            k.seq_ok = k.seq_ok && (used ? cells[i].seq_id == std::set<int32_t>{0} : cells[i].seq_id.empty()) && (i < k.n_kv || !used);
        }
        k.nodes.assign((size_t)c.n_layer * 12 * k.n, 0);
        k.in.assign((size_t)c.n_layer * k.n * c.n_state, 0.0f);
        k.seen.assign((size_t)c.n_layer * 12, 0);
        c.calls.push_back(std::move(k));
    }
    selfkv_call & k = c.calls.back();
    if (kind.second == -2) {   // a NORM: its input and digest kept until a MUL by attn_ln.weight reads it
        c.pend_norm = t; c.pend_in = tensor_bytes(t->src[0]); c.pend_d = row_digests(t, c.n_state);
        return true;
    }
    if (kind.second == -3) {   // the mask: f32 source and f16 cast, per row
        const ggml_tensor * m = t->src[0];
        c.mask_shape = c.mask_shape && m->type == GGML_TYPE_F32 && m->ne[0] == k.n_kv && m->ne[1] == k.n && m->ne[2] == 1 && m->ne[3] == 1;
        c.mask_f16 = c.mask_f16 && t->type == GGML_TYPE_F16 && ggml_nelements(t) == ggml_nelements(m);
        c.mask_once = c.mask_once && !k.mask_seen;
        k.mask_seen = true;
        const std::vector<uint8_t> a = tensor_bytes(m), h = tensor_bytes(t);
        for (int r = 0; r < k.n; r++) {
            k.m32.push_back(mask_digest(a.data() + (size_t)r * k.n_kv * 4, k.n_kv, 4));
            k.m16.push_back(mask_digest(h.data() + (size_t)r * k.n_kv * 2, k.n_kv, 2));
            for (int i = 0; i < k.n_kv; i++) {
                uint32_t u; std::memcpy(&u, a.data() + ((size_t)r * k.n_kv + i) * 4, 4);
                if (u == 0) k.mask_zero++; else if (u == 0xFF800000u) k.mask_ninf++; else k.mask_other++;
            }
        }
        selfkv_finish(c, k);
        return true;
    }
    const int il = kind.first, key = kind.second;
    check(il >= 0 && il < c.n_layer && key >= 1 && key < 12, "selfkv: an unclassified node was observed");
    if (key == 1) {
        check(c.pend_norm != nullptr && t->src[0] == c.pend_norm, "selfkv: attn_ln's MUL does not read the last NORM");
        check(c.pend_in.size() == (size_t)k.n * c.n_state * 4 && c.pend_d.size() == (size_t)k.n, "selfkv: the layer input's shape");
        std::memcpy(&k.in[(size_t)il * k.n * c.n_state], c.pend_in.data(), c.pend_in.size());
        std::copy(c.pend_d.begin(), c.pend_d.end(), k.nodes.begin() + ((size_t)il * 12 + 0) * k.n);
        k.seen[il * 12 + 0]++;
        c.pend_norm = nullptr;
    }
    if (key == 3 || key == 6 || key == 8) {
        const ggml_tensor * s1 = t->src[1];
        c.mm_src1_ln = c.mm_src1_ln && s1->op == GGML_OP_ADD && c.w.count(s1->src[1]) && c.w.at(s1->src[1]) == std::make_pair(il, 1);
        c.weights_f16 = c.weights_f16 && t->src[0]->type == GGML_TYPE_F16;
    }
    if (key == 4) c.q_add_inplace += t->data == t->src[0]->data;
    if (key == 5 || key == 7) {
        float sb[2];
        std::memcpy(sb, t->op_params, sizeof sb);
        if (c.scale_bits == 0) c.scale_bits = f32_bits(sb[0]);
        c.scale_same = c.scale_same && f32_bits(sb[0]) == c.scale_bits;
        c.scale_b_zero = c.scale_b_zero && sb[1] == 0.0f;
        c.scale_inplace += t->data == t->src[0]->data;
    }
    if (key == 10 || key == 11) {
        const ggml_tensor * cache = state_kv_self(c.st, key == 11);
        c.kv_f16 = c.kv_f16 && cache->type == GGML_TYPE_F16 && ggml_nelements(cache) == (int64_t)c.n_layer * k.size * c.n_state;
        c.cpy_into_cache = c.cpy_into_cache && t->type == GGML_TYPE_F16 && t->view_src == cache &&
                           t->view_offs == (size_t)2 * c.n_state * ((size_t)il * k.size + k.head) && ggml_nelements(t) == (int64_t)k.n * c.n_state;
    }
    std::vector<uint64_t> d = row_digests(t, c.n_state);
    check(d.size() == (size_t)k.n, "selfkv: a node is not n_tokens rows of n_state");
    std::copy(d.begin(), d.end(), k.nodes.begin() + ((size_t)il * 12 + key) * k.n);
    k.seen[il * 12 + key]++;
    selfkv_finish(c, k);
    return true;
}
static int record_selfkv(const char * model_path, const std::string & outdir, int nwav, char ** wavs) {
    if (mkdir(outdir.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create outdir (its parent must exist)");
    whisper_context * ctx = load_quiet(model_path);
    auto & tensors = model_tensors(ctx);
    const int n_state = whisper_model_n_text_state(ctx), n_layer = whisper_model_n_text_layer(ctx);
    check(n_layer * 12 <= 255, "selfkv: too many layers");
    const char * kinds[7] = {"attn_ln.weight", "attn_ln.bias", "attn.query.weight", "attn.query.bias", "attn.key.weight", "attn.value.weight", "attn.value.bias"};
    std::map<const ggml_tensor *, std::pair<int, int>> wmap;
    bool no_key_bias = true;
    for (int il = 0; il < n_layer; il++) {
        for (int j = 0; j < 7; j++) wmap[tensors.at("decoder.blocks." + std::to_string(il) + "." + kinds[j])] = {il, j};
        no_key_bias = no_key_bias && !tensors.count("decoder.blocks." + std::to_string(il) + ".attn.key.bias");
    }
    const auto yn = [](bool b) { return b ? "yes" : "NO"; };
    for (int a = 0; a < nwav; a++) {
        std::string path = wavs[a];
        std::string stem = path.substr(path.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        const std::string dir = outdir + "/" + stem;
        mkdir(dir.c_str(), 0755);
        std::vector<float> pcm = read_wav(path.c_str());
        const std::vector<whisper_token> prompt = decin_prompt();
        selfkv_capture caps[2][2];
        std::vector<std::string> res[2][4];
        const int threads[4] = {1, 4, 1, 4};
        for (int cfg = 0; cfg < 2; cfg++)
            for (int r = 0; r < 4; r++) {   // r = 0, 1: observed at 1 and 4 threads; r = 2, 3: not observed
                whisper_state * st = whisper_init_state(ctx);
                check(st != nullptr, "whisper_init_state failed");
                ggml_backend_sched_t sd = state_sched(st, VOAICE_OFF_STATE_SCHED_DECODE);
                if (r < 2) {
                    selfkv_capture & c = caps[cfg][r];
                    c.st = st; c.n_state = n_state; c.n_layer = n_layer; c.w = wmap;
                    ggml_backend_sched_set_eval_callback(sd, selfkv_cb, &c);
                }
                check(whisper_full_with_state(ctx, st, decin_params(threads[r], cfg == 1, prompt), pcm.data(), (int)pcm.size()) == 0, "whisper_full failed");
                res[cfg][r] = result_lines(ctx, st);
                if (r < 2) caps[cfg][r].st = nullptr;
                whisper_free_state(st);
            }
        std::vector<uint64_t> d64;
        std::vector<float> fin;
        FILE * cf = std::fopen((dir + "/selfkv_calls.tsv").c_str(), "w");
        int prompts[2] = {0, 0}, steps[2] = {0, 0}, rows = 0, max_n = 0, max_pos = 0, max_kv = 0, size = 0;
        long mzero = 0, mninf = 0, mother = 0;
        bool complete = true, seq_ok = true, head_rule = true;
        for (int cfg = 0; cfg < 2; cfg++)
            for (int r = 0; r < 2; r++) {
                const selfkv_capture & c = caps[cfg][r];
                for (size_t k = 0; k < c.calls.size(); k++) {
                    const selfkv_call & x = c.calls[k];
                    complete = complete && x.done && x.kv.size() == (size_t)2 * n_layer * x.size && x.m32.size() == (size_t)x.n && x.m16.size() == (size_t)x.n;
                    seq_ok = seq_ok && x.seq_ok;
                    // greedy, one sequence: the prompt at cell 0, every step at the cell after the last
                    head_rule = head_rule && x.head == (x.pos[0] == 0 ? 0 : x.pos[0]) && x.n_kv == x.head + x.n;
                    size = x.size;
                    if (r == 0) {
                        (x.pos[0] == 0 ? prompts[cfg] : steps[cfg])++;
                        rows += x.n; max_n = std::max(max_n, x.n); max_pos = std::max(max_pos, x.pos[x.n - 1]); max_kv = std::max(max_kv, x.n_kv);
                        mzero += x.mask_zero; mninf += x.mask_ninf; mother += x.mask_other;
                    }
                    std::fprintf(cf, "call\t%c\t%d\t%zu\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\t%zu\t%zu\n", "AB"[cfg], threads[r], k, x.n, x.head, x.n_kv, x.size,
                                 csv(x.tok).c_str(), csv(x.pos).c_str(), csv(x.cell_pos).c_str(), (int)x.seq_ok, d64.size(), fin.size());
                    d64.insert(d64.end(), x.nodes.begin(), x.nodes.end());
                    d64.insert(d64.end(), x.m32.begin(), x.m32.end());
                    d64.insert(d64.end(), x.m16.begin(), x.m16.end());
                    d64.insert(d64.end(), x.kv.begin(), x.kv.end());
                    fin.insert(fin.end(), x.in.begin(), x.in.end());
                }
            }
        std::fclose(cf);
        write_bin(dir + "/selfkv.d64", d64.data(), d64.size());
        write_bin(dir + "/selfkv_in.f32", fin.data(), fin.size());
        FILE * rf = std::fopen((dir + "/selfkv_result.tsv").c_str(), "w");
        for (int cfg = 0; cfg < 2; cfg++) for (auto & l : res[cfg][0]) std::fprintf(rf, "%c\t%s\n", "AB"[cfg], l.c_str());
        std::fclose(rf);
        // 1 vs 4 threads, call by call while the two runs fed the same batch: per layer, the calls whose input was the
        // same at both counts, and per node the calls where that same input gave different rows
        FILE * m = std::fopen((dir + "/selfkv.tsv").c_str(), "w");
        bool checks = true;
        for (int cfg = 0; cfg < 2; cfg++)
            for (int r = 0; r < 2; r++) {
                const selfkv_capture & c = caps[cfg][r];
                checks = checks && c.scale_b_zero && c.scale_same && c.cpy_into_cache && c.mm_src1_ln && c.weights_f16 && c.k_unbiased &&
                         c.mask_shape && c.mask_f16 && c.mask_once && c.kv_f16;
            }
        std::fprintf(m, "n_state\t%d\nn_layer\t%d\nkv_self_size\t%d\nmodel_has_no_key_bias\t%s\nA_calls_1t\t%zu\nA_calls_4t\t%zu\nA_prompts\t%d\nA_steps\t%d\n"
                        "B_calls_1t\t%zu\nB_calls_4t\t%zu\nB_prompts\t%d\nB_steps\t%d\nrows_1t\t%d\nmax_n_tokens\t%d\nmax_pos\t%d\nmax_n_kv\t%d\n"
                        "every_call_every_node_mask_and_cache\t%s\nchecks_scale_b0_same_cpy_at_il_size_head_mm_src1_attn_ln_f16_weights_k_unbiased_mask_f32_to_f16_once\t%s\n"
                        "cells_seq_id_0_used_empty_unused_none_past_n_kv\t%s\nhead_at_pos_n_kv_eq_head_plus_n\t%s\nkqscale_bits\t%08x\n"
                        "scale_in_place_1t\t%d\nq_add_in_place_1t\t%d\nmask_values_zero_ninf_other_1t\t%ld\t%ld\t%ld\n",
                     n_state, n_layer, size, yn(no_key_bias), caps[0][0].calls.size(), caps[0][1].calls.size(), prompts[0], steps[0],
                     caps[1][0].calls.size(), caps[1][1].calls.size(), prompts[1], steps[1], rows, max_n, max_pos, max_kv, yn(complete), yn(checks),
                     yn(seq_ok), yn(head_rule), caps[0][0].scale_bits, caps[0][0].scale_inplace + caps[1][0].scale_inplace,
                     caps[0][0].q_add_inplace + caps[1][0].q_add_inplace, mzero, mninf, mother);
        std::string keys;
        for (const char * k : SELF_KEYS) keys += (keys.empty() ? "" : ",") + std::string(k);
        std::fprintf(m, "node_keys\t%s\n", keys.c_str());
        std::string thr_line;
        for (int cfg = 0; cfg < 2; cfg++) {
            const auto & x1 = caps[cfg][0].calls, & x4 = caps[cfg][1].calls;
            size_t same_batch = 0, kv_differ = 0, mask_differ = 0;
            std::vector<size_t> in_same(n_layer, 0), in_differ(n_layer, 0);
            std::vector<std::vector<size_t>> out_differ(n_layer, std::vector<size_t>(12, 0));
            for (size_t k = 0; k < std::min(x1.size(), x4.size()); k++) {
                if (x1[k].tok != x4[k].tok || x1[k].pos != x4[k].pos) break;
                same_batch++;
                mask_differ += x1[k].m32 != x4[k].m32 || x1[k].m16 != x4[k].m16;
                const size_t n = x1[k].n;
                bool all_in_same = true;
                for (int il = 0; il < n_layer; il++) {
                    const float * a = &x1[k].in[(size_t)il * n * n_state], * b = &x4[k].in[(size_t)il * n * n_state];
                    const bool same = std::memcmp(a, b, n * n_state * 4) == 0;
                    all_in_same = all_in_same && same;
                    (same ? in_same : in_differ)[il]++;
                    if (!same) continue;
                    for (int key = 0; key < 12; key++)
                        out_differ[il][key] += !std::equal(x1[k].nodes.begin() + (il * 12 + key) * n, x1[k].nodes.begin() + (il * 12 + key + 1) * n,
                                                           x4[k].nodes.begin() + (il * 12 + key) * n);
                }
                if (all_in_same) kv_differ += x1[k].kv != x4[k].kv;
            }
            std::fprintf(m, "%c_threads_1_vs_4_calls_same_batch\t%zu\n%c_threads_1_vs_4_mask_differ\t%zu\n%c_threads_1_vs_4_kv_differ_all_inputs_same\t%zu\n",
                         "AB"[cfg], same_batch, "AB"[cfg], mask_differ, "AB"[cfg], kv_differ);
            thr_line += std::string(cfg ? "; B" : "A") + " " + std::to_string(same_batch) + " calls same batch:";
            for (int il = 0; il < n_layer; il++) {
                std::string od;
                size_t tot = 0;
                for (int key = 0; key < 12; key++) { od += (key ? "," : "") + std::to_string(out_differ[il][key]); tot += out_differ[il][key]; }
                std::fprintf(m, "%c_threads_1_vs_4_layer%d_input_same_differ_nodes_differ\t%zu\t%zu\t%s\n", "AB"[cfg], il, in_same[il], in_differ[il], od.c_str());
                thr_line += " L" + std::to_string(il) + " in same " + std::to_string(in_same[il]) + "/differ " + std::to_string(in_differ[il]) + ", nodes differ " + std::to_string(tot);
            }
        }
        std::fprintf(m, "A_result_observed_eq_unobserved_1t\t%s\nA_result_observed_eq_unobserved_4t\t%s\nB_result_observed_eq_unobserved_1t\t%s\n"
                        "B_result_observed_eq_unobserved_4t\t%s\nA_result_1t_eq_4t\t%s\nB_result_1t_eq_4t\t%s\nA_result_tokens\t%zu\nB_result_tokens\t%zu\n"
                        "d64_words\t%zu\nin_f32_values\t%zu\n",
                     yn(res[0][0] == res[0][2]), yn(res[0][1] == res[0][3]), yn(res[1][0] == res[1][2]), yn(res[1][1] == res[1][3]),
                     yn(res[0][2] == res[0][3]), yn(res[1][2] == res[1][3]), res[0][0].size(), res[1][0].size(), d64.size(), fin.size());
        std::fclose(m);
        std::fprintf(stderr, "whisper_oracle: %s: A %zu calls (%d prompts, %d steps), B %zu (%d, %d); %d rows, longest batch %d, last position %d, "
                             "largest n_kv %d of %d cells; complete: %s; checks: %s; cells: %s; head/n rule: %s; KQscale %08x; mask 0/-inf/other %ld/%ld/%ld; "
                             "observed = unobserved: A %s %s, B %s %s; results 1 = 4 threads: A %s, B %s; 1 vs 4 threads: %s\n",
                     stem.c_str(), caps[0][0].calls.size(), prompts[0], steps[0], caps[1][0].calls.size(), prompts[1], steps[1], rows, max_n, max_pos,
                     max_kv, size, yn(complete), yn(checks), yn(seq_ok), yn(head_rule), caps[0][0].scale_bits, mzero, mninf, mother,
                     yn(res[0][0] == res[0][2]), yn(res[0][1] == res[0][3]), yn(res[1][0] == res[1][2]), yn(res[1][1] == res[1][3]),
                     yn(res[0][2] == res[0][3]), yn(res[1][2] == res[1][3]), thr_line.c_str());
    }
    whisper_free(ctx);
    return 0;
}

// --bench-selfkv <model.bin> <threads> <n_tokens> <block|call> (0.1.3): the nodes whisper_build_graph_decoder builds before
// self-attention, as a standalone graph on the shipped CPU backend, for n_tokens rows: block = decoder layer 0's twelve
// (norm, · w, + b; Q + b, × KQscale; K, × KQscale; V + b; the two CPYs into an f16 cache of [n_layer][512][n_state] at
// row head) — call = every layer's twelve (each reading the same input: in whisper layer il > 0 reads the previous
// layer's output, which is not this increment's) and the KQ_mask's cast to f16. head = 0 for a prompt (n > 1), 226 for
// a step (n = 1: config B's first step); n_kv = head + n_tokens. The input x[i] = ((i·7919) mod 2001 − 1000) / 256
// (exact in f32), the mask the causal one for positions head..head+n−1 over cells 0..n_kv−1. One call = set the inputs
// (x, and for call the f32 mask, which whisper builds outside the graph), plan, compute; wall = the mean over >= 1 s of
// calls, cpu = CPU per call over the same loop. out_digest: digest32 over the last layer's Q (scaled), then digest16
// over the k and v cache rows written, then (call) the f16 mask.
static int bench_selfkv(const char * model_path, int threads, int n, const char * what) {
    const bool call = std::strcmp(what, "call") == 0;
    check(call || std::strcmp(what, "block") == 0, "bench-selfkv: block or call");
    whisper_context * ctx = load_quiet(model_path);
    auto & m = model_tensors(ctx);
    const int n_state = whisper_model_n_text_state(ctx), n_layer = call ? whisper_model_n_text_layer(ctx) : 1;
    const int size = (whisper_model_n_text_ctx(ctx) + 255) / 256 * 256, head = n == 1 ? 226 : 0, n_kv = head + n;
    const float KQscale = pow(float(n_state / (n_state / 64)), -0.25);   // whisper.cpp:2506, the same expression
    ggml_init_params p = { (size_t)64 << 20, nullptr, false };
    ggml_context * g = ggml_init(p);
    check(g != nullptr, "ggml_init failed");
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, n_state, n);
    ggml_tensor * kc = ggml_new_tensor_1d(g, GGML_TYPE_F16, (int64_t)n_state * n_layer * size), * vc = ggml_new_tensor_1d(g, GGML_TYPE_F16, (int64_t)n_state * n_layer * size);
    std::memset(kc->data, 0, ggml_nbytes(kc));
    std::memset(vc->data, 0, ggml_nbytes(vc));
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_tensor * q = nullptr, * mask = nullptr, * mask16 = nullptr;
    for (int il = 0; il < n_layer; il++) {
        const std::string pre = "decoder.blocks." + std::to_string(il) + ".";
        auto W = [&](const char * s) { return conv2_graph::copy(g, m.at(pre + s)); };
        ggml_tensor * cur = ggml_add(g, ggml_mul(g, ggml_norm(g, x, 1e-5f), W("attn_ln.weight")), W("attn_ln.bias"));
        q = ggml_scale(g, ggml_add(g, ggml_mul_mat(g, W("attn.query.weight"), cur), W("attn.query.bias")), KQscale);
        ggml_tensor * k = ggml_scale(g, ggml_mul_mat(g, W("attn.key.weight"), cur), KQscale);
        ggml_tensor * v = ggml_add(g, ggml_mul_mat(g, W("attn.value.weight"), cur), W("attn.value.bias"));
        ggml_build_forward_expand(gf, ggml_cpy(g, k, ggml_view_1d(g, kc, (int64_t)n * n_state, (size_t)2 * n_state * ((size_t)il * size + head))));
        ggml_build_forward_expand(gf, ggml_cpy(g, v, ggml_view_1d(g, vc, (int64_t)n * n_state, (size_t)2 * n_state * ((size_t)il * size + head))));
        ggml_build_forward_expand(gf, q);
    }
    std::vector<float> mk;
    if (call) {
        mask = ggml_new_tensor_3d(g, GGML_TYPE_F32, n_kv, n, 1);
        mask16 = ggml_cast(g, mask, GGML_TYPE_F16);
        ggml_build_forward_expand(gf, mask16);
        mk.assign((size_t)n_kv * n, 0.0f);
        for (int j = 0; j < n; j++) for (int i = 0; i < n_kv; i++) if (i > head + j) mk[(size_t)j * n_kv + i] = -INFINITY;
    }
    std::vector<float> xs((size_t)n * n_state);
    for (size_t i = 0; i < xs.size(); i++) xs[i] = (float)((int)((i * 7919) % 2001) - 1000) / 256.0f;
    std::vector<uint8_t> work;
    auto run = [&]() {
        std::memcpy(x->data, xs.data(), xs.size() * 4);
        if (call) std::memcpy(mask->data, mk.data(), mk.size() * 4);
        ggml_cplan cp = ggml_graph_plan(gf, threads, nullptr);
        if (work.size() < cp.work_size) work.resize(cp.work_size);
        cp.work_data = work.data();
        check(ggml_graph_compute(gf, &cp) == GGML_STATUS_SUCCESS, "graph compute failed");
    };
    run();
    const double c0 = cpu_seconds(), w0 = now_ms();
    long reps = 0;
    while (reps < 20 || now_ms() - w0 < 1000.0) { run(); reps++; }
    const double w1 = now_ms(), c1 = cpu_seconds();
    uint64_t h = digest32((const float *)q->data, (size_t)n * n_state);
    for (ggml_tensor * c : {kc, vc})
        for (int il = 0; il < n_layer; il++)
            for (int r = 0; r < n; r++) h = (h ^ digest16((const uint16_t *)c->data + ((size_t)il * size + head + r) * n_state, n_state)) * 0x100000001b3ULL;
    if (call) for (int j = 0; j < n; j++) h = (h ^ mask_digest((const uint8_t *)mask16->data + (size_t)j * n_kv * 2, n_kv, 2)) * 0x100000001b3ULL;
    std::printf("bench-selfkv-reference threads %d n_tokens %d what %s wall_us_per_call %.3f cpu_us_per_call %.3f reps %ld work_kb %zu out_digest %016llx\n",
                threads, n, what, (w1 - w0) * 1000.0 / reps, (c1 - c0) * 1e6 / reps, reps, (work.size() + 1023) / 1024, (unsigned long long)h);
    ggml_free(g);
    whisper_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--selfkv") == 0) return record_selfkv(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 6 && std::strcmp(argv[1], "--bench-selfkv") == 0) return bench_selfkv(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), argv[5]);
    if (argc >= 4 && std::strcmp(argv[1], "--decin") == 0) return record_decin(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 5 && std::strcmp(argv[1], "--bench-decin") == 0) return bench_decin(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    if (argc >= 4 && std::strcmp(argv[1], "--cross") == 0) return record_cross(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 5 && std::strcmp(argv[1], "--bench-cross") == 0) return bench_cross(argv[2], argv[3], std::atoi(argv[4]));
    if (argc >= 4 && std::strcmp(argv[1], "--encoder") == 0) return record_encoder(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 5 && std::strcmp(argv[1], "--bench-attn") == 0) return bench_attn(argv[2], argv[3], std::atoi(argv[4]));
    if (argc == 5 && std::strcmp(argv[1], "--bench-encode") == 0) return bench_encode(argv[2], argv[3], std::atoi(argv[4]));
    if (argc >= 4 && std::strcmp(argv[1], "--matmul") == 0) return record_matmul(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 4 && std::strcmp(argv[1], "--mm-nan") == 0) return record_mm_nan(argv[2], argv[3]);
    if (argc == 6 && std::strcmp(argv[1], "--bench-mm") == 0) return bench_mm(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
    if (argc >= 4 && std::strcmp(argv[1], "--norm") == 0) return record_norm(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 6 && std::strcmp(argv[1], "--bench-norm") == 0) return bench_norm(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
    if (argc >= 4 && std::strcmp(argv[1], "--conv2") == 0) return record_conv2(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 6 && std::strcmp(argv[1], "--bench-conv2") == 0) return bench_conv2(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
    if (argc >= 4 && std::strcmp(argv[1], "--conv1") == 0) return record_conv1(argv[2], argv[3], argc - 4, argv + 4);
    if (argc == 6 && std::strcmp(argv[1], "--bench-conv1") == 0) return bench_conv1(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
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
