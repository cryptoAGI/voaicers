# 0.0.5 research notes — what whisper-cli's read_audio_data really does (pinned 080bbbe8)

Working notes, written as found. Line numbers are upstream/whisper.cpp at the pin.

## The call
- examples/common-whisper.cpp:78 `read_audio_data(fname, pcmf32, pcmf32s, stereo)`; whisper-cli passes
  `stereo = params.diarize` (false by default).
- `decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE /*16000*/)`.
- `ma_decoder_init_file` (miniaudio). ffmpeg only if built with WHISPER_COMMON_FFMPEG (not in our build, not in
  production's cmake line) and only as a fallback if miniaudio fails. `WHISPER_COMMON_MINIAUDIO_SKIP` env skips it.
- `read_audio_from_decoder`: `ma_decoder_get_length_in_pcm_frames` -> resize vector to that length ->
  `ma_decoder_read_pcm_frames(frame_count)`; frames_read is NOT used to shrink the vector (if fewer frames come out
  than the length promised, the tail stays 0.0f from resize).
- stereo (diarize) path: pcmf32[i] = L + R (a SUM, no /2), plus the split channels. Not the default path.
- vendored miniaudio: examples/miniaudio.h, MA_VERSION 0.11.24 (David Reid). Defines before include:
  MA_NO_DEVICE_IO, MA_NO_THREADING, MA_NO_ENCODING, MA_NO_GENERATION, MA_NO_RESOURCE_MANAGER, MA_NO_NODE_GRAPH.

## How it is compiled (build/compile_commands.json)
- common-whisper.cpp.o: `c++ ... -O3 -DNDEBUG -fPIC` — NO -march=native (GGML_NATIVE applies to ggml only).
  So: x86-64 baseline (SSE2), no FMA possible. objdump on the object: 0 vfmadd, 0 ymm. Float ops are plain
  SSE scalar/packed IEEE single; the port must just keep the op order (no fused ops in Rust by default -> OK).
- whisper-cli links libcommon statically (examples/CMakeLists.txt target `common`).

## Decoder -> converter
- `ma_decoder_config_init`: zeroed config, resampling = ma_resampler_config_init(..., ma_resample_algorithm_linear);
  linear.lpfOrder = min(MA_DEFAULT_RESAMPLER_LPF_ORDER=4, MA_MAX_FILTER_ORDER=8) = 4. ditherMode 0 = none,
  channelMixMode 0 = rectangular, channel map empty -> default.
- The backend is told `preferredFormat = config.format = f32` (ma_decoding_backend_config_init(pConfig->format..)),
  and ma_wav accepts f32 -> the WAV backend's internal format is f32 for EVERY WAV. So the int->float step is
  dr_wav's (embedded as ma_dr_wav), NOT ma_pcm_*_to_f32:
    - u8 : x = (float)u; x = x * 0.00784313725490196078f; x = x - 1
    - s16: s * 0.000030517578125f  (2^-15 exact)
    - s24: (float)((double)(int24) * 0.00000011920928955078125)   (exact in double, then one rounding)
    - s32: (float)(s / 2147483648.0)                               (double divide, one rounding)
    - f32 (IEEE, 32-bit): copied
    - (f64, alaw, mulaw, adpcm exist; not in this corpus)
- ma_data_converter: formatIn = formatOut = mid = f32 -> no pre/post format conversion.
  channels_first when channelsIn >= channelsOut: channel converter, then resampler (resampler runs at the lowest
  channel count = 1).
  16 kHz mono -> passthrough (bits untouched). 16 kHz stereo -> channels_only.

## Channel mixdown (default, non-diarize: output channels = 1)
- channel map out NULL + channelsOut 1 -> ma_channel_conversion_path_mono_out (not the weights path; channel
  positions and rectangular weights are irrelevant). f32: `float t = 0; t += in[c] for c in 0..n; out = t / n`
  (n = channelsIn as uint32 -> float). Stereo: ((0 + L) + R) / 2.0f -> same bits as (L + R) / 2.
  (Contrast: the diarize path in common-whisper sums L + R with no division. Not the default; not ported.)

## Resampler: miniaudio's linear resampler, f32, 1 channel (resampler runs after the mixdown)
- rates reduced by gcd: 48000->16000 = 3:1, 44100->16000 = 441:160, 22050 = 441:320, 24000 = 3:2, 32000 = 2:1,
  8000 = 1:2. inAdvanceInt = in/out, inAdvanceFrac = in%out (reduced). Start: inTimeInt = 1, inTimeFrac = 0,
  x0 = x1 = 0.
- lpfOrder = 4, lpfNyquistFactor = 1 (ma_resampling_backend_get_config__linear calls
  ma_linear_resampler_config_init then copies only lpfOrder). LPF sampleRate = max(reduced in, out),
  cutoff = (double)(min(reduced) * 0.5 * 1).
- order 4 -> 0 first-order + 2 biquads (lpf2). For i in 0..2: a = (1 + i*2) * (MA_PI_D / (order*2)),
  q = 1 / (2 * ma_cosd(a)). ma_cosd(x) = sin(MA_PI_D*0.5 - x)  — glibc `sin` (double), nm -u on the object shows
  only `sin` among libm calls. Biquad (lpf2): w = 2*MA_PI_D*cutoff/sampleRate; s = sin(w); c = ma_cosd(w);
  a = s/(2q); b0 = (1-c)/2, b1 = 1-c, b2 = (1-c)/2, a0 = 1+a, a1 = -2c, a2 = 1-a; stored as (float)(bk/a0).
  MA_PI_D = 3.14159265358979323846264.
- biquad f32, direct form 2 transposed: y = b0*x + r1; r1 = b1*x - a1*y + r2; r2 = b2*x - a2*y. No FMA.
- DOWNsample (in > out): for each output: while inTimeInt > 0 and input left: x0 = x1; x1 = next; x1 = LPF(x1)
  (filter only if reduced in != out); inTimeInt -= 1. Then out = x0 + (x1 - x0) * a with
  a = (float)inTimeFrac / (float)sampleRateOut (reduced). Then advance inTimeInt += advInt, inTimeFrac += advFrac,
  carry when frac >= out.
- UPsample (in < out): load without filtering; out = mix(...) then LPF applied to the OUTPUT sample.
- first output: with inTimeInt = 1 the first input is loaded into x1 with x0 = 0 and frac = 0 -> out[0] = x0 = 0
  (the linear resampler delays by one input frame).
- Output length = ma_decoder_get_length_in_pcm_frames = ma_calculate_frame_count_after_resampling(16000, rateIn, N)
  — NOTE its formula uses the UNREDUCED rates and a different expression than
  ma_linear_resampler_get_expected_output_frame_count:
    out = N*16000/rateIn; fromFrac = (out*(rateIn/16000))/16000; pre = out*(rateIn%16000) + fromFrac;
    if pre <= N: out += 1.
  The vector is resized to that and read; frames actually produced may be fewer (tail stays 0.0f from resize).
  This needs the oracle to confirm.
- the decoder loop feeds the converter in chunks (4096-byte stack buffers, ma_*_get_required_input_frame_count);
  values should not depend on chunking (the resampler is sample-sequential), but whether every input frame is
  consumed is for the oracle to confirm.

## Plan for the oracle
- Link the REAL build/examples/libcommon.a (the object whisper-cli links; -O3 -DNDEBUG -fPIC, no -march) and call
  read_audio_data(path, pcmf32, pcmf32s, false); dump the f32 vector.

## Oracle first run (resample_oracle linking the real libcommon.a), corpus testing/make_resample_audio.py
- 55 files; output lengths agree with ma_calculate_frame_count_after_resampling as written above (e.g. 48 kHz
  N=62413 -> 20805; 8 kHz N=10407 -> 20814; 48 kHz N=1 -> 1; 8 kHz N=1 -> 2).
- A LIST chunk whose declared size was wrong made read_audio_data FAIL (dr_wav refuses); fixed the generator.
- Truncated data chunk (claims 48,000 frames, holds 30,000): length 10001 = rule(30000) -> dr_wav clamps the data
  size to what the file holds. Last sample 0.0 (resampler produced one fewer than the length rule promised).

## Port (src/resample.rs) against the oracle — 2026-10-08
- 55/55 files, 1,955,875 samples bit-identical on the first comparison (tests/resample.rs). 3 files carry the
  length rule's zero tail (48k n3, the truncated file, jfk_48k); confirms the "promise one more" reading.
- streaming: 55/55 files pushed in random 1..9000-byte pieces equal the reference.
- discriminators: LPF order 2 -> 45/50 resampled files differ, order 6 -> 45/50 (the 5 not caught are 1-2-frame
  inputs that emit only the leading 0); mixdown L+R 8/8, L only 8/8; length without the tail 3/55.
- No FMA anywhere: the reference object has none (no -march), Rust emits none without explicit mul_add.
