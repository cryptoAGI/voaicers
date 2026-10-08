# Attribution — the external projects voaice.rs stands on

voaice.rs and streamair are original code, MIT OR Apache-2.0, with **zero dependencies**: no external crate is
compiled in, and no file below is redistributed in this repository. The projects listed here are the **references**
the code is measured against, the **standards** it implements, and the **work** it draws on. Each one is credited
here with its licence.

## References — compiled and run only as oracles

| project | licence | role here |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp), by Georgi Gerganov and contributors | MIT | voaice.rs reproduces its compiled output bit for bit. It is pinned at commit [`080bbbe8`](https://github.com/ggml-org/whisper.cpp/commit/080bbbe85230f624f0b52127f1ae1218247989f9), the build mindX production runs ([upstream/PIN](upstream/PIN), [docs/REFERENCE.md](docs/REFERENCE.md)). The checkout and build live in `upstream/`, which is gitignored |
| [ggml](https://github.com/ggml-org/ggml) 0.16.0, by Georgi Gerganov and contributors | MIT | the tensor library inside whisper.cpp. Its CPU kernels set the float order the encoder and decoder must reproduce |
| [libopus](https://opus-codec.org) 1.4 and [opus-tools](https://github.com/xiph/opus-tools) 0.2, by Xiph.Org and contributors | BSD-3-Clause | streamair's oracle: production's `opusinfo` and `opusdec` judge every file, and libopus's encoder is the target for byte-exact packets. No code is copied from it |

## Models and data

| project | licence | role here |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper), by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey and Ilya Sutskever ([paper](https://arxiv.org/abs/2212.04356)) | MIT (code and weights) | the model and its design. voaice.rs implements the architecture |
| [`ggml-tiny.en.bin`](https://huggingface.co/ggerganov/whisper.cpp) (also `base.en`), Whisper's weights in ggml format, from the whisper.cpp Hugging Face repository | MIT | the model the oracles run on, pinned by sha256. It is downloaded into `models/`, which is gitignored |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) Face Landmarker, by Google | Apache-2.0 | its face-mesh tessellation (the canonical face model's connections) is a test fixture for streamair's fclone (`streamair/tests/fixtures/mediapipe_tessellation.json`): the triangles fCLONE reads and checks. The landmarker itself runs in the browser (ollywoo), not here |
| [Piper](https://github.com/rhasspy/piper) and its [voices](https://huggingface.co/rhasspy/piper-voices), by Michael Hansen and contributors | MIT (code). Each voice has its own licence | the 70 open-licensed voices in [PYTHAI/voaice](https://huggingface.co/PYTHAI/voaice), each published with Piper's attribution and its own licence and model card |

## Standards implemented

| standard | authors | what streamair and voaice.rs implement |
|---|---|---|
| [RFC 6716](https://www.rfc-editor.org/rfc/rfc6716) — the Opus codec | Jean-Marc Valin, Koen Vos, Timothy B. Terriberry | TOC parsing now; the CELT and SILK encoders next |
| [RFC 7845](https://www.rfc-editor.org/rfc/rfc7845) — Ogg encapsulation for Opus | Jean-Marc Valin, Ron Lee, Timothy B. Terriberry | OpusHead, OpusTags, granule position, pre-skip, end trimming |
| [RFC 3533](https://www.rfc-editor.org/rfc/rfc3533) — the Ogg format | Silvia Pfeiffer | pages, lacing, the CRC |
| RIFF/WAVE | Microsoft and IBM | the 16-bit PCM input reader |

## Behaviour matched, not code copied

| project | licence | what is matched |
|---|---|---|
| [glibc](https://www.gnu.org/software/libc/) 2.35 libm | LGPL-2.1 | voaice.rs calls the same `sincosf`, `cosf` and `log10` that libwhisper imports, through the system's dynamic libm. **Note for the portability TODO:** glibc is LGPL, so an in-crate port of those functions must be clean-room, or taken from a permissively licensed source such as [ARM optimized-routines](https://github.com/ARM-software/optimized-routines) (MIT OR Apache-2.0 WITH LLVM-exception, where glibc's single-precision `sincosf` originated). It must not be copied from glibc |
| [miniaudio](https://github.com/mackron/miniaudio), by David Reid | public domain or MIT-0 | whisper-cli's WAV path. Its s16→f32 constant is exactly 2⁻¹⁵, and voaice.rs matches it |

## Inspiration and comparison

| project | licence | how it is used |
|---|---|---|
| [Kitten TTS](https://github.com/KittenML/KittenTTS) v1 ([nano 0.1](https://huggingface.co/KittenML/kitten-tts-nano-0.1), [0.2](https://huggingface.co/KittenML/kitten-tts-nano-0.2)), by KittenML | Apache-2.0 | the shape of the speaking half: about 15M parameters, CPU-only, a voice table. A candidate oracle |
| [Kitten TTS 2](https://huggingface.co/KittenML/kitten-tts-2), by KittenML / Stellon Labs | [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), not open | design ideas only: ternary GGUF, in-context cloning. No code and no weights are taken; any work with it is measurement, never shipping |
| [KittenML ASR](https://huggingface.co/KittenML/kitten-asr-tiny) | no licence stated | a point of comparison for the listening half, not a dependency |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), by hexgrad | Apache-2.0 | stage 1 of voaice's NeuralVoiceEngine (vCLONE synthesis), run through ONNX. Future work in Rust would be checked against it |
| [OpenVoice v2](https://github.com/myshell-ai/OpenVoice), by MyShell and MIT CSAIL | MIT | stage 2 of the same engine: tone-colour transfer |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), by Microsoft | MIT | the runtime voaice's JavaScript engine uses for those models, and the oracle for any Rust port of them |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), by Georgi Gerganov and contributors | MIT | the reference [bankml](https://github.com/cryptoAGI/bankml) was built against. voaice.rs follows bankml's method |

If a credit here is missing or wrong, it is a bug: please open an issue.
