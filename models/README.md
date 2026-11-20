# Model catalog

`catalog.json` is the source-linked inventory for model-family discovery. It separates models that have a verified CTranslate2 repository for the current `faster-whisper` backend from candidate families that need a different runtime. `backend/app.py` reads the catalog and exposes only families marked `supported` through the runnable model API. Download sizes in the UI are approximate repository sizes; quantization affects memory use and does not rewrite the cached source files.

## Current runtime

The current backend can load Whisper checkpoints in CTranslate2 format. Built-in Whisper aliases resolve to Systran's `faster-whisper` conversions. Large v3 Turbo uses the public `dropbox-dash/faster-whisper-large-v3-turbo` conversion because the similarly named Systran repository was not publicly accessible during review. The curated Distil-Whisper choices use Systran's converted repositories. These records were checked against the public Hugging Face model API and the repositories exposed CTranslate2 files such as `model.bin`, `config.json`, `preprocessor_config.json`, and tokenizer/vocabulary files. Their repository cards listed MIT at the time of review.

An arbitrary Hugging Face repository ID is not automatically compatible: it must contain a CTranslate2 model supported by the installed `faster-whisper` version. A Transformers checkpoint made of `.safetensors` files does not become loadable just because it is an ASR model. Model aliases and converted repositories can change; verify the exact model card and files before adding a preset.

## Family candidates

Parakeet, Canary, IBM Granite Speech, Moonshine, Qwen3-ASR, VibeVoice-ASR, and FunASR are discovery candidates for future adapters. They use different model formats or runtimes and are not usable by the current backend. In particular, [MLX-Audio](https://github.com/Blaizzy/mlx-audio) documents Apple Silicon support for both the [Parakeet MLX conversion](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) and [Qwen3-ASR 0.6B 8-bit conversion](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit), making it a promising shared runtime candidate. These are third-party conversions and remain unvalidated in STT Studio. Their reviewed card licenses differ: Canary is CC-BY-NC-4.0 and therefore non-commercial; Parakeet is CC-BY-4.0; Granite Speech and Qwen3-ASR are Apache-2.0; the cited Moonshine and VibeVoice-ASR cards are MIT; and the selected Fun-ASR checkpoint is Apache-2.0. VibeVoice's official setup recommends CUDA, so Mac hardware support needs particular validation. These labels describe the exact Hugging Face repository card reviewed, not all models in a family or every underlying component. Check the card and repository license files again before download or redistribution.

## Publisher identity and images

The model card and repository owner identify the publisher of the weights or conversion being loaded; the family publisher identifies the upstream model family. For example, a faster-whisper Whisper conversion is published by SYSTRAN while Whisper itself is from OpenAI. The UI shows the exact repository publisher's Hugging Face profile image and keeps upstream attribution separate. Hugging Face's public organization avatar endpoint returns a profile image, but does not certify it as an official logo or grant trademark rights. Treat it as a publisher profile image, link it to the source profile, and label its provenance. Only use an “official logo” label after independently verifying the asset on the publisher's own site and checking that its terms allow this use. Never infer model identity or trust from image recognition alone; use the repository ID and card as the identity source.

The catalog is a discovery aid. Before a model is downloaded, show the exact repository, publisher/source link, runtime compatibility, download size where verified, and applicable model license or restrictions. Do not imply that a family logo means the model weights are published by that organization.

## Research sources

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) documents CTranslate2-based Whisper inference.
- [OpenAI Whisper](https://github.com/openai/whisper) documents the original Whisper family; model-format compatibility depends on the selected converted repository.
- [Hugging Face Hub model API](https://huggingface.co/docs/hub/api) provides repository metadata and public organization avatar/profile endpoints.
- [Buzz's documented backend list](https://chidiwilliams.github.io/buzz/docs) is a useful comparison for a broader family matrix, including Qwen3-ASR, VibeVoice-ASR, FunASR, Parakeet, and Whisper variants.
- [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Microsoft VibeVoice-ASR runtime notes](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md) were reviewed for formats, hardware prerequisites, and source links.
- [MLX-Audio's supported model list and examples](https://github.com/Blaizzy/mlx-audio) document local Apple Silicon inference for Whisper, Distil-Whisper, Qwen3-ASR, Parakeet, and other speech models; check exact conversion cards for their own licenses and requirements.
- Individual model-card links and exact repository names are recorded in [`catalog.json`](catalog.json). The model card and files are authoritative for each specific model; family-level descriptions are not license grants.

## MLX-Audio runtime validation (Apple Silicon)

The optional `mlx-audio` runtime (`backend/requirements-mlx.txt`) is registered only on Apple Silicon when the package is installed. Validation so far, on one Mac:

- `mlx-community/Qwen3-ASR-0.6B-8bit` downloaded into the project's `models/` cache and was detected by the cache check. A cold load including a roughly 1 GB download took about 80 seconds; a warm load took about 1 second. Warmup took about 0.2 seconds and a 4.8-second clip transcribed correctly. Peak process memory was about 2 GB, measured on a single short clip.
- Long audio (about 117 seconds, 253 words) transcribed correctly in about 2.4 seconds. Three load and transcribe cycles had peak memory of about 1.1 GB, 1.3 GB and 1.3 GB, so memory is roughly stable after the first reload. The adapter does not clear the MLX cache on eviction; this is a known follow-up.
- **Timestamp limitation:** Qwen3-ASR returns one untimed segment covering the whole file, so SRT and VTT export produce a single caption. Per-segment timestamps would need a forced aligner or another approach. This is why the family is marked as a limitation in its catalog note.
- **Promotion:** `Qwen3-ASR (MLX conversion)` is now `runtimeStatus: supported` on the Apple Silicon MLX-Audio runtime only. The audit script checks `.safetensors` weights for `mlx-audio` families.
- **Parakeet (`mlx-community/parakeet-tdt-0.6b-v3`) failed to load** with the installed `mlx-audio` 0.6.7 (the latest on PyPI). The repository's `config.json` is a raw NeMo config without a `model_type`, so the loader cannot choose an architecture. The weights downloaded completely (about 2.5 GB), so this is a loader-compatibility issue, not a download failure. Parakeet stays **candidate** and is not offered in the switcher.
- Still unvalidated: quantization variants other than 8-bit, and repeated switches across different MLX families.
- MLX conversions are third-party `mlx-community` assets, not first-party publisher weights. Do not present them as official.

## Recheck repository metadata

Run the optional network audit after changing catalog entries. It checks public visibility, the recorded license tag, and the weight artifact for supported models (`model.bin` for faster-whisper, `.safetensors` for mlx-audio):

```sh
python3 scripts/verify_model_catalog.py
```

The script is an audit aid, not a download or license approval step. Re-review model cards and applicable terms before presenting a new model as supported.
