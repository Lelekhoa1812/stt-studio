# STT Studio product notes

STT Studio is an early-stage local transcription workspace. This document records current behavior visible in the code and acceptance criteria for the next product iterations. Criteria below are targets for validation; they do not imply that unfinished work is complete.

## Current behavior

The current interface lets a user choose or drop one audio file, play it locally, choose a supported model and language, adjust compute type, beam size, VAD, and an optional vocabulary hint, then submit a job to the local FastAPI service. The service runs one transcription at a time through a runtime adapter: faster-whisper everywhere, and the optional mlx-audio runtime on Apple Silicon, where only the Qwen3-ASR MLX conversion is supported. It returns queued, running, completed, failed and cancelled job states, and timestamped segments with detected-language metadata when finished. In the browser, users can edit segments for the current session, search the text, click timestamps to seek the audio, copy the transcript, and export TXT, SRT, or JSON.

Projects can be saved explicitly in browser-local storage, including audio, transcript edits, and settings, and exported as portable `.sttproj` files. A batch queue runs several files in order and downloads per-file results. A running or queued job can be cancelled, and the partial transcript is kept. Progress is measured by segment for faster-whisper and indeterminate for MLX. Backend jobs, batches and transcripts persist in a local SQLite database until the user deletes them; audio is never stored. A job that was queued or running when the server stopped is marked failed with "Interrupted by a server restart", and job and batch ids stay valid across restarts. Automatic language detection is used when no language is specified.

The local API and the `scripts/stt_cli.py` command-line client expose the same workflow for scripts and AI agents, including video input, probing, long-file chunking and TXT, SRT, VTT and JSON exports. Optional speaker labels (`diarize`, `--diarize`) run a local pyannote model on timed transcripts. The web interface offers them for single files and batches, lets users rename speakers, and includes the names in exports. Segments appear while a faster-whisper job runs, and editing unlocks once it finishes. Batches show per-file status only.

## Research and design basis

Comparable local transcription tools establish a useful baseline. MacWhisper combines local processing with a built-in editor, synced playback, subtitle/document exports, speaker tools, and portable projects. Buzz documents searchable playback, keyboard navigation, model download/delete controls, quantized models, watch folders, plugins, and multiple local inference backends. Its current backend list includes Whisper.cpp, Faster Whisper, Parakeet, Qwen3-ASR, VibeVoice-ASR, and FunASR; that breadth demonstrates why model families need runtime adapters instead of one generic model-name field. These patterns support keeping the source audio and transcript together during review and making model lifecycle, editing, and export part of the main workspace. [MacWhisper](https://www.macwhisper.com/), [Buzz feature overview](https://chidiwilliams.github.io/buzz/docs), [Buzz model and folder-watch settings](https://chidiwilliams.github.io/buzz/docs/preferences).

Model selection should communicate resource tradeoffs without implying a universal runtime promise. OpenAI's Whisper README reports different model sizes and relative speeds, but its speed measurements use an A100 and actual performance varies by hardware, language, and speaking rate. `whisper.cpp` documents Apple Silicon acceleration, CPU inference, and quantization tradeoffs; these are implementation-specific and do not directly predict this app's `faster-whisper` runtime. [Whisper model table](https://github.com/openai/whisper/blob/main/README.md#available-models-and-languages), [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

The curated catalog now records the runtime, exact repository, publisher profile, language coverage, license, and download-size estimate for each listed checkpoint. It includes Whisper and Distil-Whisper conversions that the current backend can run, plus Parakeet, Canary, Granite Speech, Moonshine, Qwen3-ASR, VibeVoice-ASR, and FunASR as explicitly unavailable adapter candidates. Research also identified MLX-Audio conversions for both Parakeet and Qwen3-ASR as a promising shared Apple Silicon runtime, but they remain unavailable until the adapter, device matrix, option mapping, and real audio tests pass. VibeVoice-ASR's official setup currently recommends a CUDA environment. [MLX-Audio supported STT models](https://github.com/Blaizzy/mlx-audio), [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [VibeVoice-ASR runtime notes](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md).

The runtime adapter boundary now covers load, warmup, and normalized inference results; the next model-platform increment should add capability metadata for cancellation, device support, quantization formats, and emitted transcript features. Each catalog entry should state which controls it supports (language forcing, VAD, word timestamps, diarization, hotwords, translation) and disable or explain unsupported controls instead of silently ignoring them. Model names and publisher profile images are discovery aids; the exact model-card URL and license remain the source of truth. The Hugging Face avatars used by the UI are publisher profile images, not verified official logo assets.

## UX acceptance criteria

### Import and local service

- A user can select an audio file through the file picker or drag one into the drop area; unsupported or empty files produce a clear, actionable error.
- The UI displays whether the local backend is reachable and gives a recovery step when it is offline.
- Audio is sent only to the loopback service. First-run model downloads are disclosed as network activity and do not imply that the recording is uploaded.

### Model and transcription settings

- The default model and compute setting are suitable for a broad range of Macs; model selection communicates download size and relative resource cost as estimates, not a guaranteed speed or accuracy ranking for every device.
- Each model choice shows its family, runtime, repository card, license, estimated download size, and local cache state; model options must only be enabled when the installed adapter can load them.
- Runtime-specific options are capability-gated. A selected model cannot silently accept VAD, quantization, diarization, or language controls that its adapter ignores.
- Automatic language detection is passed to the backend in the inference library's supported form. A selected language is passed through unchanged and the detected language is visible after completion.
- Advanced controls have sensible bounds and labels that explain their effect. Changing a setting does not claim to improve both accuracy and speed when it may trade one for the other.
- Model download/loading shows a distinct state and failure guidance. The model is described as ready only after successful load/warmup.

### Job status and failure recovery

- The UI distinguishes queued, running, completed, and failed jobs using the backend response.
- Progress is presented as a percentage only when the backend supplies measured progress. Until then, the interface uses an indeterminate processing state.
- A failed job preserves a readable error and allows the user to retry without reselecting the file or losing transcription settings.
- The single-job limit is communicated before the user submits another job, or the backend returns a clear busy response.

### Transcript review and export

- Completed segments show timestamps and text. Selecting a timestamp seeks the source audio to that segment; current playback position highlights the corresponding segment.
- Editing a segment updates the text used by copy and each export format.
- Search filters visible segments and clearly reports when there are no matches.
- TXT, SRT, and JSON exports preserve their documented content. Export is unavailable until a transcript exists.
- The interface clearly states whether jobs and edits are session-only or persisted; it must not imply that the history sidebar saves files until persistence exists.

## Not implemented

Still future work: watch folders, live microphone capture, live text for batches, and translation. Speaker labels have been measured on few recordings, and they are not offered for the untimed MLX runtime. Alternate runtime adapters are also future work: catalog presence alone must never imply that Parakeet, Canary, Granite Speech, Moonshine, Qwen3-ASR, VibeVoice-ASR, or FunASR can run in this release.
