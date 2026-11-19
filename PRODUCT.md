# STT Studio product notes

STT Studio is an early-stage local transcription workspace. This document records current behavior visible in the code and acceptance criteria for the next product iterations. Criteria below are targets for validation; they do not imply that unfinished work is complete.

## Current behavior

The current interface lets a user choose or drop one audio file, play it locally, choose a model and language, adjust compute type, beam size, VAD, and an optional vocabulary hint, then submit a job to the local FastAPI service. The service accepts one transcription at a time, returns queued/running/completed/failed job states, and returns timestamped segments and detected-language metadata when finished. In the browser, users can edit segments for the current session, search the text, click timestamps to seek the audio, copy the transcript, and export TXT, SRT, or JSON.

Projects can be saved explicitly in browser-local storage, including audio, transcript edits, and settings. Backend jobs are not persisted. The backend currently returns results only after inference completes, so the UI cannot show measured percentage progress or a live transcript. Automatic language detection is used when no language is specified.

## Research and design basis

Comparable local transcription tools establish a useful baseline: MacWhisper combines local processing with a built-in editor, synced playback, and subtitle/document exports; Buzz documents searchable playback, keyboard navigation, batch/watch-folder workflows, and multiple local inference backends. These patterns support keeping the source audio and transcript together during review and making export part of the main workspace. [MacWhisper](https://www.macwhisper.com/), [Buzz README](https://github.com/chidiwilliams/buzz/blob/main/README.md).

Model selection should communicate resource tradeoffs without implying a universal runtime promise. OpenAI's Whisper README reports different model sizes and relative speeds, but its speed measurements use an A100 and actual performance varies by hardware, language, and speaking rate. `whisper.cpp` documents Apple Silicon acceleration, CPU inference, and quantization tradeoffs; these are implementation-specific and do not directly predict this app's `faster-whisper` runtime. [Whisper model table](https://github.com/openai/whisper/blob/main/README.md#available-models-and-languages), [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## UX acceptance criteria

### Import and local service

- A user can select an audio file through the file picker or drag one into the drop area; unsupported or empty files produce a clear, actionable error.
- The UI displays whether the local backend is reachable and gives a recovery step when it is offline.
- Audio is sent only to the loopback service. First-run model downloads are disclosed as network activity and do not imply that the recording is uploaded.

### Model and transcription settings

- The default model and compute setting are suitable for a broad range of Macs; model selection communicates download size and relative resource cost as estimates, not a guaranteed speed or accuracy ranking for every device.
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

Portable project files, background batch jobs, partial/live transcript streaming, measured progress, live microphone capture, speaker diarization, and translation remain future work. Each requires a separate product and data-flow decision before it is represented as available in the UI.
