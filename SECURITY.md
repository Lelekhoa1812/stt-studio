# Security Policy

STT Studio is an early-stage local transcription prototype. It is not designed as a hardened service and should not be exposed to untrusted networks.

## Supported versions

There are no versioned releases yet. Security fixes will be made on the current development branch while the project remains active.

## Reporting a vulnerability

Please do not publish recordings, transcripts, model files, credentials, or exploit details in a public issue. Contact the repository maintainers privately with a description, affected revision, reproduction steps, and potential impact. No private reporting address is configured in this early-stage repository; if you cannot contact a maintainer privately, report only a minimal description publicly and avoid including sensitive data.

## Security and privacy notes

- The FastAPI service accepts audio on its local HTTP endpoint and stores uploads in a temporary file while processing. It removes that file when processing completes or fails.
- The model may be downloaded on first use; check your environment's network policy before doing so.
- The API does not implement authentication or network service protections. Bind it to `127.0.0.1` and do not expose it to untrusted networks.
- Treat source audio and generated transcripts as sensitive. The repository ignore rules exclude common recordings, outputs, logs, model caches, and local environments, but verify `git status` before sharing changes.
