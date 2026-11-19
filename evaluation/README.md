# Offline transcript WER evaluator

This folder contains a dependency-free command-line utility for comparing a reference transcript with a hypothesis transcript. It reads local UTF-8 files only and makes no network requests.

## Usage

Run from the repository root:

```sh
python -m evaluation.wer reference.txt hypothesis.txt
python -m evaluation.wer reference.json hypothesis.json --format json
```

Inputs may be plain UTF-8 text or JSON (use the `.json` extension). JSON can be a string, an array of strings/segment objects, or an object with one of these fields: `transcript`, `text`, `segments`, `result`, or `transcription`. Segment objects should contain a `text` field. The report includes reference and hypothesis word counts, substitutions, insertions, deletions, total errors, and WER. JSON output uses `null` for WER when the reference has no recognized words; the text output says `undefined (empty reference)`. Insertions remain reported in that case, but dividing by zero is avoided.

The evaluator normalizes Unicode to NFC, case-folds text, converts curly apostrophes to straight apostrophes, and ignores other punctuation and whitespace when tokenizing. It uses a minimum-edit Levenshtein alignment with unit cost for substitutions, insertions, and deletions. Punctuation-only and capitalization differences therefore do not count as errors.

## WER limitations

WER is `S + I + D` divided by the number of reference words. It is a useful measure of lexical recognition errors, but it is not a measure of meaning, readability, speaker attribution, punctuation quality, or timestamp accuracy. It can exceed 100% when insertions are numerous. This tool's punctuation-insensitive, case-insensitive normalization intentionally omits those dimensions.

The tokenizer treats Unicode word characters as word tokens separated by punctuation or whitespace. That convention works for many space-delimited transcripts, but word segmentation differs across languages; WER may be misleading for languages that do not normally separate words with spaces or for scripts requiring language-specific segmentation. Contractions, clitics, hyphens, and number formatting also depend on annotation conventions. Keep one consistent convention for the reference and hypothesis. Empty-reference WER is undefined and reported as `null`, even when both inputs are empty.

## Repeatable calibration method

1. Create a small evaluation set from synthetic, openly licensed, or explicitly consented recordings; do not use private meeting files as fixtures. Keep the reference text separate from the model output.
2. Freeze the set and document language, domain, speaker/accent coverage, recording conditions, transcription policy, and the exact normalization rules. Use enough varied utterances to avoid tuning to one speaker or clip.
3. Record the model identifier/revision, inference library version, compute type, language setting, decoding parameters, audio preprocessing, and application revision for each run.
4. Compare each candidate run with the same references. Report micro-averaged WER from summed error counts divided by summed reference word counts, alongside per-recording counts/rates and the number of reference words. Do not average per-file WERs without also showing the aggregate because short clips can dominate that average.
5. Change one inference or preprocessing factor at a time, keep the set fixed, and compare paired per-recording results. Review notable regressions against the audio; select a configuration based on the intended language/domain and resource budget, rather than optimizing a single aggregate score alone.
6. Keep a held-out set for a final check after tuning. Store only consented/redistributable evaluation materials and avoid committing sensitive audio or transcripts.

## Tests

Run the synthetic unit tests with the Python standard library:

```sh
python -m unittest discover -s evaluation/tests
```
