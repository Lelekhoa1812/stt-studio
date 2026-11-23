// Display names for model publishers, bundled with the app. Only names are kept here: no images and no network lookups.
// Keys are lowercase publisher slugs or names as the engine reports them. Unknown publishers show their reported name.
const PUBLISHER_DISPLAY_NAMES: Record<string, string> = {
  'distil-whisper': 'Distil-Whisper',
  'distil-whisper contributors': 'Distil-Whisper contributors',
  funaudiollm: 'FunAudioLLM',
  'ibm granite': 'IBM Granite',
  'microsoft research': 'Microsoft Research',
  'mlx-community': 'MLX Community',
  nvidia: 'NVIDIA',
  openai: 'OpenAI',
  pyannote: 'pyannote',
  qwen: 'Qwen',
  systran: 'SYSTRAN',
  'useful sensors': 'Useful Sensors',
  'useful-sensors': 'Useful Sensors',
}

export function publisherDisplayName(publisher: string): string {
  const name = publisher.trim()
  return PUBLISHER_DISPLAY_NAMES[name.toLowerCase()] ?? name
}
