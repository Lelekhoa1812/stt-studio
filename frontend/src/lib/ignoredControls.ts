const CONTROL_LABELS: Record<string, string> = {
  beam_size: 'beam size',
  vad_filter: 'voice activity detection',
  initial_prompt: 'vocabulary hint',
}

function label(control: string) {
  return CONTROL_LABELS[control] ?? control.replace(/_/g, ' ')
}

export function ignoredControlsMessage(controls: unknown): string | null {
  if (!Array.isArray(controls) || !controls.length) return null
  const names = controls.map(item => label(String(item)))
  const tail = names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`
  return `This model ignored ${tail}.`
}