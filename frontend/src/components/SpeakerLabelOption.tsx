import { useId } from 'react'

export type SpeakerSupport = { checked: boolean; available: boolean; reason: string | null }

export const UNKNOWN_SPEAKER_SUPPORT: SpeakerSupport = { checked: false, available: false, reason: null }

export function SpeakerLabelOption({ checked, support, disabled, onChange }: { checked: boolean; support: SpeakerSupport; disabled: boolean; onChange: (value: boolean) => void }) {
  const hintId = useId()
  const available = support.checked && support.available
  const hint = !support.checked
    ? 'Checking speaker label support…'
    : available
      ? 'Adds Speaker 1, Speaker 2 labels with a local model after transcription. If labelling fails, the transcript is kept.'
      : support.reason || 'Speaker labels are not available on this server.'
  return (
    <div className="speaker-option">
      <label className="speaker-option-label">
        <input type="checkbox" checked={available && checked} disabled={!available || disabled} onChange={e => onChange(e.target.checked)} aria-describedby={hintId}/>
        <span><strong>Label speakers</strong></span>
      </label>
      <small id={hintId}>{hint}</small>
    </div>
  )
}
