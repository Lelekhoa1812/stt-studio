import { Copy, LoaderCircle, RefreshCw } from 'lucide-react'

export type AiMode = 'off' | 'summary' | 'action_items'
export type AiResult = { mode: string; text: string; model?: string; created_at?: number }
export type AiSupport = { checked: boolean; available: boolean; reason: string | null }

const OPTIONS: Array<[AiMode, string]> = [['off', 'Off'], ['summary', 'Summary'], ['action_items', 'Action items']]
const TITLES: Record<Exclude<AiMode, 'off'>, string> = { summary: 'Summary', action_items: 'Action items' }
const cx = (...xs: Array<string | false | undefined | null>) => xs.filter(Boolean).join(' ')

type Props = {
  support: AiSupport
  ready: boolean
  mode: AiMode
  result?: AiResult
  pending: boolean
  error: string
  onSelect: (mode: AiMode) => void
  onRun: () => void
  onCopy: (text: string) => void
}

export function AiModes({ support, ready, mode, result, pending, error, onSelect, onRun, onCopy }: Props) {
  const enabled = ready && support.available
  const hint = !support.checked ? 'Checking AI mode support…' : !support.available ? support.reason : !ready ? 'AI modes are available once the transcript is complete.' : null
  return <div className="ai-modes">
    <div className="ai-modes-bar">
      <div className="ai-switcher" role="radiogroup" aria-label="AI mode">
        {OPTIONS.map(([value, label]) => <button key={value} type="button" role="radio" aria-checked={mode === value} disabled={!enabled} className={cx('ai-option', mode === value && 'is-selected')} onClick={() => onSelect(value)}>{label}</button>)}
      </div>
      <p className="ai-notice">Selecting a mode sends the transcript text, never the audio, to NVIDIA NIM.</p>
    </div>
    {hint && <p className="ai-hint">{hint}</p>}
    {mode !== 'off' && (pending
      ? <div className="ai-status" role="status"><LoaderCircle size={15} className="spin"/> Generating {TITLES[mode].toLowerCase()}. This can take up to two minutes.</div>
      : error
        ? <div className="ai-error" role="alert"><div><strong>{TITLES[mode]} could not be generated.</strong><span>{error}</span></div><button type="button" className="button-secondary" onClick={onRun} disabled={!enabled}><RefreshCw size={14}/> Try again</button></div>
        : result
          ? <section className="ai-result" aria-label={`${TITLES[mode]} result`}>
              <div className="ai-result-head"><h3>{TITLES[mode]}</h3><div className="ai-result-actions"><button type="button" className="button-secondary" onClick={onRun} disabled={!enabled}><RefreshCw size={14}/> Regenerate</button><button type="button" className="button-secondary" onClick={() => onCopy(result.text)}><Copy size={14}/> Copy AI result</button></div></div>
              <div className="ai-text">{result.text}</div>
              <small>Written by {result.model ?? 'NVIDIA NIM'}. Check it against the transcript before you rely on it.</small>
            </section>
          : null)}
  </div>
}
