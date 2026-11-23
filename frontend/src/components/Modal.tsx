import { useEffect, useId, useRef, type KeyboardEvent } from 'react'
import { X } from 'lucide-react'

const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]):not([type=hidden]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])'

export function Modal({open,title,onClose,children,wide=false}:{open:boolean;title:string;onClose:()=>void;children:React.ReactNode;wide?:boolean}) {
  const ref = useRef<HTMLDialogElement>(null)
  // Each instance gets its own ids, so two dialogs with the same title never collide.
  const id = useId()
  const titleId = `${id}-title`
  const bodyId = `${id}-body`
  useEffect(()=>{const dialog=ref.current;if(!dialog)return;if(open&&!dialog.open){dialog.showModal();requestAnimationFrame(()=>dialog.querySelector<HTMLElement>('[data-autofocus]')?.focus())}else if(!open&&dialog.open)dialog.close()},[open])
  // Native dialogs do not keep Tab inside themselves, so cycle focus between the first and last control.
  const trapTab = (e: KeyboardEvent<HTMLDialogElement>) => {
    const dialog = ref.current
    if (e.key !== 'Tab' || !dialog) return
    const items = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(el => el.getClientRects().length > 0)
    if (!items.length) return
    const first = items[0]
    const last = items[items.length - 1]
    const active = document.activeElement
    const outside = !active || !dialog.contains(active)
    if (e.shiftKey && (outside || active === first)) { e.preventDefault(); last.focus() }
    else if (!e.shiftKey && (outside || active === last)) { e.preventDefault(); first.focus() }
  }
  return <dialog ref={ref} className={wide ? 'app-dialog wide-dialog' : 'app-dialog'} aria-labelledby={titleId} aria-describedby={bodyId} onKeyDown={trapTab} onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===ref.current)onClose()}}><div className="dialog-heading"><h2 id={titleId}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={17}/></button></div><div className="dialog-body" id={bodyId}>{children}</div></dialog>
}
