import { useEffect, useId, useRef } from 'react'
import { X } from 'lucide-react'

export function Modal({open,title,onClose,children,wide=false}:{open:boolean;title:string;onClose:()=>void;children:React.ReactNode;wide?:boolean}) {
  const ref = useRef<HTMLDialogElement>(null)
  // Each instance gets its own ids, so two dialogs with the same title never collide.
  const id = useId()
  const titleId = `${id}-title`
  const bodyId = `${id}-body`
  useEffect(()=>{const dialog=ref.current;if(!dialog)return;if(open&&!dialog.open){dialog.showModal();requestAnimationFrame(()=>dialog.querySelector<HTMLElement>('[data-autofocus]')?.focus())}else if(!open&&dialog.open)dialog.close()},[open])
  return <dialog ref={ref} className={wide ? 'app-dialog wide-dialog' : 'app-dialog'} aria-labelledby={titleId} aria-describedby={bodyId} onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===ref.current)onClose()}}><div className="dialog-heading"><h2 id={titleId}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={17}/></button></div><div className="dialog-body" id={bodyId}>{children}</div></dialog>
}
