import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

export function Modal({open,title,onClose,children,wide=false}:{open:boolean;title:string;onClose:()=>void;children:React.ReactNode;wide?:boolean}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(()=>{const dialog=ref.current;if(!dialog)return;if(open&&!dialog.open){dialog.showModal();requestAnimationFrame(()=>dialog.querySelector<HTMLElement>('[data-autofocus]')?.focus())}else if(!open&&dialog.open)dialog.close()},[open])
  return <dialog ref={ref} className={wide ? 'app-dialog wide-dialog' : 'app-dialog'} aria-labelledby={`dialog-${title.replace(/[^a-z0-9]+/gi,'-')}`} onCancel={e=>{e.preventDefault();onClose()}} onClick={e=>{if(e.target===ref.current)onClose()}}><div className="dialog-heading"><h2 id={`dialog-${title.replace(/[^a-z0-9]+/gi,'-')}`}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={17}/></button></div><div className="dialog-body">{children}</div></dialog>
}
