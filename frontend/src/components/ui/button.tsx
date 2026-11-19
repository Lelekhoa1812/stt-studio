import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
}

const variants = {
  primary: 'button-primary',
  secondary: 'button-secondary',
  danger: 'button-danger',
  ghost: 'icon-button',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'secondary', type = 'button', ...props }, ref) => (
    <button ref={ref} type={type} className={cn(variants[variant], className)} {...props} />
  ),
)

Button.displayName = 'Button'
