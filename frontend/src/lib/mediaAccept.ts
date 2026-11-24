export type MediaCaps = {
  accepted_extensions?: string[]
  accepts_any_ffmpeg_decodable_media?: boolean
}

const FALLBACK_EXTENSIONS = ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'opus', 'wma', 'mp4', 'webm', 'm4v', 'mov', 'mkv']

export function acceptedExtensions(caps?: MediaCaps | null): string[] {
  if (Array.isArray(caps?.accepted_extensions) && caps.accepted_extensions.length) return caps.accepted_extensions
  return FALLBACK_EXTENSIONS
}

export function fileAcceptAttribute(caps?: MediaCaps | null): string {
  const parts = ['audio/*', 'video/*', ...acceptedExtensions(caps).map(ext => `.${ext}`)]
  return parts.join(',')
}

export function isAcceptedMediaFile(file: File, caps?: MediaCaps | null): boolean {
  if (file.size === 0) return false
  const acceptsAny = caps?.accepts_any_ffmpeg_decodable_media !== false
  if (acceptsAny && (file.type.startsWith('audio/') || file.type.startsWith('video/'))) return true
  const escaped = acceptedExtensions(caps).map(ext => ext.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  return new RegExp(`\\.(${escaped})$`, 'i').test(file.name)
}