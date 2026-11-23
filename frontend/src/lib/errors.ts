const UNREACHABLE = 'Can’t reach the local transcription service. Check that the STT backend is running, then retry.'
const CANCELLED = 'The request was cancelled before it finished. Try again.'

// The one place a panel turns a caught error into text. A failed network call (TypeError, "Failed to fetch")
// and an aborted request (AbortError) get readable wording instead of browser messages.
export function readableError(error: unknown, fallback = 'Something went wrong.'): string {
  if (error instanceof TypeError) return UNREACHABLE
  if (error instanceof DOMException && error.name === 'AbortError') return CANCELLED
  if (error instanceof Error) return error.message || fallback
  return fallback
}
