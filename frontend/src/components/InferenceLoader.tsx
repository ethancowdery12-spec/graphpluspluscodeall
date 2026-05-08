'use client'
import { useEffect, useRef, useState } from 'react'

interface InferenceLoaderProps {
  active: boolean
  activeStep: number // 0-4 pipeline step index
  // Live token-stream stats from backend WS `generation_token` events.
  // All optional so older callers still work; default to zero/empty.
  tokens?: number
  tokensPerSec?: number
  preview?: string
}

const PHASES = [
  { label: 'Parsing intent', icon: '🧠', detail: 'Classifying query type & extracting entities…' },
  { label: 'Resolving entities', icon: '🔍', detail: 'Matching concepts to graph nodes via embeddings…' },
  { label: 'Traversing graph', icon: '🕸️', detail: 'Walking multi-hop paths through knowledge graph…' },
  { label: 'Fusing context', icon: '⚡', detail: 'Compressing evidence from paths into LLM context…' },
  { label: 'Generating answer', icon: '✨', detail: 'Streaming tokens from fine-tuned GraphRAG++ model…' },
]

export default function InferenceLoader({
  active,
  activeStep,
  tokens = 0,
  tokensPerSec = 0,
  preview = '',
}: InferenceLoaderProps) {
  const [elapsed, setElapsed] = useState(0)        // real seconds since query started
  const [dots, setDots] = useState('.')
  const [glitch, setGlitch] = useState(false)
  const startRef = useRef<number>(0)
  const clockRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const dotsRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const glitchRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!active) {
      setElapsed(0)
      if (clockRef.current) clearInterval(clockRef.current)
      if (dotsRef.current) clearInterval(dotsRef.current)
      return
    }

    // Real elapsed-time clock — no fake token counter
    startRef.current = Date.now()
    clockRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000))
    }, 250)

    // Animated dots
    dotsRef.current = setInterval(() => {
      setDots(d => (d.length >= 3 ? '.' : d + '.'))
    }, 400)

    // Glitch on phase change
    setGlitch(true)
    glitchRef.current = setTimeout(() => setGlitch(false), 300)

    return () => {
      if (clockRef.current) clearInterval(clockRef.current)
      if (dotsRef.current) clearInterval(dotsRef.current)
      if (glitchRef.current) clearTimeout(glitchRef.current)
    }
  }, [active, activeStep])

  if (!active) return null

  const phase = PHASES[Math.min(activeStep, PHASES.length - 1)]

  return (
    <div className="inference-loader" aria-live="polite" aria-label="Inference in progress">
      {/* Animated scanner bar */}
      <div className="scanner-bar" />

      {/* Main status */}
      <div className={`loader-phase ${glitch ? 'glitch' : ''}`}>
        <span className="loader-phase-icon">{phase.icon}</span>
        <div className="loader-phase-text">
          <span className="loader-phase-label">{phase.label}{dots}</span>
          <span className="loader-phase-detail">{phase.detail}</span>
        </div>
      </div>

      {/* Elapsed-time indicator — honest, no fake token counter */}
      <div className="loader-tokens">
        <div className="loader-token-bar">
          <div
            className={`loader-token-fill ${activeStep >= 4 ? 'fast' : ''}`}
            // Logarithmic fill: ~50% at 30s, ~80% at 120s, asymptotes to 99%
            style={{ width: `${Math.min(99, 99 * (1 - Math.exp(-elapsed / 60)))}%` }}
          />
        </div>
        <div className="loader-token-stats">
          <span className="loader-token-count">
            <span className="loader-token-num">{elapsed}s</span>
            <span className="loader-token-label">{elapsed > 90 ? ' — local LLM is slow on iGPU, hang tight' : ' elapsed'}</span>
          </span>
          <span className="loader-token-phase">
            Step {Math.min(activeStep + 1, 5)}/5
          </span>
        </div>
      </div>

      {/* Live generation stats — only shown once tokens start arriving from
          the backend's `generation_token` WS events. */}
      {tokens > 0 && (
        <div className="loader-gen-stats">
          <div className="loader-gen-stats-row">
            <span>
              <span className="loader-token-num">{tokens}</span>
              <span className="loader-token-label"> tokens</span>
            </span>
            <span>
              <span className="loader-token-num">{tokensPerSec.toFixed(1)}</span>
              <span className="loader-token-label"> tok/s</span>
            </span>
          </div>
          {preview && (
            <div className="loader-gen-preview" title="First tokens of the answer">
              {preview}
              <span className="loader-gen-preview-cursor">▍</span>
            </div>
          )}
        </div>
      )}

      {/* Step dots */}
      <div className="loader-steps">
        {PHASES.map((p, i) => (
          <div
            key={i}
            className={[
              'loader-step-dot',
              i < activeStep ? 'done' : '',
              i === activeStep ? 'active' : '',
            ].join(' ')}
            title={p.label}
          />
        ))}
      </div>
    </div>
  )
}
