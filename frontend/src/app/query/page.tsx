'use client'
import { useState, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { runQuery, fetchGraph, createWebSocket } from '@/lib/api'
import type { QueryResult, GraphData, WSHandle } from '@/lib/api'
import InferenceLoader from '@/components/InferenceLoader'

const GraphViewer = dynamic(() => import('@/components/GraphViewer'), { ssr: false })

const EXAMPLE_QUERIES = [
  'How does the Transformer architecture relate to BERT and GPT-4?',
  'What techniques enable efficient LLM fine-tuning?',
  'How is RAG different from GraphRAG and what improvements does it offer?',
  'What frameworks are used for LLM deployment and serving?',
  'Explain the connection between LoRA and QLoRA for parameter-efficient training',
]

const INTENT_COLORS: Record<string, string> = {
  'multi-hop': 'var(--purple)',
  'factual': 'var(--cyan)',
  'aggregative': 'var(--amber)',
  'comparative': 'var(--green)',
}

export default function QueryPage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [showGraph, setShowGraph] = useState(false)
  const [graphLoading, setGraphLoading] = useState(false)
  const [backendUp, setBackendUp] = useState<boolean | null>(null)   // null = checking
  const [activeStep, setActiveStep] = useState<number>(0)
  // Live token-stream stats from the backend's `generation_token` WS events
  const [tokenStats, setTokenStats] = useState<{
    tokens: number; tokensPerSec: number; preview: string
  }>({ tokens: 0, tokensPerSec: 0, preview: '' })
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const wsRef = useRef<WSHandle | null>(null)
  // Track whether we've received at least one real WS step event this query
  const wsActiveRef = useRef(false)

  // ── WebSocket: real-time step events from the backend ────────────────────
  useEffect(() => {
    try {
      wsRef.current = createWebSocket((data: unknown) => {
        const d = data as {
          event?: string
          data?: {
            step?: string
            tokens?: number
            tokens_per_sec?: number
            preview?: string
          }
        }
        if (d.event === 'pipeline_step' && d.data?.step) {
          const STEP_ORDER: Record<string, number> = {
            intent_classification: 0,
            node_matching:         1,
            passage_retrieval:     2,
            graph_traversal:       3,
            context_fusion:        4,
            generation:            5,
          }
          const idx = STEP_ORDER[d.data.step]
          if (idx !== undefined) {
            wsActiveRef.current = true
            // Only advance — never jump backward
            setActiveStep(prev => Math.max(prev, idx))
          }
        } else if (d.event === 'generation_token' && d.data) {
          setTokenStats({
            tokens: d.data.tokens ?? 0,
            tokensPerSec: d.data.tokens_per_sec ?? 0,
            preview: d.data.preview ?? '',
          })
        }
      })
    } catch { /* ws unavailable in SSR or blocked */ }
    return () => {
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [])

  // ── Timer fallback: smoothly advance steps when WS isn't available ───────
  useEffect(() => {
    if (!loading) return
    // Reset tracking each new query
    wsActiveRef.current = false
    setActiveStep(0)

    // Estimated step durations (ms). Generation is open-ended.
    const durations = [950, 750, 300, 850, 650, 99_999]
    const timers: ReturnType<typeof setTimeout>[] = []
    let cumulative = 0

    for (let i = 0; i < durations.length - 1; i++) {
      cumulative += durations[i]
      const targetStep = i + 1
      timers.push(setTimeout(() => {
        // Skip timer if WS is already driving steps — avoids fighting
        if (!wsActiveRef.current) {
          setActiveStep(prev => Math.max(prev, targetStep))
        }
      }, cumulative))
    }

    return () => timers.forEach(clearTimeout)
  }, [loading])

  // Health-check on mount (no graph fetch — opt-in only)
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(r => setBackendUp(r.ok))
      .catch(() => setBackendUp(false))
  }, [])

  // Fetch graph when user opts in, or refresh after query completes if already shown
  useEffect(() => {
    if (!showGraph) return
    setGraphLoading(true)
    fetchGraph()
      .then(d => { setGraphData(d); setGraphLoading(false) })
      .catch(() => setGraphLoading(false))
  }, [showGraph])

  const submit = async (q = question) => {
    if (!q.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    setTokenStats({ tokens: 0, tokensPerSec: 0, preview: '' })

    // Refresh graph for path highlighting only if user has opted in
    if (showGraph) fetchGraph().then(setGraphData).catch(() => {/* non-fatal */})

    try {
      const r = await runQuery(q)
      setResult(r)
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e)
      const isDown = raw.toLowerCase().includes('failed to fetch') || raw.includes('ECONNREFUSED')
      setError(isDown
        ? '🔴 Backend not reachable at http://localhost:8000 — run: cd backend && uvicorn main:app --host 0.0.0.0 --port 8000'
        : raw
      )
    } finally {
      setLoading(false)
    }
  }

  // Extract highlighted node IDs from paths
  const highlightPaths = result?.paths.map(path =>
    path.flatMap(hop => [hop.from.id, hop.to.id])
  ) ?? []

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - var(--nav-h))', overflow: 'hidden' }}>
      {/* ── Left Panel ───────────────────────────────────────────── */}
      <div style={{
        width: 480, flexShrink: 0, display: 'flex', flexDirection: 'column',
        borderRight: '1px solid var(--border)', overflow: 'hidden',
        background: 'var(--bg-surface)',
      }}>
        {/* Input */}
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ background: 'var(--grad-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>⚡ Multi-hop Query Engine</span>
            <span style={{
              marginLeft: 'auto', fontSize: '0.7rem', fontWeight: 500,
              color: backendUp === null ? 'var(--text-muted)' : backendUp ? '#34d399' : '#f87171',
              fontFamily: 'var(--font-mono)',
            }}>
              {backendUp === null ? '○ checking…' : backendUp ? '● backend online' : '● backend offline'}
            </span>
          </h2>
          {backendUp === false && (
            <div style={{
              background: '#f8717122', border: '1px solid #f8717144', borderRadius: 8,
              padding: '0.6rem 0.875rem', fontSize: '0.78rem', color: '#fca5a5', marginBottom: '0.75rem',
            }}>
              Backend not running. Start it with:<br />
              <code style={{ fontFamily: 'var(--font-mono)', color: '#f87171' }}>
                cd backend &amp;&amp; uvicorn main:app --host 0.0.0.0 --port 8000
              </code>
            </div>
          )}
          <textarea
            ref={textareaRef}
            id="query-input"
            className="input"
            placeholder="Ask a complex question about AI/ML concepts, models, codebases…"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit() }}
            style={{ minHeight: 100, marginBottom: '0.75rem', resize: 'none' }}
          />
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              id="run-query-btn"
              className="btn btn-primary"
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={() => submit()}
              disabled={loading || !question.trim()}
            >
              {loading
                ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Reasoning…</>
                : '⚡ Run Query'}
            </button>
          </div>

          {/* Example queries */}
          <div style={{ marginTop: '0.875rem' }}>
            <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Examples</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  id={`example-query-${i}`}
                  onClick={() => { setQuestion(q); submit(q) }}
                  className="example-query-btn"
                >
                  <span className="example-query-arrow">›</span> {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline Trace / Loader */}
        <div style={{ flex: 1, overflow: 'auto', padding: '1rem' }}>
          {error && (
            <div className="error-banner">
              <span>⚠</span> {error}
            </div>
          )}

          {/* Dynamic inference loader replaces the boring spinner */}
          <InferenceLoader
            active={loading}
            activeStep={activeStep}
            tokens={tokenStats.tokens}
            tokensPerSec={tokenStats.tokensPerSec}
            preview={tokenStats.preview}
          />

          {(result) && (
            <>
              <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Pipeline Trace</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {result.steps.map((step, i) => (
                  <div key={step.step} className="pipeline-step done">
                    <div className="step-icon">{step.icon}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="step-label">{step.label}</span>
                        <span className="step-ms">{step.ms}ms</span>
                        <span className="step-check">✓</span>
                      </div>
                      {result.steps[i] && (
                        <div className="step-details">
                          {Object.entries(result.steps[i].result)
                            .filter(([k]) => !['start_nodes', 'paths'].includes(k))
                            .slice(0, 3)
                            .map(([k, v]) => `${k}: ${typeof v === 'number' ? v : JSON.stringify(v).slice(0, 30)}`).join(' · ')}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <MetaPanel result={result} />
            </>
          )}

          {!loading && !result && (
            <div className="empty-state">
              <div className="empty-state-icon">⚡</div>
              <div className="empty-state-title">Ready to reason</div>
              <div className="empty-state-sub">Enter a question to start multi-hop graph traversal</div>
            </div>
          )}
        </div>
      </div>

      {/* ── Right Panel ──────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Answer */}
        {result && (
          <div className="answer-panel">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '0.9rem', fontWeight: 700 }}>✨ Answer</h3>
              <span className="intent-badge" style={{
                background: `${INTENT_COLORS[result.intent.intent] ?? 'var(--cyan)'}22`,
                color: INTENT_COLORS[result.intent.intent] ?? 'var(--cyan)',
                border: `1px solid ${INTENT_COLORS[result.intent.intent] ?? 'var(--cyan)'}44`,
              }}>
                {result.intent.intent} query
              </span>
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {result.total_ms}ms · {result.source}
              </span>
            </div>

            {result.thinking && (
              <details style={{ marginBottom: '0.75rem' }}>
                <summary style={{ cursor: 'pointer', fontSize: '0.78rem', color: 'var(--purple)', marginBottom: '0.5rem' }}>🧠 Thinking process</summary>
                <div className="thinking-block">{result.thinking}</div>
              </details>
            )}

            <div className="answer-block">{result.answer}</div>

            {/* Retrieved passages */}
            {result.passages?.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  📄 Retrieved Passages ({result.passages.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  {result.passages.map((p, i) => (
                    <details key={i} style={{ background: 'var(--bg-base)', borderRadius: 6, border: '1px solid var(--border)' }}>
                      <summary style={{
                        cursor: 'pointer', padding: '0.4rem 0.6rem',
                        fontSize: '0.72rem', color: 'var(--text-muted)',
                        display: 'flex', alignItems: 'center', gap: '0.5rem',
                      }}>
                        <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
                          p.{p.page}
                        </span>
                        <span style={{ color: 'var(--green)', fontFamily: 'var(--font-mono)' }}>
                          {(p.score * 100).toFixed(0)}%
                        </span>
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {p.text.slice(0, 60).trim()}…
                        </span>
                      </summary>
                      <div style={{
                        padding: '0.5rem 0.6rem', fontSize: '0.75rem',
                        color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
                        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                        borderTop: '1px solid var(--border)',
                      }}>
                        {p.text}
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            )}

            {/* Path provenance — only show high-confidence, clean paths */}
            {result.provenance.filter(p => p.confidence >= 0.70).length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Path Provenance</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  {result.provenance.filter(p => p.confidence >= 0.70).map(p => (
                    <div key={p.path_id} className="path-item">
                      <span style={{ color: 'var(--text-muted)' }}>Path {p.path_id} ({p.hops}h, {(p.confidence * 100).toFixed(0)}%): </span>
                      {p.entities.map((e, i) => (
                        <span key={i}>
                          <span className="hop-node">{e}</span>
                          {i < p.entities.length - 1 && <span className="hop-arrow"> → </span>}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Graph Visualization — opt-in */}
        <div style={{ flex: 1, position: 'relative', background: 'var(--bg-base)' }}>
          {/* Toggle button always visible */}
          <button
            onClick={() => setShowGraph(v => !v)}
            style={{
              position: 'absolute', top: 10, right: 10, zIndex: 10,
              padding: '0.3rem 0.75rem', borderRadius: 6, border: '1px solid var(--border-accent)',
              background: showGraph ? 'var(--cyan-dim)' : 'var(--bg-surface)',
              color: showGraph ? 'var(--cyan)' : 'var(--text-muted)',
              fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
              fontFamily: 'var(--font-sans)', transition: 'all 0.15s',
            }}
          >
            {showGraph ? '🕸️ Hide Graph' : '🕸️ Show Graph'}
          </button>

          {showGraph ? (
            <>
              <div className="graph-badge">
                {result
                  ? `Highlighted: ${highlightPaths.flat().length} nodes across ${result.paths.length} paths`
                  : graphData
                  ? `${graphData.nodes.length} nodes · ${graphData.edges.length} edges`
                  : 'Loading graph…'}
              </div>
              {graphLoading || !graphData ? (
                <div className="graph-loading">
                  <div className="spinner" style={{ width: 32, height: 32, borderWidth: 2.5 }} />
                  <span>Loading knowledge graph…</span>
                </div>
              ) : (
                <GraphViewer data={graphData} highlightPaths={highlightPaths} />
              )}
            </>
          ) : (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', height: '100%', gap: '0.75rem',
              color: 'var(--text-muted)',
            }}>
              <span style={{ fontSize: '2rem' }}>🕸️</span>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Graph hidden</span>
              <span style={{ fontSize: '0.75rem', textAlign: 'center', maxWidth: 260, lineHeight: 1.5 }}>
                Toggle "Show Graph" to visualise the knowledge graph and query paths. Hidden by default to save GPU compute.
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function MetaPanel({ result }: { result: QueryResult }) {
  return (
    <div className="meta-panel">
      <div className="meta-panel-title">Query Metadata</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem', fontSize: '0.78rem' }}>
        {[
          ['Total Time', `${result.total_ms}ms`],
          ['LLM Source', result.source],
          ['Intent', result.intent.intent],
          ['Confidence', `${(result.intent.confidence * 100).toFixed(0)}%`],
          ['Paths Found', result.paths.length],
          ['Passages', (result.passages?.length ?? 0) + ' chunks'],
          ['Provenance', result.provenance.length + ' paths'],
        ].map(([k, v]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0', borderBottom: '1px solid var(--border)' }}>
            <span style={{ color: 'var(--text-muted)' }}>{k}</span>
            <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
