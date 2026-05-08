/**
 * GraphRAG++ API client
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: GraphStats
}

export interface GraphNode {
  id: string
  label: string
  type: string
  color: string
  pagerank: number
  community: number
  degree: number
  size: number
}

export interface GraphEdge {
  source: string
  target: string
  predicate: string
  confidence: number
  weight: number
}

export interface GraphStats {
  node_count: number
  edge_count: number
  density: number
  avg_degree: number
  version: number
  ingestion_count: number
  query_count: number
  entity_types: Record<string, number>
}

export interface QueryResult {
  question: string
  answer: string
  thinking?: string
  intent: { intent: string; entities: string[]; confidence: number }
  steps: PipelineStep[]
  paths: HopPath[][]
  total_ms: number
  source: string
  provenance: Provenance[]
}

export interface PipelineStep {
  step: string
  label: string
  icon: string
  result: Record<string, unknown>
  thinking?: string
  ms: number
}

export interface HopPath {
  from: { id: string; label: string; type: string }
  edge: { predicate: string; confidence: number }
  to: { id: string; label: string; type: string }
}

export interface Provenance {
  path_id: number
  entities: string[]
  hops: number
  confidence: number
}

export interface IngestResult {
  status: string
  triples_extracted: number
  nodes_added: number
  edges_added: number
  version: number
  triples: Array<{ subject: string; predicate: string; object: string; confidence: number }>
  source: string
}

// ─── API functions ───────────────────────────────────────────────────────────

export async function fetchGraph(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/graph`)
  if (!res.ok) throw new Error('Failed to fetch graph')
  return res.json()
}

export async function fetchStats(): Promise<GraphStats> {
  const res = await fetch(`${API_BASE}/graph/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`)
    return res.json()
  } catch {
    return null
  }
}

export async function runQuery(question: string, maxHops = 2): Promise<QueryResult> {
  // Local llama.cpp on iGPU is slow — give it up to 10 minutes before aborting,
  // and surface aborts/server-deaths as real errors instead of hanging forever.
  const ctrl = new AbortController()
  const timeoutMs = 10 * 60 * 1000
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, max_hops: maxHops }),
      signal: ctrl.signal,
    })
    if (!res.ok) throw new Error(`Query failed: ${res.status} ${res.statusText}`)
    return res.json()
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      throw new Error(`Query timed out after ${timeoutMs / 1000}s — backend may be hung or crashed.`)
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

export async function ingestText(text: string, source = 'manual'): Promise<IngestResult> {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source }),
  })
  if (!res.ok) throw new Error('Ingestion failed')
  return res.json()
}

export async function ingestFile(file: File): Promise<IngestResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/ingest/file`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('File ingestion failed')
  return res.json()
}

export async function reseedGraph(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/ingest/seed`, { method: 'POST' })
  if (!res.ok) throw new Error('Reseed failed')
  return res.json()
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/metrics`)
  if (!res.ok) throw new Error('Metrics failed')
  return res.json()
}

export interface WSHandle {
  close: () => void
}

/**
 * Creates a self-reconnecting WebSocket that survives backend restarts.
 * Returns a handle with a close() method to tear down cleanly.
 */
export function createWebSocket(onMessage: (data: unknown) => void): WSHandle {
  let ws: WebSocket | null = null
  let closed = false
  let retry = 0
  const maxRetry = 8
  const pingInterval = 25_000 // 25 s keepalive

  let pingTimer: ReturnType<typeof setInterval> | null = null

  function connect() {
    if (closed) return
    const url = `${API_BASE.replace(/^http/, 'ws')}/ws`
    ws = new WebSocket(url)

    ws.onopen = () => {
      retry = 0
      // keepalive ping
      pingTimer = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) ws.send('ping')
      }, pingInterval)
    }

    ws.onmessage = (e) => {
      if (e.data === 'pong') return // ignore server pongs
      try { onMessage(JSON.parse(e.data)) } catch { /* ignore parse errors */ }
    }

    ws.onerror = () => { /* handled via onclose */ }

    ws.onclose = () => {
      if (pingTimer) { clearInterval(pingTimer); pingTimer = null }
      if (closed || retry >= maxRetry) return
      // Exponential backoff: 500ms, 1s, 2s, 4s … capped at 16s
      const delay = Math.min(500 * Math.pow(2, retry), 16_000)
      retry++
      setTimeout(connect, delay)
    }
  }

  connect()

  return {
    close() {
      closed = true
      if (pingTimer) clearInterval(pingTimer)
      ws?.close()
    },
  }
}
