'use client'
import { useEffect, useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { fetchGraph, fetchStats } from '@/lib/api'
import type { GraphData, GraphNode, GraphStats } from '@/lib/api'

const GraphViewer = dynamic(() => import('@/components/GraphViewer'), { ssr: false })

export default function GraphPage() {
  const [data, setData] = useState<GraphData | null>(null)
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [filter, setFilter] = useState<string>('all')
  const [search, setSearch] = useState('')

  const [loadError, setLoadError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const [g, s] = await Promise.all([fetchGraph(), fetchStats()])
      setData(g)
      setStats(s)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : 'Failed to load graph — is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const filteredData = data ? {
    ...data,
    nodes: data.nodes.filter(n => {
      const matchType = filter === 'all' || n.type === filter
      const matchSearch = !search || n.label.toLowerCase().includes(search.toLowerCase())
      return matchType && matchSearch
    }),
    edges: data.edges.filter(e => {
      const srcNode = data.nodes.find(n => n.id === e.source)
      const tgtNode = data.nodes.find(n => n.id === e.target)
      const matchType = filter === 'all' || srcNode?.type === filter || tgtNode?.type === filter
      const matchSearch = !search || srcNode?.label.toLowerCase().includes(search.toLowerCase()) || tgtNode?.label.toLowerCase().includes(search.toLowerCase())
      return matchType && matchSearch
    }),
  } : null

  const entityTypes = stats?.entity_types ? Object.keys(stats.entity_types) : []

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - var(--nav-h))', overflow: 'hidden' }}>
      {/* Sidebar */}
      <div style={{
        width: 280, flexShrink: 0, background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--border)' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem' }}>🕸 Graph Explorer</h2>
          <input
            id="graph-search"
            className="input"
            placeholder="Search entities…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ padding: '0.5rem 0.75rem', fontSize: '0.82rem' }}
          />
        </div>

        {/* Filters */}
        <div style={{ padding: '1rem', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Filter by Type</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
            <button
              id="filter-all"
              className={`tag tag-unknown`}
              style={{ cursor: 'pointer', border: filter === 'all' ? '1px solid var(--cyan)' : '' }}
              onClick={() => setFilter('all')}
            >all · {stats?.node_count ?? 0}</button>
            {entityTypes.map(type => (
              <button
                key={type}
                id={`filter-${type}`}
                className={`tag tag-${type}`}
                style={{ cursor: 'pointer', border: filter === type ? '1px solid var(--cyan)' : '' }}
                onClick={() => setFilter(type)}
              >
                {type} · {stats?.entity_types[type]}
              </button>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div style={{ padding: '1rem', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            {[
              { label: 'Nodes', value: filteredData?.nodes.length ?? 0 },
              { label: 'Edges', value: filteredData?.edges.length ?? 0 },
              { label: 'Avg Degree', value: stats?.avg_degree ?? 0 },
              { label: 'Density', value: stats ? (stats.density * 100).toFixed(3) + '%' : '0%' },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, padding: '0.625rem' }}>
                <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, background: 'var(--grad-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Node Inspector */}
        <div style={{ flex: 1, overflow: 'auto', padding: '1rem' }}>
          {selectedNode ? (
            <div>
              <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Node Inspector</div>
              <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-accent)', borderRadius: 8, padding: '1rem' }}>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: selectedNode.color, marginBottom: '0.5rem', wordBreak: 'break-word' }}>{selectedNode.label}</div>
                <span className={`tag tag-${selectedNode.type}`} style={{ marginBottom: '0.75rem', display: 'inline-flex' }}>{selectedNode.type}</span>
                <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  {[
                    ['ID', selectedNode.id.slice(0, 12) + '…'],
                    ['Degree', selectedNode.degree],
                    ['PageRank', selectedNode.pagerank.toFixed(5)],
                    ['Community', selectedNode.community],
                    ['Size', selectedNode.size.toFixed(1)],
                  ].map(([k, v]) => (
                    <div key={String(k)} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                      <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', textAlign: 'center', padding: '2rem 1rem' }}>
              Click any node to inspect its properties
            </div>
          )}
        </div>

        {/* Refresh */}
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border)' }}>
          <button id="graph-refresh-btn" className="btn btn-secondary" style={{ width: '100%', justifyContent: 'center', fontSize: '0.82rem' }} onClick={load}>
            ↺ Refresh Graph
          </button>
        </div>
      </div>

      {/* Main graph canvas */}
      <div style={{ flex: 1, position: 'relative', background: 'var(--bg-base)' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
            <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
            <p style={{ color: 'var(--text-muted)' }}>Loading knowledge graph…</p>
          </div>
        ) : loadError ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem', padding: '2rem', textAlign: 'center' }}>
            <div style={{ fontSize: '2rem' }}>⚠️</div>
            <p style={{ color: 'var(--text-muted)', maxWidth: 400 }}>{loadError}</p>
            <button className="btn btn-secondary" onClick={load} style={{ marginTop: '0.5rem' }}>↺ Retry</button>
          </div>
        ) : filteredData ? (
          <GraphViewer
            data={filteredData}
            onNodeClick={setSelectedNode}
          />
        ) : null}

        {/* Legend */}
        <div style={{
          position: 'absolute', bottom: 16, right: 16,
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '0.75rem 1rem', fontSize: '0.72rem',
          backdropFilter: 'blur(10px)',
        }}>
          <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Legend</div>
          {[
            ['concept', '#00d4ff'], ['model', '#a855f7'], ['person', '#f59e0b'],
            ['organization', '#10b981'], ['dataset', '#f43f5e'], ['codebase', '#fb923c'],
            ['technique', '#6366f1'], ['paper', '#84cc16'],
          ].map(([type, color]) => (
            <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
              <span style={{ color: 'var(--text-muted)', textTransform: 'capitalize' }}>{type}</span>
            </div>
          ))}
          <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.68rem' }}>
            Scroll to zoom · Drag to pan · Click node to inspect
          </div>
        </div>
      </div>
    </div>
  )
}
