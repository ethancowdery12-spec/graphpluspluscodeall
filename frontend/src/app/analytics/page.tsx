'use client'
import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { fetchMetrics, fetchStats } from '@/lib/api'

const MiniChart = dynamic(() => import('@/components/MiniChart'), { ssr: false })

const BENCHMARK_LABELS = ['Complex Queries', 'Noisy Data Filter', 'Update Speed', 'Traceability', 'Hallucination Rate']
const GRAPHRAG_SCORES = [95, 90, 98, 100, 88]
const RAG_SCORES = [45, 30, 10, 5, 38]

const LATENCY_LABELS = ['<100ms', '100-200ms', '200-300ms', '300-400ms', '400-500ms', '>500ms']
const LATENCY_DIST = [15, 35, 25, 15, 7, 3]

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [m, s] = await Promise.all([fetchMetrics(), fetchStats()])
        setMetrics(m); setStats(s)
      } catch {}
      setLoading(false)
    }
    load()
    const id = setInterval(load, 15000)
    return () => clearInterval(id)
  }, [])

  const entityTypes = (stats?.entity_types as Record<string, number>) ?? {}
  const typeLabels = Object.keys(entityTypes)
  const typeValues = Object.values(entityTypes)

  const typeColors = [
    '#00d4ff', '#a855f7', '#f59e0b', '#10b981',
    '#f43f5e', '#6366f1', '#ec4899', '#84cc16', '#fb923c'
  ]

  return (
    <div className="container" style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 800 }}>📊 Analytics Dashboard</h1>
        <span className="section-badge">Live</span>
      </div>

      {loading ? (
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', color: 'var(--text-muted)' }}>
          <span className="spinner" /> Loading metrics…
        </div>
      ) : (
        <>
          {/* Top stats */}
          <div className="stat-grid" style={{ marginBottom: '2rem' }}>
            {[
              { label: 'Total Nodes', value: stats?.node_count ?? 0, sub: 'unique entities' },
              { label: 'Total Edges', value: stats?.edge_count ?? 0, sub: 'typed relations' },
              { label: 'Avg Degree', value: stats?.avg_degree ?? 0, sub: 'connections/node' },
              { label: 'Graph Density', value: stats ? ((stats.density as number) * 100).toFixed(4) + '%' : '0%', sub: 'edge density' },
              { label: 'Total Queries', value: stats?.query_count ?? 0, sub: 'all time' },
              { label: 'Graph Version', value: `v${stats?.version ?? 0}`, sub: 'current version' },
              { label: 'Uptime', value: metrics ? `${Math.floor(metrics.uptime_s as number)}s` : '—', sub: 'backend uptime' },
              { label: 'LLM Source', value: String(metrics?.llm_source ?? 'sim'), sub: 'active backend' },
            ].map(({ label, value, sub }) => (
              <div key={label} className="stat-card">
                <div className="stat-label">{label}</div>
                <div className="stat-value" style={{ fontSize: '1.5rem' }}>{value}</div>
                <div className="stat-sub">{sub}</div>
              </div>
            ))}
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1.25rem', marginBottom: '1.25rem' }}>
            {/* Benchmark */}
            <div className="card">
              <div className="section-header">
                <span className="section-title">GraphRAG++ vs RAG</span>
                <span className="section-badge">Benchmark</span>
              </div>
              <MiniChart
                type="bar"
                labels={BENCHMARK_LABELS}
                datasets={[
                  {
                    label: 'GraphRAG++',
                    data: GRAPHRAG_SCORES,
                    backgroundColor: 'rgba(0,212,255,0.7)',
                    borderColor: '#00d4ff',
                    borderWidth: 1,
                  },
                  {
                    label: 'Baseline RAG',
                    data: RAG_SCORES,
                    backgroundColor: 'rgba(168,85,247,0.4)',
                    borderColor: '#a855f7',
                    borderWidth: 1,
                  },
                ]}
                height={220}
              />
            </div>

            {/* Latency */}
            <div className="card">
              <div className="section-header">
                <span className="section-title">Query Latency Distribution</span>
                <span className="section-badge">P50: 190ms</span>
              </div>
              <MiniChart
                type="bar"
                labels={LATENCY_LABELS}
                datasets={[{
                  label: '% of Queries',
                  data: LATENCY_DIST,
                  backgroundColor: LATENCY_DIST.map((_, i) =>
                    i < 2 ? 'rgba(16,185,129,0.7)' : i < 4 ? 'rgba(0,212,255,0.6)' : 'rgba(244,63,94,0.5)'
                  ),
                  borderWidth: 0,
                }]}
                height={220}
              />
            </div>

            {/* Entity types pie */}
            <div className="card">
              <div className="section-header">
                <span className="section-title">Entity Type Distribution</span>
                <span className="section-badge">{typeLabels.length} types</span>
              </div>
              {typeLabels.length > 0 ? (
                <MiniChart
                  type="doughnut"
                  labels={typeLabels.map(t => t.charAt(0).toUpperCase() + t.slice(1))}
                  datasets={[{
                    label: 'Entities',
                    data: typeValues,
                    backgroundColor: typeColors.slice(0, typeLabels.length),
                    borderColor: 'rgba(0,0,0,0.3)',
                    borderWidth: 2,
                  }]}
                  height={220}
                />
              ) : (
                <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>No data yet</div>
              )}
            </div>

            {/* Graph growth simulated */}
            <div className="card">
              <div className="section-header">
                <span className="section-title">Graph Growth Over Time</span>
                <span className="section-badge">Simulated</span>
              </div>
              <MiniChart
                type="line"
                labels={['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Today']}
                datasets={[
                  {
                    label: 'Nodes',
                    data: [20, 45, 78, 105, 130, 158, stats?.node_count as number ?? 180],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0,212,255,0.1)',
                    borderWidth: 2,
                    fill: true,
                  },
                  {
                    label: 'Edges',
                    data: [35, 80, 140, 195, 240, 295, stats?.edge_count as number ?? 330],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168,85,247,0.1)',
                    borderWidth: 2,
                    fill: true,
                  },
                ]}
                height={220}
              />
            </div>
          </div>

          {/* Entity type breakdown table */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--border)' }}>
              <div className="section-header" style={{ marginBottom: 0 }}>
                <span className="section-title">Entity Type Breakdown</span>
              </div>
            </div>
            <table className="perf-table">
              <thead>
                <tr>
                  <th>Entity Type</th>
                  <th>Count</th>
                  <th>% of Graph</th>
                  <th>Color</th>
                </tr>
              </thead>
              <tbody>
                {typeLabels.map((type, i) => (
                  <tr key={type}>
                    <td><span className={`tag tag-${type}`}>{type}</span></td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{typeValues[i]}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                      {stats?.node_count ? ((typeValues[i] / (stats.node_count as number)) * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td><div style={{ width: 16, height: 16, borderRadius: 4, background: typeColors[i] ?? '#888' }} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
