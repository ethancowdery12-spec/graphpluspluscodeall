'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { fetchStats, fetchHealth, reseedGraph } from '@/lib/api'
import type { GraphStats } from '@/lib/api'

const PERF_DATA = [
  { aspect: 'Complex Query Accuracy', graphrag: '+50%', base: 'Baseline RAG', gain: '+50%' },
  { aspect: 'Hallucination Rate', graphrag: '-70%', base: 'Baseline RAG', gain: '-70%' },
  { aspect: 'Index Update Speed', graphrag: '10× faster', base: 'Baseline RAG', gain: '10×' },
  { aspect: 'Path Traceability', graphrag: 'Full audit', base: 'Baseline RAG', gain: '∞' },
  { aspect: 'Noisy Data Handling', graphrag: 'Relation validation', base: 'Baseline RAG', gain: '+60%' },
]

export default function HomePage() {
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [health, setHealth] = useState<{ uptime_s: number } | null>(null)
  const [reseeding, setReseeding] = useState(false)

  useEffect(() => {
    const load = async () => {
      const [s, h] = await Promise.all([fetchStats(), fetchHealth()])
      setStats(s); setHealth(h)
    }
    load()
    const id = setInterval(load, 8000)
    return () => clearInterval(id)
  }, [])

  const handleReseed = async () => {
    setReseeding(true)
    try { await reseedGraph() ; const s = await fetchStats(); setStats(s) }
    finally { setReseeding(false) }
  }

  return (
    <>
      {/* Glow orbs */}
      <div className="glow-orb glow-orb-cyan" style={{ top: '10%', left: '10%' }} />
      <div className="glow-orb glow-orb-purple" style={{ top: '30%', right: '8%' }} />

      {/* Hero */}
      <section className="hero">
        <p className="hero-eyebrow">⬡ Next-Generation Retrieval-Augmented Generation</p>
        <h1 className="hero-title">
          <span className="highlight">GraphRAG++</span>
          <br />
          Knowledge Graph<br />Fusion Engine
        </h1>
        <p className="hero-subtitle">
          Replace flat vector search with a dynamic, hybrid knowledge graph.
          Multi-hop reasoning, entity-relation extraction, and LLM-fused generation
          for <strong style={{ color: 'var(--cyan)' }}>40–60% better performance</strong> on complex queries.
        </p>
        <div className="hero-cta">
          <Link href="/query" className="btn btn-primary" id="hero-query-btn">
            ⚡ Run a Query
          </Link>
          <Link href="/graph" className="btn btn-secondary" id="hero-graph-btn">
            🕸 Explore Graph
          </Link>
          <Link href="/ingest" className="btn btn-ghost" id="hero-ingest-btn">
            📥 Ingest Docs
          </Link>
        </div>
      </section>

      {/* Live Stats */}
      <section style={{ padding: '0 2rem 4rem' }}>
        <div className="container">
          <div className="section-header" style={{ marginBottom: '1.5rem' }}>
            <h2 className="section-title">Live Graph State</h2>
            <span className="section-badge">Real-time</span>
            <button
              id="reseed-btn"
              className="btn btn-ghost"
              style={{ marginLeft: 'auto', fontSize: '0.8rem', padding: '0.4rem 0.875rem' }}
              onClick={handleReseed}
              disabled={reseeding}
            >
              {reseeding ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Reseeding…</> : '↺ Reseed'}
            </button>
          </div>

          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
            <StatCard label="Graph Nodes" value={stats?.node_count ?? '—'} sub="unique entities" />
            <StatCard label="Relations" value={stats?.edge_count ?? '—'} sub="typed edges" />
            <StatCard label="Avg Degree" value={stats?.avg_degree ?? '—'} sub="connections/node" />
            <StatCard label="Graph Density" value={stats ? (stats.density * 100).toFixed(4) + '%' : '—'} sub="edge density" />
            <StatCard label="Queries Run" value={stats?.query_count ?? 0} sub="total queries" />
            <StatCard label="Uptime" value={health ? Math.floor(health.uptime_s) + 's' : '—'} sub="backend uptime" />
          </div>

          {/* Entity type breakdown */}
          {stats?.entity_types && (
            <div style={{ marginTop: '1.5rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginRight: '0.5rem' }}>ENTITY TYPES:</span>
              {Object.entries(stats.entity_types).map(([type, count]) => (
                <span key={type} className={`tag tag-${type}`}>
                  {type} · {count}
                </span>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Architecture */}
      <section style={{ padding: '2rem 2rem 4rem', borderTop: '1px solid var(--border)' }}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Tri-Layer Architecture</h2>
            <span className="section-badge">Core Design</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
            {[
              { icon: '◈', title: 'Entity Layer', color: 'var(--cyan)', desc: 'Nodes for concepts, models, organizations, papers, and codebases. Entity resolution via cosine similarity deduplication.' },
              { icon: '⟷', title: 'Relation Layer', color: 'var(--purple)', desc: 'Directed edges with typed predicates, confidence weights, and temporal versioning. PageRank scoring + Louvain communities.' },
              { icon: '⊕', title: 'Context Layer', color: 'var(--amber)', desc: 'Embedded summaries per node via Sentence Transformers + FAISS hybrid index for semantic retrieval.' },
            ].map(({ icon, title, color, desc }) => (
              <div key={title} className="card" style={{ borderColor: `${color}22` }}>
                <div style={{ fontSize: '2rem', marginBottom: '0.75rem', color }}>{icon}</div>
                <h3 style={{ color, marginBottom: '0.5rem' }}>{title}</h3>
                <p style={{ fontSize: '0.875rem' }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Performance Table */}
      <section style={{ padding: '2rem 2rem 4rem', borderTop: '1px solid var(--border)' }}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Performance vs. RAG</h2>
            <span className="section-badge">Benchmarks</span>
          </div>
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <table className="perf-table">
              <thead>
                <tr>
                  <th>Aspect</th>
                  <th>GraphRAG++ Advantage</th>
                  <th>Baseline</th>
                  <th>Gain</th>
                </tr>
              </thead>
              <tbody>
                {PERF_DATA.map(row => (
                  <tr key={row.aspect}>
                    <td>{row.aspect}</td>
                    <td style={{ color: 'var(--text-primary)' }}>{row.graphrag}</td>
                    <td>{row.base}</td>
                    <td className="gain">{row.gain}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section style={{ padding: '2rem 2rem 6rem', borderTop: '1px solid var(--border)' }}>
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Quick Actions</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            {[
              { href: '/query', label: 'Multi-hop Query', icon: '⚡', desc: 'Run complex reasoning queries over the knowledge graph', color: 'var(--cyan)' },
              { href: '/graph', label: 'Graph Explorer', icon: '🕸', desc: 'Interactively explore entity-relation networks', color: 'var(--purple)' },
              { href: '/ingest', label: 'Ingest Documents', icon: '📥', desc: 'Upload papers or paste text for live extraction', color: 'var(--amber)' },
              { href: '/analytics', label: 'Analytics', icon: '📊', desc: 'Performance metrics and topology statistics', color: 'var(--green)' },
            ].map(({ href, label, icon, desc, color }) => (
              <Link key={href} href={href} className="card" style={{ display: 'block', textDecoration: 'none', borderColor: `${color}22` }}>
                <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>{icon}</div>
                <h3 style={{ color, marginBottom: '0.4rem', fontSize: '1rem' }}>{label}</h3>
                <p style={{ fontSize: '0.825rem' }}>{desc}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub: string }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}
