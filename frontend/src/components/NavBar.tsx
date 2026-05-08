'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { fetchHealth } from '@/lib/api'

const LINKS = [
  { href: '/', label: 'Dashboard', icon: '⬡' },
  { href: '/graph', label: 'Graph Explorer', icon: '🕸' },
  { href: '/query', label: 'Query', icon: '⚡' },
  { href: '/ingest', label: 'Ingest', icon: '📥' },
  { href: '/analytics', label: 'Analytics', icon: '📊' },
]

export default function NavBar() {
  const pathname = usePathname()
  const [health, setHealth] = useState<{ graph_nodes: number; graph_edges: number } | null>(null)

  useEffect(() => {
    const check = async () => {
      const h = await fetchHealth()
      setHealth(h)
    }
    check()
    const id = setInterval(check, 10000)
    return () => clearInterval(id)
  }, [])

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <span className="navbar-logo">GraphRAG++</span>

      <div className="navbar-links">
        {LINKS.map(({ href, label, icon }) => (
          <Link
            key={href}
            href={href}
            id={`nav-${label.toLowerCase().replace(/\s+/g, '-')}`}
            className={`nav-link${pathname === href ? ' active' : ''}`}
          >
            <span>{icon}</span>
            <span>{label}</span>
          </Link>
        ))}
      </div>

      <div className="navbar-status">
        {health ? (
          <>
            <span className="status-dot" />
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {health.graph_nodes}N · {health.graph_edges}E
            </span>
          </>
        ) : (
          <span style={{ color: 'var(--text-muted)' }}>connecting…</span>
        )}
      </div>
    </nav>
  )
}
