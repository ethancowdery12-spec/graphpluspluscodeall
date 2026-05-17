'use client'
import { useState, useRef, useCallback, DragEvent } from 'react'
import { ingestText, ingestFiles, ingestDirectory } from '@/lib/api'
import type { IngestResult, IngestDirResult } from '@/lib/api'

type Mode = 'file' | 'text' | 'directory'

interface Props {
  onIngestComplete: () => void
}

const TIER_COLORS: Record<string, string> = {
  EXTRACTED: '#34d399',
  INFERRED:  '#60a5fa',
  AMBIGUOUS: '#f59e0b',
}

const ACCEPT = [
  // code
  '.py','.js','.jsx','.ts','.tsx','.go','.rs','.java','.c','.cpp','.cc','.rb',
  // docs
  '.md','.txt','.rst','.pdf',
  // archives
  '.zip',
  // data
  '.json','.yaml','.yml','.toml','.csv',
  // images
  '.png','.jpg','.jpeg','.webp','.gif',
].join(',')

export default function IngestPanel({ onIngestComplete }: Props) {
  const [mode, setMode]           = useState<Mode>('file')
  const [loading, setLoading]     = useState(false)
  const [progress, setProgress]   = useState('')
  const [error, setError]         = useState<string | null>(null)
  const [result, setResult]       = useState<IngestResult | IngestDirResult | null>(null)
  const [dragging, setDragging]   = useState(false)
  const [text, setText]           = useState('')
  const [dirPath, setDirPath]     = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const reset = () => {
    setError(null)
    setResult(null)
    setProgress('')
  }

  // ── File mode ──────────────────────────────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files)
    if (arr.length === 0) return

    // 50 MB per-file client guard
    const oversized = arr.filter(f => f.size > 50 * 1024 * 1024)
    if (oversized.length > 0) {
      setError(`File(s) too large (>50 MB): ${oversized.map(f => f.name).join(', ')}`)
      return
    }

    reset()
    setLoading(true)
    const fileErrs: string[] = []
    try {
      const results = await ingestFiles(
        arr,
        (done, total, name) => {
          if (name) setProgress(`${done + 1} / ${total} — ${name}`)
          else setProgress(`Done — ${total} file(s) processed`)
        },
        (filename, err) => fileErrs.push(`${filename}: ${err}`),
      )
      // Aggregate totals
      const agg: IngestResult = results.reduce((acc, r) => ({
        ...acc,
        triples_extracted: acc.triples_extracted + r.triples_extracted,
        nodes_added:       acc.nodes_added       + r.nodes_added,
        edges_added:       acc.edges_added       + r.edges_added,
        triples:           [...acc.triples, ...r.triples],
      }), {
        status: 'ok', triples_extracted: 0, nodes_added: 0,
        edges_added: 0, version: 0, triples: [], source: '',
      })
      setResult(agg)
      if (fileErrs.length > 0) setError(`${fileErrs.length} file(s) failed: ${fileErrs.join('; ')}`)
      onIngestComplete()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [onIngestComplete])

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files)
  }

  // ── Text mode ──────────────────────────────────────────────────────────────
  const handleText = async () => {
    if (!text.trim()) return
    reset()
    setLoading(true)
    try {
      const r = await ingestText(text, 'manual')
      setResult(r)
      setText('')
      onIngestComplete()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  // ── Directory mode ─────────────────────────────────────────────────────────
  const handleDirectory = async () => {
    if (!dirPath.trim()) return
    reset()
    setLoading(true)
    setProgress('Walking directory…')
    try {
      const r = await ingestDirectory(dirPath.trim())
      setResult(r)
      setDirPath('')
      onIngestComplete()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  // ── Result summary ─────────────────────────────────────────────────────────
  const renderResult = () => {
    if (!result) return null
    const isDir   = 'files_processed' in result
    const nodes   = result.nodes_added
    const edges   = result.edges_added

    // Confidence tier breakdown from triples (file / text mode)
    const tierCounts: Record<string, number> = {}
    if ('triples' in result) {
      for (const t of result.triples) {
        const tier = t.confidence_tier || 'INFERRED'
        tierCounts[tier] = (tierCounts[tier] || 0) + 1
      }
    }

    return (
      <div style={{
        background: 'rgba(16,185,129,0.07)', border: '1px solid rgba(16,185,129,0.25)',
        borderRadius: 8, padding: '0.75rem', marginTop: '0.75rem', fontSize: '0.78rem',
      }}>
        <div style={{ color: '#10b981', fontWeight: 700, marginBottom: '0.4rem' }}>
          ✓ Ingested successfully
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem', color: 'var(--text-muted)' }}>
          {isDir && <span>Files: <b style={{ color: 'var(--text-primary)' }}>{(result as IngestDirResult).files_processed}</b></span>}
          {'triples_extracted' in result && (
            <span>Triples: <b style={{ color: 'var(--text-primary)' }}>{result.triples_extracted}</b></span>
          )}
          {'triples_total' in result && (
            <span>Triples: <b style={{ color: 'var(--text-primary)' }}>{(result as IngestDirResult).triples_total}</b></span>
          )}
          <span>Nodes: <b style={{ color: 'var(--cyan)' }}>+{nodes}</b></span>
          <span>Edges: <b style={{ color: 'var(--purple)' }}>+{edges}</b></span>
        </div>
        {Object.keys(tierCounts).length > 0 && (
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            {Object.entries(tierCounts).map(([tier, count]) => (
              <span key={tier} style={{
                fontSize: '0.68rem', padding: '0.1rem 0.45rem',
                borderRadius: 100, border: `1px solid ${TIER_COLORS[tier]}44`,
                background: `${TIER_COLORS[tier]}18`, color: TIER_COLORS[tier],
                fontWeight: 600,
              }}>
                {tier}: {count}
              </span>
            ))}
          </div>
        )}
      </div>
    )
  }

  const TABS: { id: Mode; label: string; icon: string }[] = [
    { id: 'file',      label: 'Files',     icon: '📁' },
    { id: 'text',      label: 'Text',      icon: '📝' },
    { id: 'directory', label: 'Directory', icon: '🗂' },
  ]

  return (
    <div style={{ padding: '0.875rem 1rem' }}>
      {/* Tab strip */}
      <div style={{ display: 'flex', gap: '0.25rem', marginBottom: '0.875rem' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => { setMode(tab.id); reset() }}
            style={{
              flex: 1, padding: '0.35rem 0', border: 'none', borderRadius: 6,
              fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
              background: mode === tab.id ? 'var(--cyan-dim)' : 'var(--bg-surface)',
              color:      mode === tab.id ? 'var(--cyan)'     : 'var(--text-muted)',
              outline:    mode === tab.id ? '1px solid var(--border-accent)' : '1px solid var(--border)',
              transition: 'all 0.15s',
            }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* File mode */}
      {mode === 'file' && (
        <>
          <div
            className={`dropzone ${dragging ? 'active' : ''}`}
            style={{ padding: '1.25rem', borderRadius: 8 }}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <span className="dropzone-icon" style={{ fontSize: '1.75rem' }}>
              {loading ? '⏳' : '📂'}
            </span>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
              {loading ? progress || 'Processing…' : 'Drop files or click to browse'}
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              Code · PDF · Images · ZIP · Text · Markdown
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPT}
            style={{ display: 'none' }}
            onChange={e => e.target.files && handleFiles(e.target.files)}
          />
        </>
      )}

      {/* Text mode */}
      {mode === 'text' && (
        <>
          <textarea
            className="input"
            placeholder="Paste text, article, documentation, or any prose to extract knowledge triples…"
            value={text}
            onChange={e => setText(e.target.value)}
            style={{ minHeight: 100, marginBottom: '0.5rem', resize: 'vertical', fontSize: '0.82rem' }}
          />
          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', fontSize: '0.82rem' }}
            disabled={loading || !text.trim()}
            onClick={handleText}
          >
            {loading ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Extracting…</> : '⚡ Extract & Ingest'}
          </button>
        </>
      )}

      {/* Directory mode */}
      {mode === 'directory' && (
        <>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem', lineHeight: 1.5 }}>
            Enter the absolute path to a directory on the server. All code, docs, PDFs, and images will be extracted and ingested automatically. Already-ingested files are skipped.
          </div>
          <input
            className="input"
            placeholder="C:\Users\ethan\projects\my-repo"
            value={dirPath}
            onChange={e => setDirPath(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleDirectory()}
            style={{ marginBottom: '0.5rem', fontSize: '0.82rem' }}
          />
          {loading && (
            <div style={{ fontSize: '0.72rem', color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem' }}>
              {progress || 'Walking directory…'}
            </div>
          )}
          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', fontSize: '0.82rem' }}
            disabled={loading || !dirPath.trim()}
            onClick={handleDirectory}
          >
            {loading
              ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Ingesting…</>
              : '🗂 Ingest Codebase'}
          </button>
        </>
      )}

      {/* Error */}
      {error && (
        <div className="error-banner" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
          <span>⚠</span> {error}
        </div>
      )}

      {/* Result */}
      {renderResult()}
    </div>
  )
}
