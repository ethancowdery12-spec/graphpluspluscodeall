'use client'
import { useState, useRef, useCallback, DragEvent } from 'react'
import { ingestText, ingestFile, ingestFiles, ingestDirectory } from '@/lib/api'
import type { IngestResult, IngestDirResult } from '@/lib/api'

type Mode = 'file' | 'text' | 'directory'

const ACCEPT = [
  '.py', '.js', '.jsx', '.ts', '.tsx', '.go', '.rs', '.java', '.c', '.cpp', '.cc', '.rb',
  '.md', '.txt', '.rst', '.pdf',
  '.zip',
  '.json', '.yaml', '.yml', '.toml', '.csv',
  '.png', '.jpg', '.jpeg', '.webp', '.gif',
  '.ipynb',
].join(',')

const TIER_COLORS: Record<string, string> = {
  EXTRACTED: '#34d399',
  INFERRED:  '#60a5fa',
  AMBIGUOUS: '#f59e0b',
}

export default function IngestPage() {
  const [mode, setMode]         = useState<Mode>('file')
  const [text, setText]         = useState('')
  const [source, setSource]     = useState('manual')
  const [dirPath, setDirPath]   = useState('')
  const [loading, setLoading]   = useState(false)
  const [progress, setProgress] = useState('')
  const [result, setResult]     = useState<IngestResult | IngestDirResult | null>(null)
  const [fileErrors, setFileErrors] = useState<string[]>([])
  const [error, setError]       = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const reset = () => { setError(null); setResult(null); setProgress(''); setFileErrors([]) }

  // ── File mode ──────────────────────────────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files)
    if (arr.length === 0) return

    const oversized = arr.filter(f => f.size > 50 * 1024 * 1024)
    if (oversized.length > 0) {
      setError(`File(s) too large (>50 MB): ${oversized.map(f => f.name).join(', ')}`)
      return
    }

    reset()
    setLoading(true)
    const errs: string[] = []
    try {
      const results = await ingestFiles(
        arr,
        (done, total, name) => {
          if (name) setProgress(`${done + 1} / ${total} — ${name}`)
          else setProgress(`Done — ${total} file(s) processed`)
        },
        (filename, err) => errs.push(`${filename}: ${err}`),
      )
      if (errs.length > 0) setFileErrors(errs)
      const agg: IngestResult = results.reduce((acc, r) => ({
        ...acc,
        triples_extracted: acc.triples_extracted + r.triples_extracted,
        nodes_added:       acc.nodes_added       + r.nodes_added,
        edges_added:       acc.edges_added       + r.edges_added,
        triples:           [...acc.triples, ...r.triples],
      }), { status: 'ok', triples_extracted: 0, nodes_added: 0, edges_added: 0, version: 0, triples: [], source: '' })
      setResult(agg)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files)
    } else {
      setError('Folder drag-drop is not supported in browsers. Use the Directory tab and enter the folder path.')
    }
  }

  // ── Text mode ──────────────────────────────────────────────────────────────
  const handleText = async () => {
    if (!text.trim()) return
    reset()
    setLoading(true)
    try {
      const r = await ingestText(text, source)
      setResult(r)
      setText('')
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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  // ── Result ─────────────────────────────────────────────────────────────────
  const renderResult = () => {
    if (!result) return null
    const isDir = 'files_processed' in result

    const tierCounts: Record<string, number> = {}
    if ('triples' in result) {
      for (const t of result.triples) {
        const tier = t.confidence_tier || 'INFERRED'
        tierCounts[tier] = (tierCounts[tier] || 0) + 1
      }
    }

    return (
      <div style={{
        marginTop: '1.5rem', background: 'var(--bg-surface)',
        border: '1px solid var(--border-accent)', borderRadius: 12, padding: '1.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <span style={{ fontSize: '1.2rem' }}>✅</span>
          <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>Ingestion complete</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.25rem' }}>
          {isDir
            ? [
                ['Files processed', (result as IngestDirResult).files_processed],
                ['Nodes added', result.nodes_added],
                ['Edges added', result.edges_added],
              ].map(([label, value]) => (
                <div key={String(label)} style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '0.875rem', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, background: 'var(--grad-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{value}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{label}</div>
                </div>
              ))
            : [
                ['Triples', (result as IngestResult).triples_extracted],
                ['Nodes added', result.nodes_added],
                ['Edges added', result.edges_added],
              ].map(([label, value]) => (
                <div key={String(label)} style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '0.875rem', textAlign: 'center',
                }}>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, background: 'var(--grad-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{value}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{label}</div>
                </div>
              ))
          }
        </div>

        {Object.keys(tierCounts).length > 0 && (
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            {Object.entries(tierCounts).map(([tier, count]) => (
              <span key={tier} style={{
                fontSize: '0.72rem', padding: '0.15rem 0.5rem',
                borderRadius: 100, border: `1px solid ${TIER_COLORS[tier]}44`,
                background: `${TIER_COLORS[tier]}18`, color: TIER_COLORS[tier],
                fontWeight: 600,
              }}>
                {tier}: {count}
              </span>
            ))}
          </div>
        )}

        {'triples' in result && (result as IngestResult).triples.length > 0 && (
          <>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Extracted triples
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: 320, overflow: 'auto' }}>
              {(result as IngestResult).triples.map((t, i) => (
                <div key={i} className="path-item" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.82rem' }}>
                  <span className="hop-node">{t.subject}</span>
                  <span className="hop-arrow" style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>{t.predicate}</span>
                  <span className="hop-node">{t.object}</span>
                  <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {(t.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    )
  }

  const TABS: { id: Mode; label: string; icon: string }[] = [
    { id: 'file',      label: 'Upload Files', icon: '📄' },
    { id: 'text',      label: 'Paste Text',   icon: '✏️' },
    { id: 'directory', label: 'Directory',    icon: '🗂️' },
  ]

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '2rem 1.5rem' }}>
      <h1 style={{
        fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.375rem',
        background: 'var(--grad-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
      }}>
        Ingest Documents
      </h1>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '2rem' }}>
        Extract entities and relations from text, files, or an entire codebase directory.
      </p>

      {/* Mode tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => { setMode(tab.id); reset() }}
            className={`btn ${mode === tab.id ? 'btn-primary' : 'btn-secondary'}`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 12, padding: '1.5rem', marginBottom: '1.5rem',
      }}>

        {/* File mode */}
        {mode === 'file' && (
          <>
            <div
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? 'var(--cyan)' : 'var(--border-accent)'}`,
                borderRadius: 10, padding: '3rem 2rem', textAlign: 'center', cursor: 'pointer',
                transition: 'border-color 0.2s',
                background: dragging ? 'var(--cyan-dim)' : 'transparent',
              }}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>
                {loading ? '⏳' : '📂'}
              </div>
              <div style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: '0.25rem' }}>
                {loading ? (progress || 'Processing…') : 'Drop files or click to browse'}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Code · PDF · Images · ZIP · Markdown · Notebooks — multiple files OK
              </div>
            </div>
            <input
              ref={fileRef}
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
            <label style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
              Source label
            </label>
            <input
              className="input"
              value={source}
              onChange={e => setSource(e.target.value)}
              placeholder="e.g. paper, blog-post, manual"
              style={{ marginBottom: '1rem', padding: '0.5rem 0.75rem', fontSize: '0.85rem' }}
            />
            <label style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
              Text content
            </label>
            <textarea
              className="input"
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Paste or type text to extract entities and relations…"
              style={{ minHeight: 220, resize: 'vertical' }}
            />
          </>
        )}

        {/* Directory mode */}
        {mode === 'directory' && (
          <>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: 1.6 }}>
              Enter the absolute path to a directory on this machine. All code, docs, PDFs, images, and archives
              will be recursively extracted and ingested. Already-ingested files are skipped automatically.
            </p>
            <label style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
              Directory path
            </label>
            <input
              className="input"
              value={dirPath}
              onChange={e => setDirPath(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleDirectory()}
              placeholder="C:\Users\ethan\projects\my-repo"
              style={{ marginBottom: '0.5rem', fontSize: '0.85rem' }}
            />
            {loading && (
              <div style={{ fontSize: '0.78rem', color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginTop: '0.5rem' }}>
                {progress || 'Walking directory…'}
              </div>
            )}
          </>
        )}
      </div>

      {/* Submit button */}
      {mode !== 'file' && (
        <button
          className="btn btn-primary"
          style={{ width: '100%', justifyContent: 'center', padding: '0.75rem' }}
          onClick={mode === 'text' ? handleText : handleDirectory}
          disabled={loading || (mode === 'text' ? !text.trim() : !dirPath.trim())}
        >
          {loading
            ? <><span className="spinner" style={{ width: 14, height: 14 }} /> {mode === 'directory' ? 'Ingesting…' : 'Extracting…'}</>
            : mode === 'directory' ? '🗂️ Ingest Codebase' : '⚡ Ingest'}
        </button>
      )}

      {/* Per-file errors */}
      {fileErrors.length > 0 && (
        <div className="error-banner" style={{ marginTop: '1rem' }}>
          <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>⚠ Some files failed:</div>
          {fileErrors.map((e, i) => <div key={i} style={{ fontSize: '0.8rem' }}>{e}</div>)}
        </div>
      )}

      {/* General error */}
      {error && (
        <div className="error-banner" style={{ marginTop: '1rem' }}>
          <span>⚠</span> {error}
        </div>
      )}

      {renderResult()}
    </div>
  )
}
