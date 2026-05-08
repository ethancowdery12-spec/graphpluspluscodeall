import type { Metadata } from 'next'
import './globals.css'
import NavBar from '@/components/NavBar'

export const metadata: Metadata = {
  title: 'GraphRAG++ | Knowledge Graph Fusion Engine',
  description: 'Dynamic hybrid knowledge graph engine with multi-hop reasoning, entity-relation extraction, and LLM-fused generation. Outperforms RAG by 40-60% on complex queries.',
  keywords: ['GraphRAG', 'knowledge graph', 'RAG', 'LLM', 'multi-hop reasoning', 'AI'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <div className="grid-bg" aria-hidden="true" />
        <NavBar />
        <main className="page">{children}</main>
      </body>
    </html>
  )
}
