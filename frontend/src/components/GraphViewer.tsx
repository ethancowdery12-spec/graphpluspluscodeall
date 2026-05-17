'use client'
import { useEffect, useRef, useCallback } from 'react'
import type { GraphData, GraphNode } from '@/lib/api'

interface Props {
  data: GraphData
  highlightPaths?: string[][]
  onNodeClick?: (node: GraphNode) => void
}

// Confidence tier → edge colour
const TIER_STROKE: Record<string, string> = {
  EXTRACTED: '#34d399',   // emerald — AST-parsed, deterministic
  INFERRED:  '#60a5fa',   // blue    — LLM inferred
  AMBIGUOUS: '#f59e0b',   // amber   — regex / low-confidence
}
const TIER_OPACITY: Record<string, number> = {
  EXTRACTED: 0.85,
  INFERRED:  0.65,
  AMBIGUOUS: 0.45,
}

export default function GraphViewer({ data, highlightPaths = [], onNodeClick }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const simRef = useRef<unknown>(null)

  const render = useCallback(async () => {
    if (!svgRef.current || !data.nodes.length) return

    const d3 = await import('d3')
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Use the parent container's concrete pixel size to avoid SVGLength % errors.
    const container = svgRef.current.parentElement ?? svgRef.current
    const rect   = container.getBoundingClientRect()
    const width  = rect.width  || container.clientWidth  || 900
    const height = rect.height || container.clientHeight || 600

    // Don't render until the container has actual dimensions
    if (width < 10 || height < 10) return

    // ── Highlight sets ─────────────────────────────────────────────────────
    const highlightNodeIds = new Set(highlightPaths.flat())
    const highlightEdgePairs = new Set(
      highlightPaths.flatMap(path =>
        path.slice(0, -1).map((id, i) => `${id}::${path[i + 1]}`)
      )
    )

    // ── Defs ───────────────────────────────────────────────────────────────
    const defs = svg.append('defs')

    const addMarker = (id: string, color: string, opacity = 1) =>
      defs.append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -4 8 8')
        .attr('refX', 20).attr('refY', 0)
        .attr('markerWidth', 5).attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L8,0L0,4')
        .attr('fill', color).attr('opacity', opacity)

    addMarker('arrow-extracted', '#34d399', 0.85)
    addMarker('arrow-inferred',  '#60a5fa', 0.65)
    addMarker('arrow-ambiguous', '#f59e0b', 0.45)
    addMarker('arrow-highlight', '#a855f7')

    data.nodes.forEach(node => {
      const grad = defs.append('radialGradient')
        .attr('id', `grad-${node.id}`)
        .attr('cx', '35%').attr('cy', '35%')
      grad.append('stop').attr('offset',   '0%').attr('stop-color', '#fff').attr('stop-opacity', 0.45)
      grad.append('stop').attr('offset', '100%').attr('stop-color', node.color).attr('stop-opacity', 1)
    })

    // ── Zoom ───────────────────────────────────────────────────────────────
    const g = svg.append('g')
    const zoomBehavior = (d3.zoom() as d3.ZoomBehavior<SVGSVGElement, unknown>)
      .scaleExtent([0.05, 10])
      .extent([[0, 0], [width, height]])   // explicit pixels — prevents SVGLength % error
      .translateExtent([[-width * 3, -height * 3], [width * 4, height * 4]])
      .on('zoom', event => g.attr('transform', event.transform))
    svg.call(zoomBehavior)
    svg.on('dblclick.zoom', null)

    // ── Auto-fit ──────────────────────────────────────────────────────────
    const fitViewport = (animate = true) => {
      const ns = nodes as Array<{ x?: number; y?: number; size: number }>
      const xs = ns.map(n => n.x ?? width / 2)
      const ys = ns.map(n => n.y ?? height / 2)
      const x0 = Math.min(...xs), x1 = Math.max(...xs)
      const y0 = Math.min(...ys), y1 = Math.max(...ys)
      const gw  = (x1 - x0) || 200
      const gh  = (y1 - y0) || 200
      const pad = 80
      const scale = Math.min((width - pad * 2) / gw, (height - pad * 2) / gh, 2.0)
      const tx = (width  - scale * (x0 + x1)) / 2
      const ty = (height - scale * (y0 + y1)) / 2
      const transform = d3.zoomIdentity.translate(tx, ty).scale(scale)
      if (animate) svg.transition().duration(800).call(zoomBehavior.transform, transform)
      else         svg.call(zoomBehavior.transform, transform)
    }

    // ── Simulation ─────────────────────────────────────────────────────────
    const nodes = data.nodes.map(n => ({
      ...n,
      x: width  / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
    }))
    const nodeMap = new Map(nodes.map(n => [n.id, n]))
    const edges = data.edges.map(e => ({
      ...e,
      source: nodeMap.get(e.source) ?? e.source,
      target: nodeMap.get(e.target) ?? e.target,
    }))

    const sim = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force('link',      d3.forceLink(edges)
        .id((d: unknown) => (d as GraphNode).id)
        .distance(110).strength(0.5))
      .force('charge',    d3.forceManyBody().strength(-280).distanceMax(600))
      .force('center',    d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius((d: unknown) => (d as GraphNode).size + 10))

    simRef.current = sim

    // ── Edges ──────────────────────────────────────────────────────────────
    type EdgeDatum = typeof edges[0]

    const isHighlightEdge = (d: EdgeDatum) => {
      const key = `${(d.source as GraphNode).id}::${(d.target as GraphNode).id}`
      return highlightEdgePairs.has(key)
    }

    const edgeColor = (d: EdgeDatum) => {
      if (isHighlightEdge(d)) return '#a855f7'
      return TIER_STROKE[d.confidence_tier] ?? '#60a5fa'
    }
    const edgeOpacity = (d: EdgeDatum) => {
      if (isHighlightEdge(d)) return 1
      return TIER_OPACITY[d.confidence_tier] ?? 0.65
    }
    const edgeMarker = (d: EdgeDatum) => {
      if (isHighlightEdge(d)) return 'url(#arrow-highlight)'
      const tier = (d.confidence_tier ?? 'INFERRED').toLowerCase()
      return `url(#arrow-${tier})`
    }
    const edgeWidth = (d: EdgeDatum) =>
      isHighlightEdge(d) ? 2.5 : d.confidence_tier === 'EXTRACTED' ? 1.6 : 1.2

    const link = g.append('g').attr('class', 'edges').selectAll('line')
      .data(edges).enter().append('line')
      .attr('stroke',       edgeColor)
      .attr('stroke-width', edgeWidth)
      .attr('stroke-dasharray', (d: EdgeDatum) =>
        d.confidence_tier === 'AMBIGUOUS' && !isHighlightEdge(d) ? '4,3' : null)
      .attr('marker-end',   edgeMarker)
      .attr('opacity',      edgeOpacity)

    // Edge labels on highlighted paths only
    const linkLabel = g.append('g').selectAll('text')
      .data(edges.filter(e => isHighlightEdge(e)))
      .enter().append('text')
      .attr('font-size', '10px')
      .attr('fill', 'rgba(168,85,247,0.9)')
      .attr('text-anchor', 'middle')
      .attr('dy', '-5px')
      .attr('pointer-events', 'none')
      .text(d => d.predicate.replace(/_/g, ' '))

    // ── Edge tooltip ───────────────────────────────────────────────────────
    const edgeTooltipSel = d3.select<HTMLElement, unknown>('#graph-tooltip')
    link
      .on('mouseover', (event, d) => {
        const tier = d.confidence_tier ?? 'INFERRED'
        const color = TIER_STROKE[tier] ?? '#60a5fa'
        edgeTooltipSel
          .style('display', 'block')
          .style('left', `${event.pageX + 14}px`)
          .style('top',  `${event.pageY - 12}px`)
          .html(`<div style="color:${color};font-weight:700;margin-bottom:3px">${d.predicate.replace(/_/g,' ')}</div>
                 <div style="color:#8b9ab5">Confidence: ${((d.confidence ?? 1) * 100).toFixed(0)}%</div>
                 <div style="color:${color};font-size:0.7em">${tier}</div>`)
      })
      .on('mousemove', event =>
        edgeTooltipSel
          .style('left', `${event.pageX + 14}px`)
          .style('top',  `${event.pageY - 12}px`)
      )
      .on('mouseout', () => edgeTooltipSel.style('display', 'none'))

    // ── Nodes ──────────────────────────────────────────────────────────────
    const node = g.append('g').attr('class', 'nodes').selectAll('g')
      .data(nodes).enter().append('g')
      .attr('cursor', 'pointer')
      .call(
        (d3.drag() as d3.DragBehavior<SVGGElement, unknown, unknown>)
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart()
            const n = d as { fx?: number; fy?: number; x: number; y: number }
            n.fx = n.x; n.fy = n.y
          })
          .on('drag', (event, d) => {
            const n = d as { fx: number; fy: number }
            n.fx = event.x; n.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0)
            const n = d as { fx: number | null; fy: number | null }
            n.fx = null; n.fy = null
          })
      )
      .on('click', (_, d) => onNodeClick?.(d as GraphNode))

    // Glow ring for highlighted nodes
    node.filter(d => highlightNodeIds.has(d.id))
      .append('circle')
      .attr('r', d => d.size + 7)
      .attr('fill', 'none')
      .attr('stroke', '#00d4ff')
      .attr('stroke-width', 2.5)
      .attr('opacity', 0.65)
      .attr('class', 'pulse-ring')

    // Main circle
    node.append('circle')
      .attr('r', d => d.size)
      .attr('fill', d => `url(#grad-${d.id})`)
      .attr('stroke', d => highlightNodeIds.has(d.id) ? '#00d4ff' : d.color)
      .attr('stroke-width', d => highlightNodeIds.has(d.id) ? 2.5 : 1.5)
      .attr('opacity', d =>
        !highlightNodeIds.size || highlightNodeIds.has(d.id) ? 1 : 0.35)

    // Label
    node.append('text')
      .attr('dy', d => d.size + 13)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-family', 'Inter, sans-serif')
      .attr('fill', d => highlightNodeIds.has(d.id) ? '#fff' : 'rgba(255,255,255,0.72)')
      .attr('font-weight', d => highlightNodeIds.has(d.id) ? '700' : '400')
      .attr('pointer-events', 'none')
      .text(d => d.label.length > 18 ? d.label.slice(0, 16) + '…' : d.label)

    // ── Node Tooltip ───────────────────────────────────────────────────────
    const tooltipSel = d3.select<HTMLElement, unknown>('#graph-tooltip')
    node
      .on('mouseover', (event, d) =>
        tooltipSel
          .style('display', 'block')
          .style('left', `${event.pageX + 14}px`)
          .style('top',  `${event.pageY - 12}px`)
          .html(`<div style="color:${d.color};font-weight:700;margin-bottom:4px">${d.label}</div>
                 <div style="color:#8b9ab5">Type: ${d.type}</div>
                 <div style="color:#8b9ab5">Degree: ${d.degree}</div>
                 <div style="color:#8b9ab5">PageRank: ${d.pagerank.toFixed(4)}</div>`)
      )
      .on('mousemove', event =>
        tooltipSel
          .style('left', `${event.pageX + 14}px`)
          .style('top',  `${event.pageY - 12}px`)
      )
      .on('mouseout', () => tooltipSel.style('display', 'none'))

    // ── Tick ───────────────────────────────────────────────────────────────
    let fitted = false

    sim.on('tick', () => {
      link
        .attr('x1', d => (d.source as { x: number }).x)
        .attr('y1', d => (d.source as { y: number }).y)
        .attr('x2', d => (d.target as { x: number }).x)
        .attr('y2', d => (d.target as { y: number }).y)

      linkLabel
        .attr('x', d => ((d.source as { x: number }).x + (d.target as { x: number }).x) / 2)
        .attr('y', d => ((d.source as { y: number }).y + (d.target as { y: number }).y) / 2)

      node.attr('transform', d => {
        const n = d as { x: number; y: number }
        return `translate(${n.x},${n.y})`
      })

      if (!fitted && sim.alpha() < 0.05) {
        fitted = true
        fitViewport(true)
      }
    })

    sim.on('end', () => {
      if (!fitted) { fitted = true; fitViewport(true) }
    })

    // ── Pulse animation ────────────────────────────────────────────────────
    const style = document.createElement('style')
    style.textContent = `
      .pulse-ring { animation: graphPulse 2.2s ease-in-out infinite; }
      @keyframes graphPulse {
        0%, 100% { opacity: 0.65; }
        50%       { opacity: 0.1; }
      }
    `
    document.head.appendChild(style)

    return () => {
      sim.stop()
      if (document.head.contains(style)) document.head.removeChild(style)
    }
  }, [data, highlightPaths, onNodeClick])

  useEffect(() => {
    let cleanup: (() => void) | undefined
    render()
      .then(fn => { cleanup = fn })
      .catch(err => console.error('[GraphViewer] render error:', err))
    return () => cleanup?.()
  }, [render])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 400 }}>
      <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      <div id="graph-tooltip" className="tooltip" style={{ display: 'none', position: 'fixed' }} />
    </div>
  )
}
