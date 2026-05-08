'use client'
import { useEffect, useRef, useCallback } from 'react'
import type { GraphData, GraphNode, GraphEdge } from '@/lib/api'

interface Props {
  data: GraphData
  highlightPaths?: string[][]  // node IDs to highlight
  onNodeClick?: (node: GraphNode) => void
}

export default function GraphViewer({ data, highlightPaths = [], onNodeClick }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const simRef = useRef<unknown>(null)

  const render = useCallback(async () => {
    if (!svgRef.current || !data.nodes.length) return

    const d3 = await import('d3')
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth || 900
    const height = svgRef.current.clientHeight || 600

    // Build highlight set
    const highlightNodeIds = new Set(highlightPaths.flat())
    const highlightEdgePairs = new Set(
      highlightPaths.flatMap(path =>
        path.slice(0, -1).map((id, i) => `${id}::${path[i + 1]}`)
      )
    )

    // Defs — arrowhead markers
    const defs = svg.append('defs')
    const addMarker = (id: string, color: string) => {
      defs.append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -4 8 8')
        .attr('refX', 14).attr('refY', 0)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-4L8,0L0,4')
        .attr('fill', color)
    }
    addMarker('arrow-dim', 'rgba(255,255,255,0.1)')
    addMarker('arrow-cyan', '#00d4ff')
    addMarker('arrow-purple', '#a855f7')

    // Radial gradient for nodes
    data.nodes.forEach(node => {
      const grad = defs.append('radialGradient')
        .attr('id', `grad-${node.id}`)
        .attr('cx', '35%').attr('cy', '35%')
      grad.append('stop').attr('offset', '0%').attr('stop-color', '#fff').attr('stop-opacity', 0.3)
      grad.append('stop').attr('offset', '100%').attr('stop-color', node.color).attr('stop-opacity', 1)
    })

    // Zoom
    const g = svg.append('g')
    svg.call(
      (d3.zoom() as d3.ZoomBehavior<SVGSVGElement, unknown>)
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => g.attr('transform', event.transform))
    )

    // Force simulation
    const nodes = data.nodes.map(n => ({ ...n, x: width / 2, y: height / 2 }))
    const nodeMap = new Map(nodes.map(n => [n.id, n]))
    const edges = data.edges.map(e => ({
      ...e,
      source: nodeMap.get(e.source) || e.source,
      target: nodeMap.get(e.target) || e.target,
    }))

    const sim = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(edges).id((d: unknown) => (d as GraphNode).id).distance(80).strength(0.3))
      .force('charge', d3.forceManyBody().strength(-120).distanceMax(300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: unknown) => (d as GraphNode).size + 4))

    simRef.current = sim

    // Edges
    const link = g.append('g').selectAll('line')
      .data(edges).enter().append('line')
      .attr('stroke', (d) => {
        const key = `${(d.source as GraphNode).id}::${(d.target as GraphNode).id}`
        return highlightEdgePairs.has(key) ? '#a855f7' : 'rgba(255,255,255,0.08)'
      })
      .attr('stroke-width', (d) => {
        const key = `${(d.source as GraphNode).id}::${(d.target as GraphNode).id}`
        return highlightEdgePairs.has(key) ? 2.5 : Math.max(0.5, d.weight)
      })
      .attr('marker-end', (d) => {
        const key = `${(d.source as GraphNode).id}::${(d.target as GraphNode).id}`
        return highlightEdgePairs.has(key) ? 'url(#arrow-purple)' : 'url(#arrow-dim)'
      })
      .attr('opacity', (d) => {
        const key = `${(d.source as GraphNode).id}::${(d.target as GraphNode).id}`
        return highlightEdgePairs.has(key) ? 1 : 0.4
      })

    // Edge labels (only for highlighted)
    const linkLabel = g.append('g').selectAll('text')
      .data(edges.filter(e => {
        const key = `${(e.source as GraphNode).id}::${(e.target as GraphNode).id}`
        return highlightEdgePairs.has(key)
      }))
      .enter().append('text')
      .attr('font-size', '9px')
      .attr('fill', 'rgba(168,85,247,0.8)')
      .attr('text-anchor', 'middle')
      .attr('dy', '-4px')
      .text(d => d.predicate.replace(/_/g, ' '))

    // Nodes
    const node = g.append('g').selectAll('g')
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

    // Outer glow ring for highlighted nodes
    node.filter(d => highlightNodeIds.has(d.id))
      .append('circle')
      .attr('r', d => d.size + 6)
      .attr('fill', 'none')
      .attr('stroke', '#00d4ff')
      .attr('stroke-width', 2)
      .attr('opacity', 0.5)
      .attr('class', 'pulse-ring')

    // Main circle
    node.append('circle')
      .attr('r', d => d.size)
      .attr('fill', d => `url(#grad-${d.id})`)
      .attr('stroke', d => highlightNodeIds.has(d.id) ? '#00d4ff' : d.color)
      .attr('stroke-width', d => highlightNodeIds.has(d.id) ? 2 : 1)
      .attr('opacity', d => highlightNodeIds.has(d.id) || !highlightNodeIds.size ? 1 : 0.3)

    // Label
    node.append('text')
      .attr('dy', d => d.size + 12)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('font-family', 'Inter, sans-serif')
      .attr('fill', d => highlightNodeIds.has(d.id) ? '#fff' : 'rgba(255,255,255,0.5)')
      .attr('font-weight', d => highlightNodeIds.has(d.id) ? '700' : '400')
      .text(d => d.label.length > 18 ? d.label.slice(0, 16) + '…' : d.label)

    // Tooltip
    const tooltip = d3.select('body').select('#graph-tooltip')
    node.on('mouseover', (event, d) => {
      tooltip
        .style('display', 'block')
        .style('left', `${event.pageX + 12}px`)
        .style('top', `${event.pageY - 10}px`)
        .html(`
          <div style="color:${d.color};font-weight:700;margin-bottom:4px">${d.label}</div>
          <div style="color:#8b9ab5">Type: ${d.type}</div>
          <div style="color:#8b9ab5">Degree: ${d.degree}</div>
          <div style="color:#8b9ab5">PageRank: ${d.pagerank.toFixed(4)}</div>
        `)
    }).on('mousemove', (event) => {
      tooltip.style('left', `${event.pageX + 12}px`).style('top', `${event.pageY - 10}px`)
    }).on('mouseout', () => {
      tooltip.style('display', 'none')
    })

    // Simulation tick
    sim.on('tick', () => {
      link
        .attr('x1', d => (d.source as { x: number }).x)
        .attr('y1', d => (d.source as { y: number }).y)
        .attr('x2', d => (d.target as { x: number }).x)
        .attr('y2', d => (d.target as { y: number }).y)

      linkLabel
        .attr('x', d => ((d.source as { x: number }).x + (d.target as { x: number }).x) / 2)
        .attr('y', d => ((d.source as { y: number }).y + (d.target as { y: number }).y) / 2)

      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    // CSS animation for pulse rings
    const style = document.createElement('style')
    style.textContent = `
      .pulse-ring { animation: pulseRing 2s ease-in-out infinite; }
      @keyframes pulseRing {
        0%, 100% { r: 0; opacity: 0.8; }
        50% { r: 12; opacity: 0; }
      }
    `
    document.head.appendChild(style)

    return () => {
      sim.stop()
      document.head.removeChild(style)
    }
  }, [data, highlightPaths, onNodeClick])

  useEffect(() => {
    render()
  }, [render])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <svg ref={svgRef} width="100%" height="100%" />
      <div
        id="graph-tooltip"
        className="tooltip"
        style={{ display: 'none', position: 'fixed' }}
      />
    </div>
  )
}
