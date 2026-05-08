'use client'
import { useEffect, useRef } from 'react'

interface ChartProps {
  type?: 'bar' | 'line' | 'doughnut'
  labels: string[]
  datasets: Array<{
    label: string
    data: number[]
    backgroundColor?: string | string[]
    borderColor?: string | string[]
    borderWidth?: number
    fill?: boolean
  }>
  height?: number
  options?: Record<string, unknown>
}

export default function MiniChart({ type = 'bar', labels, datasets, height = 200, options = {} }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<unknown>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    let chart: unknown

    const init = async () => {
      const { Chart, registerables } = await import('chart.js')
      Chart.register(...registerables)

      if (chartRef.current) {
        (chartRef.current as { destroy: () => void }).destroy()
      }

      chart = new Chart(canvasRef.current!, {
        type,
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: '#8b9ab5', font: { family: 'Inter' }, boxWidth: 12 },
            },
            tooltip: {
              backgroundColor: '#12121e',
              borderColor: 'rgba(0,212,255,0.2)',
              borderWidth: 1,
              titleColor: '#f0f4ff',
              bodyColor: '#8b9ab5',
            },
          },
          scales: type !== 'doughnut' ? {
            x: {
              ticks: { color: '#4a566d', font: { size: 10 } },
              grid: { color: 'rgba(255,255,255,0.04)' },
            },
            y: {
              ticks: { color: '#4a566d', font: { size: 10 } },
              grid: { color: 'rgba(255,255,255,0.04)' },
              beginAtZero: true,
            },
          } : undefined,
          ...options,
        },
      })
      chartRef.current = chart
    }

    init()
    return () => {
      if (chartRef.current) {
        (chartRef.current as { destroy: () => void }).destroy()
        chartRef.current = null
      }
    }
  }, [type, labels, datasets, options])

  return (
    <div style={{ height, position: 'relative' }}>
      <canvas ref={canvasRef} />
    </div>
  )
}
