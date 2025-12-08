'use client'
import React, { useMemo } from 'react'
import { Group, Line, Arrow, Rect } from 'react-konva'


interface Leak {
    coord: [number, number]
    rate_kg_per_s: number
}

interface Road {
    id: number
    start: [number, number]
    end: [number, number]
    pipe_type?: string
    leak?: Leak
}
interface Props {
    roads: Road[]
    frame: number
    pipeVelocities?: Record<string, number[]>
    pipeFlows?: Record<string, number[]>
    showFlow: boolean
    onSelect?: (sel: { type: 'road'; id: number }) => void
}

const mapColor = (v: number, min: number, max: number) => {
    const t = min === max ? 0.5 : (v - min) / (max - min)
    const r = Math.round(255 * t)
    const g = Math.round(255 * (1 - t))
    return `rgb(${r},${g},0)`
}

function getLeakColor(rate: number): string {
    // Clamp between 0 and 0.5 (your max)
    const clamped = Math.min(rate, 0.5);

    // Map value (0 → greenish, 0.5 → red)
    const t = clamped / 0.5; // normalize 0–1
    const r = Math.round(255 * t);
    const g = Math.round(255 * (1 - t));
    const b = 0;

    // Slight transparency for layering
    const alpha = 0.6 + 0.3 * t; // 0.6→0.9
    return `rgba(${r},${g},${b},${alpha})`;
}

export function PipesLayer({
    roads,
    frame,
    pipeVelocities = {},
    pipeFlows = {},
    showFlow,
    onSelect,
}: Props) {
    const stats = useMemo(() => {
        const velGroups: Record<string, number[]> = { main: [], side: [], 'building connection': [] }
        roads.forEach(r => {
            const arr = pipeVelocities[`v_mean_pipe_${r.id}`] || []
            arr.forEach(v => velGroups[r.pipe_type ?? 'side'].push(Math.abs(v)))
        })
        const vMinGroup: Record<string, number> = {}
        const vMaxGroup: Record<string, number> = {}
        Object.entries(velGroups).forEach(([t, vals]) => {
            vMinGroup[t] = vals.length ? Math.min(...vals) : 0
            vMaxGroup[t] = vals.length ? Math.max(...vals) : 1
        })
        const allFlows = roads.flatMap(r => (pipeFlows[`flow_pipe_${r.id}`] || []).map(f => Math.abs(f)))
        return {
            vMinGroup,
            vMaxGroup,
            fMin: allFlows.length ? Math.min(...allFlows) : 0,
            fMax: allFlows.length ? Math.max(...allFlows) : 1,
        }
    }, [roads, pipeVelocities, pipeFlows])

    return (
        <Group>
            {roads.map(r => {
                const type = r.pipe_type ?? 'side'
                const vel = (pipeVelocities[`v_mean_pipe_${r.id}`] || [0])[frame]
                let color: string
                if (showFlow) {
                    const flow = (pipeFlows[`flow_pipe_${r.id}`] || [0])[frame]
                    color =
                        stats.fMin === stats.fMax
                            ? '#666'
                            : mapColor(Math.abs(flow), stats.fMin, stats.fMax)
                } else {
                    const vmin = stats.vMinGroup[type]
                    const vmax = stats.vMaxGroup[type]
                    color = vmin === vmax ? '#666' : mapColor(Math.abs(vel), vmin, vmax)
                }

                const width = type === 'main' ? 5 : type === 'side' ? 2 : 1

                return (
                    <Group key={r.id}>
                        <Line
                            points={[r.start[0], r.start[1], r.end[0], r.end[1]]}
                            stroke={color}
                            strokeWidth={width}
                            onClick={() => onSelect?.({ type: 'road', id: r.id })}
                        />
                        {/* velocity arrow */}
                        <Arrow
                            points={[
                                (r.start[0] + r.end[0]) / 2,
                                (r.start[1] + r.end[1]) / 2,
                                vel >= 0 ? r.end[0] : r.start[0],
                                vel >= 0 ? r.end[1] : r.start[1],
                            ]}
                            pointerLength={6}
                            pointerWidth={6}
                            fill={color}
                            stroke={color}
                        />
                        {r.leak && (
                            <Rect
                                x={r.leak.coord[0] - 4}
                                y={r.leak.coord[1] - 4}
                                width={8}
                                height={8}
                                fill={getLeakColor(r.leak.rate_kg_per_s)}
                                stroke="black"
                                strokeWidth={0.5}
                                cornerRadius={2}
                                onClick={() => onSelect?.({ type: 'road', id: r.id })}
                            />
                        )}
                    </Group>
                )
            })}
        </Group>
    )
}
