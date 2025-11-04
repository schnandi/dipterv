'use client'
import React, { useMemo } from 'react'
import { Group, Line } from 'react-konva'

interface Building {
    id: number
    corners: [number, number][]
    district?: string
}
interface Props {
    buildings: Building[]
    sinkFlows: Record<string, number[]>
    frame: number
    onSelect?: (sel: { type: 'building'; id: number }) => void
}

const mapColor = (v: number, min: number, max: number) => {
    const t = min === max ? 0.5 : (v - min) / (max - min)
    const r = Math.round(255 * t)
    const g = Math.round(255 * (1 - t))
    return `rgb(${r},${g},0)`
}

export function BuildingsLayer({ buildings, sinkFlows, frame, onSelect }: Props) {
    const stats = useMemo(() => {
        const res = buildings.filter(b => b.district === 'residential')
        const ind = buildings.filter(b => b.district === 'industrial')
        const resVals = res.map(b => (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame])
        const indVals = ind.map(b => (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame])
        const rMin = resVals.length ? Math.min(...resVals) : 0
        const rMax = resVals.length ? Math.max(...resVals) : 1
        const iMin = indVals.length ? Math.min(...indVals) : 0
        const iMax = indVals.length ? Math.max(...indVals) : 1
        return { rMin, rMax, iMin, iMax }
    }, [buildings, sinkFlows, frame])

    return (
        <Group>
            {buildings.map(b => {
                const raw = (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame]
                const isRes = b.district === 'residential'
                const [min, max] = isRes ? [stats.rMin, stats.rMax] : [stats.iMin, stats.iMax]
                const fill = mapColor(raw, min, max)
                const stroke = isRes ? '#00A5E3' : '#FF60A8'
                const pts = b.corners.flat()
                return (
                    <Line
                        key={b.id}
                        points={pts}
                        closed
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={2}
                        onClick={() => onSelect?.({ type: 'building', id: b.id })}
                    />
                )
            })}
        </Group>
    )
}
