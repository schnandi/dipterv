'use client'
import React from 'react'
import { Group, Circle } from 'react-konva'

type JunctionRecord = Record<string, number[]>
interface Junction {
    id: number
    coord: [number, number]
    buildingId?: number
}

interface Props {
    junctions: JunctionRecord | Junction[]
    frame: number
    junctionCoords: [number, number][]
    externalGrid?: { coord: [number, number] }
    onSelect?: (sel: { type: 'junction'; id: number }) => void
}

const mapColor = (v: number, min: number, max: number) => {
    const t = min === max ? 0.5 : (v - min) / (max - min)
    const r = Math.round(255 * t)
    const g = Math.round(255 * (1 - t))
    return `rgb(${r},${g},0)`
}

export function JunctionsLayer({ junctions, frame, junctionCoords, externalGrid, onSelect }: Props) {
    const pres = Object.values(junctions).map(a => a[frame])
    const pMin = Math.min(...pres)
    const pMax = Math.max(...pres)
    return (
        <Group>
            {Object.entries(junctions).map(([key, arr]) => {
                const idx = parseInt(key.replace('p_bar_junction_', ''), 10)
                const coord = junctionCoords[idx]
                if (!coord) return null
                const [x, y] = coord
                const p = arr[frame]
                const color = mapColor(p, pMin, pMax)
                return (
                    <Circle
                        key={key}
                        x={x}
                        y={y}
                        radius={6}
                        fill={color}
                        onClick={() => onSelect?.({ type: 'junction', id: idx })}
                    />
                )
            })}
            {externalGrid && (
                <Circle
                    x={externalGrid.coord[0]}
                    y={externalGrid.coord[1]}
                    radius={8}
                    fill="blue"
                />
            )}
        </Group>
    )
}
