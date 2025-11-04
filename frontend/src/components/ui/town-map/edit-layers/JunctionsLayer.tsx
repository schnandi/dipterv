'use client'
import React from 'react'
import { Group, Circle } from 'react-konva'

interface Junction {
    id: number
    coord: [number, number]
    buildingId?: number
}

interface Props {
    junctions: Junction[]
    frame?: number
    onSelect?: (sel: { type: 'junction'; id: number }) => void
}

/**
 * Edit-mode Junction Layer
 * - Shows all current draft junctions as blue circles
 * - Building-linked junctions are light blue
 * - Regular junctions are dark blue
 */
export function JunctionsLayer({ junctions, onSelect }: Props) {
    return (
        <Group>
            {junctions.map((j) => (
                <Circle
                    key={j.id}
                    x={j.coord[0]}
                    y={j.coord[1]}
                    radius={6}
                    fill={j.buildingId ? '#0099FF' : '#004080'}
                    stroke={j.buildingId ? '#66CCFF' : undefined}
                    strokeWidth={j.buildingId ? 1.5 : 0}
                    onClick={() => onSelect?.({ type: 'junction', id: j.id })}
                />
            ))}
        </Group>
    )
}
