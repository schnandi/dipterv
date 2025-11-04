'use client'
import React from 'react'
import { Line } from 'react-konva'

interface Props {
    roads: { id: number; start: [number, number]; end: [number, number]; pipe_type?: string }[]
    onSelect?: (sel: { type: 'road'; id: number }) => void
}

/**
 * Renders all roads/pipes for Edit mode.
 * Main pipes are thick and dark, others thinner gray.
 */
export function PipesLayer({ roads, onSelect }: Props) {
    return (
        <>
            {roads.map(r => (
                <Line
                    key={r.id}
                    points={[r.start[0], r.start[1], r.end[0], r.end[1]]}
                    stroke={r.pipe_type === 'main' ? '#000' : '#666'}
                    strokeWidth={r.pipe_type === 'main' ? 3 : 2}
                    onClick={() => onSelect?.({ type: 'road', id: r.id })}
                />
            ))}
        </>
    )
}
