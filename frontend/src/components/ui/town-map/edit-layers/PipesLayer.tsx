'use client'
import React from 'react'
import { Line, Rect, Group } from 'react-konva'

interface Leak {
    coord: [number, number]
    rate_kg_per_s: number
}

interface Props {
    roads: {
        id: number
        start: [number, number]
        end: [number, number]
        pipe_type?: string
        leak?: Leak
    }[]
    onSelect?: (sel: { type: 'road'; id: number }) => void
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

/**
 * Renders all roads/pipes for Edit mode.
 * Main pipes are thick and dark, others thinner gray.
 * Adds leak rectangles for pipes that have leaks.
 */
export function PipesLayer({ roads, onSelect }: Props) {
    return (
        <>
            {roads.map(r => (
                <Group key={r.id}>
                    <Line
                        points={[r.start[0], r.start[1], r.end[0], r.end[1]]}
                        stroke={r.pipe_type === 'main' ? '#000' : '#666'}
                        strokeWidth={r.pipe_type === 'main' ? 3 : 2}
                        onClick={() => onSelect?.({ type: 'road', id: r.id })}
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
                        />
                    )}
                </Group>
            ))}
        </>
    )
}
