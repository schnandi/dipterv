'use client'

import React, { JSX, useMemo } from 'react'
import { Group, Line, Text } from 'react-konva'

interface GridLayerProps {
    bounds: [number, number, number, number]
    step?: number        // grid cell size in world units
    labelEvery?: number  // draw a label every Nth line
    color?: string       // grid line color
    fontSize?: number
}

/**
 * Renders a faint grey grid with coordinate labels for orientation.
 */
export function GridLayer({
    bounds,
    step = 50,
    labelEvery = 5,
    color = '#ccc',
    fontSize = 10
}: GridLayerProps) {
    const [minX, minY, maxX, maxY] = bounds

    const elements = useMemo(() => {
        const arr: JSX.Element[] = []

        const gridColor = color
        const labelColor = '#888'

        // verticals + X labels
        let vCount = 0
        for (let x = Math.ceil(minX / step) * step; x <= maxX; x += step) {
            arr.push(
                <Line
                    key={`v-${x}`}
                    points={[x, minY, x, maxY]}
                    stroke={gridColor}
                    strokeWidth={1}
                    listening={false}
                />
            )
            if (vCount % labelEvery === 0) {
                arr.push(
                    <Text
                        key={`vx-${x}`}
                        x={x + 2}
                        y={minY + 4}
                        text={x.toFixed(0)}
                        fontSize={fontSize}
                        fill={labelColor}
                        listening={false}
                    />
                )
            }
            vCount++
        }

        // horizontals + Y labels
        let hCount = 0
        for (let y = Math.ceil(minY / step) * step; y <= maxY; y += step) {
            arr.push(
                <Line
                    key={`h-${y}`}
                    points={[minX, y, maxX, y]}
                    stroke={gridColor}
                    strokeWidth={0.5}
                    listening={false}
                />
            )
            if (hCount % labelEvery === 0) {
                arr.push(
                    <Text
                        key={`hy-${y}`}
                        x={minX + 4}
                        y={y + 2}
                        text={y.toFixed(0)}
                        fontSize={fontSize}
                        fill={labelColor}
                        listening={false}
                    />
                )
            }
            hCount++
        }

        return arr
    }, [bounds, step, labelEvery, color, fontSize])

    return <Group>{elements}</Group>
}
