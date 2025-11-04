'use client'

import React, { useMemo } from 'react'
import { Image as KonvaImage } from 'react-konva'

interface TerrainLayerProps {
    heightMap: number[][]
    bounds: [number, number, number, number]
}

/**
 * TerrainLayer — renders a smooth, shaded terrain image from the height map.
 * Produces a more natural gradient and subtle hill shading.
 */
export function TerrainLayer({ heightMap, bounds }: TerrainLayerProps) {
    const image = useMemo(() => {
        const width = heightMap[0].length
        const height = heightMap.length
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')!
        const img = ctx.createImageData(width, height)

        const flat = heightMap.flat()
        const minH = Math.min(...flat)
        const maxH = Math.max(...flat)
        const scale = 1 / (maxH - minH)

        // small helper for color interpolation
        const lerp = (a: number, b: number, t: number) => a + (b - a) * t
        const color = (t: number): [number, number, number] => {
            // smooth gradient: dark brown → green → gray → white
            if (t < 0.25) return [139, 90, 43]      // dirt
            if (t < 0.5) return [60, 180, 75]       // green
            if (t < 0.8) return [150, 150, 150]     // rock
            return [235, 235, 235]                  // snow
        }

        // generate with simple hill-shading
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const h = (heightMap[y][x] - minH) * scale
                const dzdx = (heightMap[y][x + 1] - heightMap[y][x - 1]) * scale
                const dzdy = (heightMap[y + 1][x] - heightMap[y - 1][x]) * scale
                const shade = Math.max(0, Math.min(1, 0.8 - 0.7 * dzdx - 0.4 * dzdy))
                const [r, g, b] = color(h)
                const i = (y * width + x) * 4
                img.data[i] = r * shade
                img.data[i + 1] = g * shade
                img.data[i + 2] = b * shade
                img.data[i + 3] = 255
            }
        }

        ctx.putImageData(img, 0, 0)
        const imgObj = new Image()
        imgObj.src = canvas.toDataURL()
        return imgObj
    }, [heightMap])

    const [minX, minY, maxX, maxY] = bounds
    const wWorld = maxX - minX
    const hWorld = maxY - minY

    return (
        <KonvaImage
            image={image}
            x={minX}
            y={minY}
            width={wWorld}
            height={hWorld}
            listening={false}
        />
    )
}
