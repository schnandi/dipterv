'use client'

import React, { useRef, useEffect } from 'react'
import { Box, AspectRatio } from '@chakra-ui/react'

interface Road {
    id: number
    start: [number, number]
    end: [number, number]
    pipe_type?: string
}

interface Building {
    id: number
    corners: [number, number][]
    district?: string
}

interface TownMapEditProps {
    roads: Road[]
    buildings: Building[]
    junctions: Record<string, number[]>
    frame: number
    onSelect: (sel: { type: 'road' | 'building' | 'junction'; id: number }) => void
}

function pointInPolygon(x: number, y: number, vs: [number, number][]) {
    let inside = false
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        const [xi, yi] = vs[i]
        const [xj, yj] = vs[j]
        const intersect =
            (yi > y) !== (yj > y) &&
            x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
        if (intersect) inside = !inside
    }
    return inside
}

function distToSegment(
    px: number, py: number,
    x1: number, y1: number,
    x2: number, y2: number
) {
    const dx = x2 - x1, dy = y2 - y1
    const l2 = dx * dx + dy * dy
    if (l2 === 0) return Math.hypot(px - x1, py - y1)
    let t = ((px - x1) * dx + (py - y1) * dy) / l2
    t = Math.max(0, Math.min(1, t))
    const projX = x1 + t * dx, projY = y1 + t * dy
    return Math.hypot(px - projX, py - projY)
}

export default function TownMapEdit({
    roads,
    buildings,
    junctions,
    frame,
    onSelect,
}: TownMapEditProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const offset = useRef({ x: 0, y: 0 })
    const scale = useRef(1)
    const isPanning = useRef(false)
    const panStart = useRef({ x: 0, y: 0 })

    const junctionListRef = useRef<[number, number][]>([])
    useEffect(() => {
        const seen = new Set<string>()
        const list: [number, number][] = []
        roads.forEach(r => {
            for (const pt of [r.start, r.end] as const) {
                const key = pt.join(',')
                if (!seen.has(key)) {
                    seen.add(key)
                    list.push(pt)
                }
            }
        })
        junctionListRef.current = list
    }, [roads])

    useEffect(() => {
        const canvas = canvasRef.current!
        const ctx = canvas.getContext('2d')!

        function draw() {
            const { width, height } = canvas
            ctx.resetTransform()
            ctx.clearRect(0, 0, width, height)
            ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height)

            // pan & zoom
            ctx.translate(offset.current.x, offset.current.y)
            ctx.scale(scale.current, scale.current)
            ctx.translate(0, height); ctx.scale(1, -1)

            // pipes (plain)
            roads.forEach(r => {
                ctx.strokeStyle = r.pipe_type === 'main' ? '#000' : '#666'
                ctx.lineWidth = r.pipe_type === 'main' ? 3 : 1
                ctx.beginPath()
                ctx.moveTo(r.start[0], r.start[1])
                ctx.lineTo(r.end[0], r.end[1])
                ctx.stroke()
            })

            // junctions (plain)
            Object.keys(junctions).forEach(key => {
                const idx = parseInt(key.replace('p_bar_junction_', ''), 10)
                const coord = junctionListRef.current[idx]
                if (!coord) return
                const [x, y] = coord
                ctx.fillStyle = '#004080'
                ctx.beginPath()
                ctx.arc(x, y, 4, 0, 2 * Math.PI)
                ctx.fill()
            })

            // buildings (plain)
            buildings.forEach(b => {
                let fill = '#999'
                if (b.district === 'residential') fill = 'lightblue'
                else if (b.district === 'industrial') fill = 'lightcoral'
                ctx.fillStyle = fill
                ctx.beginPath()
                b.corners.forEach(([x, y], i) =>
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
                )
                ctx.closePath()
                ctx.fill()
            })
        }

        // pan & zoom handlers
        const onMouseDown = (e: MouseEvent) => {
            isPanning.current = true
            panStart.current = { x: e.clientX - offset.current.x, y: e.clientY - offset.current.y }
        }
        const onMouseMove = (e: MouseEvent) => {
            if (isPanning.current) {
                offset.current.x = e.clientX - panStart.current.x
                offset.current.y = e.clientY - panStart.current.y
                draw()
            }
        }
        const onMouseUp = () => { isPanning.current = false }
        const onWheel = (e: WheelEvent) => {
            e.preventDefault()
            const prev = scale.current
            const delta = -e.deltaY * 0.001
            scale.current = Math.max(0.3, Math.min(5, scale.current * (1 + delta)))
            const rect = canvas.getBoundingClientRect()
            const mx = e.clientX - rect.left, my = e.clientY - rect.top
            offset.current.x = mx - (mx - offset.current.x) * (scale.current / prev)
            offset.current.y = my - (my - offset.current.y) * (scale.current / prev)
            draw()
        }

        canvas.addEventListener('mousedown', onMouseDown)
        window.addEventListener('mousemove', onMouseMove)
        window.addEventListener('mouseup', onMouseUp)
        canvas.addEventListener('wheel', onWheel, { passive: false })

        draw()

        return () => {
            canvas.removeEventListener('mousedown', onMouseDown)
            window.removeEventListener('mousemove', onMouseMove)
            window.removeEventListener('mouseup', onMouseUp)
            canvas.removeEventListener('wheel', onWheel)
        }
    }, [roads, buildings, junctions, frame])

    return (
        <Box w="100%" h="100%" position="absolute">
            <AspectRatio ratio={1} w="100%">
                <canvas
                    ref={canvasRef}
                    width={2000}
                    height={2000}
                    style={{ width: '100%', height: '100%' }}
                    onClick={e => {
                        const canvas = canvasRef.current!
                        const rect = canvas.getBoundingClientRect()
                        const px = ((e.clientX - rect.left) / rect.width) * canvas.width
                        const py = ((e.clientY - rect.top) / rect.height) * canvas.height
                        const x1 = (px - offset.current.x) / scale.current
                        const y1 = (py - offset.current.y) / scale.current
                        const x = x1, y = canvas.height - y1

                        // junction hit?
                        for (const [idx, [jx, jy]] of junctionListRef.current.entries()) {
                            if (Math.hypot(x - jx, y - jy) < 5) {
                                return onSelect({ type: 'junction', id: idx })
                            }
                        }
                        // building hit?
                        for (const b of buildings) {
                            if (pointInPolygon(x, y, b.corners)) {
                                return onSelect({ type: 'building', id: b.id })
                            }
                        }
                        // road hit?
                        for (const r of roads) {
                            if (distToSegment(x, y, r.start[0], r.start[1], r.end[0], r.end[1]) < 5) {
                                return onSelect({ type: 'road', id: r.id })
                            }
                        }
                    }}
                />
            </AspectRatio>
        </Box>
    )
}
