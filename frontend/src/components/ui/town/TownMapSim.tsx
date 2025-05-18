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

interface ExternalGrid {
    junction: number
    coord: [number, number]
}

interface PumpInfo {
    pump_index: number
    from_junction: number
    to_junction: number
    from_coord: [number, number]
    to_coord: [number, number]
}

interface TownMapSimProps {
    roads: Road[]
    buildings: Building[]
    junctions: Record<string, number[]>
    pipeVelocities: Record<string, number[]>
    pipeFlows: Record<string, number[]>
    sinkFlows: Record<string, number[]>
    frame: number
    mapSize: [number, number]
    heightMapResolution: [number, number]
    heightMap: number[][]
    heightMapBounds: [number, number, number, number]
    onSelect: (sel: { type: 'road' | 'building' | 'junction'; id: number }) => void
    externalGrid: ExternalGrid
    pumps: PumpInfo[]
    showTerrain: boolean,
    showFlow: boolean,
}

// hit‐testing helpers
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

// normalize [min,max] → rgb(green→red)
function mapColor(v: number, min: number, max: number) {
    const t = min === max ? 0.5 : (v - min) / (max - min)
    const r = Math.round(255 * t)
    const g = Math.round(255 * (1 - t))
    return `rgb(${r},${g},0)`
}

// draw arrow in middle of pipe indicating flow direction
function drawVelocityArrow(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number,
    x2: number, y2: number,
    vel: number,
    color: string
) {
    const dx = x2 - x1, dy = y2 - y1
    const len = Math.hypot(dx, dy)
    if (!len) return
    let ux = dx / len, uy = dy / len
    if (vel < 0) { ux = -ux; uy = -uy }
    const midX = (x1 + x2) / 2, midY = (y1 + y2) / 2
    const arrowLen = 10, arrowWid = 8
    const p1: [number, number] = [midX, midY]
    const p2: [number, number] = [
        midX - ux * arrowLen - uy * arrowWid,
        midY - uy * arrowLen + ux * arrowWid
    ]
    const p3: [number, number] = [
        midX - ux * arrowLen + uy * arrowWid,
        midY - uy * arrowLen - ux * arrowWid
    ]
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.moveTo(...p1)
    ctx.lineTo(...p2)
    ctx.lineTo(...p3)
    ctx.closePath()
    ctx.fill()
}

export default function TownMapSim({
    roads,
    buildings,
    junctions,
    pipeVelocities,
    sinkFlows,
    pipeFlows,
    frame,
    mapSize,
    heightMapResolution,
    heightMap,
    heightMapBounds,
    onSelect,
    externalGrid,
    pumps,
    showTerrain,
    showFlow
}: TownMapSimProps) {
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
        const [minX, minY, maxX, maxY] = heightMapBounds
        const widthWorld = maxX - minX
        const heightWorld = maxY - minY
        const canvas = canvasRef.current!
        const ctx = canvas.getContext('2d')!

        // 1) Build offscreen buffer from heightMap
        const [resX, resY] = heightMapResolution
        const buffer = document.createElement('canvas')
        buffer.width = resX
        buffer.height = resY
        const bufCtx = buffer.getContext('2d')!

        // Create ImageData
        const img = bufCtx.createImageData(resX, resY)
        const data = img.data

        // Find max height for normalization
        const flat = heightMap.flat()
        const maxH = Math.max(...flat)

        const dirt = [139, 69, 19]   // deep brown
        const grass = [34, 139, 34]   // green
        const rock = [128, 128, 128]   // gray-brown rock
        const snow = [245, 245, 245]   // pale snow

        // define our normalized thresholds (on the gamma-corrected height)
        const t1 = 0.3   // up to 30%  → dirt→grass
        const t2 = 0.6   // 30–60%      → grass→rock
        const t3 = 0.9   // 60–90%      → rock→snow
        // 90–100%        → pure snow

        for (let y = 0; y < resY; y++) {
            for (let x = 0; x < resX; x++) {
                const h = heightMap[y][x] / maxH      // raw [0..1]
                const h2 = Math.pow(h, 1.2)            // gamma-corrected

                let c1: number[], c2: number[], t: number

                if (h2 < t1) {
                    // dirt → grass
                    c1 = dirt; c2 = grass; t = h2 / t1
                }
                else if (h2 < t2) {
                    // grass → rock
                    c1 = grass; c2 = rock; t = (h2 - t1) / (t2 - t1)
                }
                else if (h2 < t3) {
                    // rock → snow
                    c1 = rock; c2 = snow; t = (h2 - t2) / (t3 - t2)
                }
                else {
                    // solid snow
                    c1 = snow; c2 = snow; t = 0
                }

                const idx = (y * resX + x) * 4
                data[idx + 0] = Math.round(c1[0] + (c2[0] - c1[0]) * t)
                data[idx + 1] = Math.round(c1[1] + (c2[1] - c1[1]) * t)
                data[idx + 2] = Math.round(c1[2] + (c2[2] - c1[2]) * t)
                data[idx + 3] = 255
            }
        }

        bufCtx.putImageData(img, 0, 0)

        function draw() {
            const { width, height } = canvas
            ctx.resetTransform()
            ctx.clearRect(0, 0, width, height)
            ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height)

            // pan & zoom
            ctx.translate(offset.current.x, offset.current.y)
            ctx.scale(scale.current, scale.current)
            ctx.translate(0, height); ctx.scale(1, -1)

            if (showTerrain) {
                ctx.drawImage(
                    buffer,
                  /* dx */ minX, /* dy */ minY,
                  /* dWidth */ widthWorld, /* dHeight */ heightWorld
                )
            }

            // split buildings by district
            const resBuildings = buildings.filter(b => b.district === 'residential');
            const indBuildings = buildings.filter(b => b.district === 'industrial');

            // get their current sink‐flows
            const resVals = resBuildings.map(b =>
                (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame]
            );
            const indVals = indBuildings.map(b =>
                (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame]
            );

            // compute residential min/max (fallback to [0,1] if none)
            // and industrial min/max
            const rMin = resVals.length ? Math.min(...resVals) : 0;
            const rMax = resVals.length ? Math.max(...resVals) : 1;
            const iMin = indVals.length ? Math.min(...indVals) : 0;
            const iMax = indVals.length ? Math.max(...indVals) : 1;
            const noResData = rMin === rMax;
            const noIndData = iMin === iMax;

            // compute min/max & detect uniform data
            const pres = Object.values(junctions).map(a => a[frame])
            const pMin = Math.min(...pres), pMax = Math.max(...pres)
            const noPresData = pMin === pMax

            const velGroups: Record<string, number[]> = {
                main: [],
                side: [],
                'building connection': []
            };

            roads.forEach(r => {
                const arr = pipeVelocities[`v_mean_pipe_${r.id}`] || [];
                arr.forEach(v => {
                    velGroups[r.pipe_type ?? 'side'].push(Math.abs(v));
                });
            });

            const vMinGroup: Record<string, number> = {};
            const vMaxGroup: Record<string, number> = {};
            Object.entries(velGroups).forEach(([type, vals]) => {
                vMinGroup[type] = vals.length ? Math.min(...vals) : 0;
                vMaxGroup[type] = vals.length ? Math.max(...vals) : 1;
            });


            const allFlows = roads.flatMap(r =>
                (pipeFlows[`flow_pipe_${r.id}`] || []).map(f => Math.abs(f))
            );
            const fMin = allFlows.length ? Math.min(...allFlows) : 0;
            const fMax = allFlows.length ? Math.max(...allFlows) : 1;





            // draw pipes
            roads.forEach(r => {
                const type = r.pipe_type ?? "side";
                const vel = (pipeVelocities[`v_mean_pipe_${r.id}`] || [0])[frame];
                let color: string;

                if (showFlow) {
                    const flow = (pipeFlows[`flow_pipe_${r.id}`] || [0])[frame];
                    color = fMin === fMax
                        ? "#666"
                        : mapColor(Math.abs(flow), fMin, fMax);
                } else {
                    // colour by abs(velocity)
                    const vmin = vMinGroup[type], vmax = vMaxGroup[type];
                    color = (vmin === vmax)
                        ? "#666"
                        : mapColor(Math.abs(vel), vmin, vmax);
                }

                ctx.strokeStyle = color;
                ctx.lineWidth = type === "main" ? 5 : type === "side" ? 2 : 1;
                ctx.beginPath();
                ctx.moveTo(r.start[0], r.start[1]);
                ctx.lineTo(r.end[0], r.end[1]);
                ctx.stroke();

                // always draw arrow from signed velocity
                drawVelocityArrow(
                    ctx,
                    r.start[0], r.start[1],
                    r.end[0], r.end[1],
                    vel,
                    color
                );
            });

            //highlight pump locations
            pumps.forEach((p) => {
                const [x1, y1] = p.from_coord
                const [x2, y2] = p.to_coord
                ctx.strokeStyle = 'yellow'
                ctx.lineWidth = 4
                ctx.beginPath()
                ctx.moveTo(x1, y1)
                ctx.lineTo(x2, y2)
                ctx.stroke()
            })

            // draw junctions
            Object.entries(junctions).forEach(([key, arr]) => {
                const idx = parseInt(key.replace('p_bar_junction_', ''), 10)
                const coord = junctionListRef.current[idx]
                if (!coord) return
                const [x, y] = coord
                const p = arr[frame]
                ctx.fillStyle = noPresData
                    ? '#666'
                    : mapColor(p, pMin, pMax)
                ctx.beginPath()
                ctx.arc(x, y, 6, 0, 2 * Math.PI)
                ctx.fill()
            })

            // 4) mark the external grid junction in blue
            if (externalGrid) {
                const [ex, ey] = externalGrid.coord
                ctx.fillStyle = 'blue'
                ctx.beginPath()
                ctx.arc(ex, ey, 8, 0, 2 * Math.PI)
                ctx.fill()
            }

            // draw buildings
            buildings.forEach(b => {
                // raw flow value for this building
                const raw = (sinkFlows[`mdot_sink_${b.id}`] ?? [0])[frame];

                // pick which min/max to use
                const isRes = b.district === 'residential';
                const [mini, maxi, empty] = isRes
                    ? [rMin, rMax, noResData]
                    : [iMin, iMax, noIndData];

                // map to color
                const fill = empty
                    ? '#999'
                    : mapColor(raw, mini, maxi);

                // actually draw it
                ctx.fillStyle = fill;
                ctx.beginPath();
                b.corners.forEach(([x, y], i) =>
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
                );
                ctx.closePath();
                ctx.fill();

                ctx.lineWidth = 2
                ctx.strokeStyle = isRes ? '#00A5E3' /*blue*/ : '#FF60A8' /*orange*/
                ctx.stroke()
            });
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
    }, [roads, buildings, junctions, pipeVelocities, sinkFlows, pipeFlows, frame, externalGrid, pumps, heightMap, heightMapResolution, mapSize, heightMapBounds, showTerrain, showFlow])

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
                            if (Math.hypot(x - jx, y - jy) < 6) {
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
                            if (
                                distToSegment(
                                    x, y,
                                    r.start[0], r.start[1],
                                    r.end[0], r.end[1]
                                ) < 5
                            ) {
                                return onSelect({ type: 'road', id: r.id })
                            }
                        }
                    }}
                />
            </AspectRatio>
        </Box>
    )
}
