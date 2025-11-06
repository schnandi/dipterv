'use client'

import React, { useState } from 'react'
import { Layer, Line, Rect, Circle } from 'react-konva'

interface Junction {
    id: number
    coord: [number, number]
    buildingId?: number
}

interface Pipe {
    id: number
    start: [number, number]
    end: [number, number]
    startJunctionId?: number
    endJunctionId?: number
    pipe_type?: string
    age: number
}

interface Props {
    tool: 'addBuilding' | 'addPipe' | 'delete' | null
    draftBuildings: any[]
    draftPipes: Pipe[]
    draftJunctions: Junction[]
    setDraftBuildings: React.Dispatch<React.SetStateAction<any[]>>
    setDraftPipes: React.Dispatch<React.SetStateAction<Pipe[]>>
    setDraftJunctions: React.Dispatch<React.SetStateAction<Junction[]>>
    buildingConfig: {
        size: number
        rotation: number
        buildingType: string
    }
    pipeConfig: {
        pipe_type: string
        age: number
    }
    getDistrictForType: (type: string) => string
    onToolDone?: () => void
    onSelect?: (sel: { type: 'road' | 'building' | 'junction'; id: number } | null) => void
}

export default function EditOverlay({
    tool,
    draftBuildings,
    draftPipes,
    draftJunctions,
    setDraftBuildings,
    setDraftPipes,
    setDraftJunctions,
    buildingConfig,
    onToolDone,
    getDistrictForType,
    pipeConfig,
    onSelect
}: Props) {
    const [draftPipe, setDraftPipe] = useState<{
        start: [number, number]
        end: [number, number]
        startJunctionId?: number
        endJunctionId?: number
    } | null>(null)

    const SNAP_RADIUS = 25

    const generateId = () => Date.now() + Math.floor(Math.random() * 100000);

    /** 🔍 Find nearest junction or pipe projection */
    const findSnapTarget = (
        x: number,
        y: number
    ): { type: 'junction' | 'pipe'; coord: [number, number]; id: number; t?: number } | null => {
        let nearest: { type: 'junction' | 'pipe'; coord: [number, number]; id: number; t?: number } | null = null
        let minDist = SNAP_RADIUS

        // 1️⃣ Check junctions
        for (const j of draftJunctions) {
            const dx = j.coord[0] - x
            const dy = j.coord[1] - y
            const dist = Math.sqrt(dx * dx + dy * dy)
            if (dist < SNAP_RADIUS * 0.6) {
                return { type: 'junction', coord: j.coord, id: j.id }
            }
            if (dist < minDist) {
                minDist = dist
                nearest = { type: 'junction', coord: j.coord, id: j.id }
            }
        }

        // 2️⃣ Check pipes
        for (const p of draftPipes) {
            const [x1, y1] = p.start
            const [x2, y2] = p.end
            const dx = x2 - x1
            const dy = y2 - y1
            const lenSq = dx * dx + dy * dy
            if (lenSq === 0) continue
            const t = Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / lenSq))
            const proj: [number, number] = [x1 + t * dx, y1 + t * dy]
            const dist = Math.hypot(proj[0] - x, proj[1] - y)
            if (dist < minDist) {
                minDist = dist
                nearest = { type: 'pipe', coord: proj, id: p.id, t }
            }
        }

        return nearest
    }

    const handleClick = (e: any) => {
        if (!tool) return
        const stage = e.target.getStage()
        const pointer = stage.getPointerPosition()
        const x = (pointer.x - stage.x()) / stage.scaleX()
        const y = (pointer.y - stage.y()) / stage.scaleY()

        /** 🏠 Add Building */
        if (tool === 'addBuilding') {
            const { size, rotation, buildingType } = buildingConfig
            const district = getDistrictForType(buildingType)
            const id = generateId()
            const corners: [number, number][] = [
                [x - size / 2, y - size / 2],
                [x + size / 2, y - size / 2],
                [x + size / 2, y + size / 2],
                [x - size / 2, y + size / 2],
            ]
            const rounded: [number, number] = [Number(x.toFixed(3)), Number(y.toFixed(3))];
            const newBuilding = {
                id,
                corners,
                rotation,
                district,
                building_type: buildingType,
                center: rounded,
            };
            const newJunction: Junction = { id: id + 1, coord: rounded, buildingId: id };

            setDraftBuildings(prev => [...prev, newBuilding])
            setDraftJunctions(prev => [...prev, newJunction])
            onSelect?.({ type: 'building', id })
            onToolDone?.()
            return
        }

        if (tool === 'delete') {
            const toDelete: number[] = [];

            for (const b of draftBuildings) {
                const [cx, cy] = b.center;
                const rotation = ((b.rotation ?? 0) * Math.PI) / 180;
                const cos = Math.cos(-rotation);
                const sin = Math.sin(-rotation);

                // Convert click to local coordinates (relative to building)
                const dx = x - cx;
                const dy = y - cy;
                const localX = dx * cos - dy * sin;
                const localY = dx * sin + dy * cos;

                // Compute building extents in local space
                const localCorners = b.corners.map(([wx, wy]: [number, number]): [number, number] => {
                    const dx = wx - cx;
                    const dy = wy - cy;
                    const lx = dx * cos - dy * sin;
                    const ly = dx * sin + dy * cos;
                    return [lx, ly];
                });
                const minX = Math.min(...localCorners.map((p: any[]) => p[0]));
                const maxX = Math.max(...localCorners.map((p: any[]) => p[0]));
                const minY = Math.min(...localCorners.map((p: any[]) => p[1]));
                const maxY = Math.max(...localCorners.map((p: any[]) => p[1]));

                // Check if click is inside
                if (localX >= minX && localX <= maxX && localY >= minY && localY <= maxY) {
                    toDelete.push(b.id);
                }
            }

            if (toDelete.length > 0) {
                setDraftBuildings(prev => prev.filter(b => !toDelete.includes(b.id)));
                setDraftJunctions(prev => prev.filter(j => !toDelete.includes(j.buildingId ?? -1)));
                setDraftPipes(prev =>
                    prev.filter(
                        p =>
                            !toDelete.includes(-1) &&
                            !toDelete.includes(-1)
                    )
                );
            }

            return;
        }

        /** ➕ Add Pipe */
        if (tool === 'addPipe') {
            const snap = findSnapTarget(x, y);

            if (!draftPipe) {
                // --- Start pipe ---
                if (snap?.type === 'pipe') {
                    const newJ: Junction = { id: generateId(), coord: snap.coord };
                    setDraftJunctions(prev => [...prev, newJ]);

                    setDraftPipes(prev => {
                        const updated: Pipe[] = [];
                        for (const p of prev) {
                            if (p.id === snap.id) {
                                // split into two new pipes
                                const left = {
                                    ...p,
                                    end: snap.coord,
                                    endJunctionId: newJ.id,
                                };
                                const right = {
                                    ...p,
                                    id: generateId(),
                                    start: snap.coord,
                                    startJunctionId: newJ.id,
                                };
                                updated.push(left, right);
                            } else {
                                updated.push(p);
                            }
                        }
                        return updated;
                    });

                    setDraftPipe({
                        start: newJ.coord,
                        end: newJ.coord,
                        startJunctionId: newJ.id,
                    });
                } else {
                    // not snapping to existing pipe
                    setDraftPipe({
                        start: snap ? snap.coord : [x, y],
                        end: [x, y],
                        startJunctionId: snap?.type === 'junction' ? snap.id : undefined,
                    });
                }
            } else {
                // --- Finish pipe ---
                let endCoord: [number, number] = [x, y];
                let endJunctionId: number | undefined;

                if (snap?.type === 'pipe') {
                    const newJ: Junction = { id: generateId(), coord: snap.coord };
                    setDraftJunctions(prev => [...prev, newJ]);

                    setDraftPipes(prev => {
                        const updated: Pipe[] = [];
                        for (const p of prev) {
                            if (p.id === snap.id) {
                                const left = {
                                    ...p,
                                    end: snap.coord,
                                    endJunctionId: newJ.id,
                                };
                                const right = {
                                    ...p,
                                    id: generateId(),
                                    start: snap.coord,
                                    startJunctionId: newJ.id,
                                };
                                updated.push(left, right);
                            } else {
                                updated.push(p);
                            }
                        }
                        return updated;
                    });

                    endCoord = newJ.coord;
                    endJunctionId = newJ.id;
                } else if (snap?.type === 'junction') {
                    endCoord = snap.coord;
                    endJunctionId = snap.id;

                    // If this junction belongs to a building, snap that building's center
                    const linked = draftJunctions.find(j => j.id === snap.id && j.buildingId);
                    if (linked?.buildingId) {
                        setDraftBuildings(prev =>
                            prev.map(b =>
                                b.id === linked.buildingId ? { ...b, center: snap.coord } : b
                            )
                        );
                    }
                } else {
                    const newJ: Junction = { id: generateId(), coord: [x, y] };
                    setDraftJunctions(prev => [...prev, newJ]);
                    endCoord = newJ.coord;
                    endJunctionId = newJ.id;
                }

                const newPipe: Pipe = {
                    id: generateId(),
                    start: draftPipe.start,
                    end: endCoord,
                    startJunctionId: draftPipe.startJunctionId,
                    endJunctionId,
                    pipe_type: pipeConfig.pipe_type,
                    age: pipeConfig.age,
                };

                setDraftPipes(prev => [...prev, newPipe]);
                setDraftPipe(null);
                onToolDone?.();
            }
        }
    }

    const handleMouseMove = (e: any) => {
        if (tool !== 'addPipe' || !draftPipe) return
        const stage = e.target.getStage()
        const pointer = stage.getPointerPosition()
        const x = (pointer.x - stage.x()) / stage.scaleX()
        const y = (pointer.y - stage.y()) / stage.scaleY()
        const snap = findSnapTarget(x, y)

        setDraftPipe(prev =>
            prev
                ? {
                    ...prev,
                    end: snap ? snap.coord : ([x, y] as [number, number]),
                    endJunctionId: snap?.type === 'junction' ? snap.id : undefined,
                }
                : null
        )
    }

    return (
        <Layer onClick={handleClick} onMouseMove={handleMouseMove} listening={!!tool}>
            {/* Invisible hit area so clicks anywhere register */}
            <Rect
                x={-5000}
                y={-5000}
                width={10000}
                height={10000}
                fill="transparent"
                listening={true}
            />

            {/* Pipe preview */}
            {draftPipe && (
                <>
                    <Line
                        points={[
                            draftPipe.start[0],
                            draftPipe.start[1],
                            draftPipe.end[0],
                            draftPipe.end[1],
                        ]}
                        stroke="gray"
                        strokeWidth={3}
                        dash={[8, 4]}
                    />
                    <Circle
                        x={draftPipe.start[0]}
                        y={draftPipe.start[1]}
                        radius={6}
                        fill="lightblue"
                        stroke="blue"
                        strokeWidth={1}
                    />
                    <Circle
                        x={draftPipe.end[0]}
                        y={draftPipe.end[1]}
                        radius={6}
                        fill="limegreen"
                        stroke="gray"
                        strokeWidth={1}
                    />
                </>
            )}
        </Layer>
    )
}
