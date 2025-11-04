'use client'

import React, { useMemo, useRef, useState } from 'react'
import { Stage, Layer } from 'react-konva'
import * as EditLayers from './edit-layers'
import * as SimLayers from './sim-layers'
import EditOverlay from './edit-tools/EditOverlay'
import Toolbox from './edit-tools/Toolbox'
import BuildingSettingsPanel from './edit-tools/BuildingSettingsPanel'


interface Road {
    id: number
    start: [number, number]
    end: [number, number]
    pipe_type?: string
    startBuildingId?: number  // optional link to buildings
    endBuildingId?: number
    startJunctionId?: number
    endJunctionId?: number
}
interface Building {
    id: number
    corners: [number, number][]
    center: [number, number]
    district?: string
    rotation?: number
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
interface Junction {
    id: number
    coord: [number, number]
    buildingId?: number       // optional link to the building that owns it
}

export interface TownCanvasProps {
    mode: 'view' | 'edit' | 'simulate'
    roads: Road[]
    buildings: Building[]
    junctions: Record<string, number[]>
    pipeVelocities?: Record<string, number[]>
    pipeFlows?: Record<string, number[]>
    sinkFlows?: Record<string, number[]>
    frame?: number
    mapSize?: [number, number]
    heightMap?: number[][]
    heightMapBounds?: [number, number, number, number]
    externalGrid?: ExternalGrid
    pumps?: PumpInfo[]
    showTerrain?: boolean
    showFlow?: boolean
    onSelect?: (sel: { type: 'road' | 'building' | 'junction'; id: number } | null) => void
    selected?: { type: 'road' | 'building' | 'junction'; id: number } | null
}

/**
 * Unified map component for both Edit and Simulate modes
 */
export default function TownCanvas({
    mode,
    roads,
    buildings,
    junctions,
    pipeVelocities,
    pipeFlows,
    sinkFlows,
    frame = 0,
    mapSize,
    heightMap,
    heightMapBounds,
    externalGrid,
    pumps,
    showTerrain = true,
    showFlow = false,
    onSelect,
    selected
}: TownCanvasProps) {
    const [editMode, setEditMode] = useState<'addBuilding' | 'addPipe' | 'delete' | null>(null)
    const [draftBuildings, setDraftBuildings] = useState(buildings)
    React.useEffect(() => {
        setDraftBuildings(
            buildings.map(b => {
                const cx = b.corners.reduce((sum, c) => sum + c[0], 0) / b.corners.length
                const cy = b.corners.reduce((sum, c) => sum + c[1], 0) / b.corners.length
                return { ...b, center: [cx, cy] }
            })
        )
    }, [buildings])
    const [draftPipes, setDraftPipes] = useState(roads)
    const [draftJunctions, setDraftJunctions] = useState<Junction[]>([])
    const [activeResize, setActiveResize] = useState<{ bId: number; cornerIdx: number } | null>(null);
    const [activeRotation, setActiveRotation] = useState<{
        bId: number
        startAngle: number
        baseRotation: number
        originalCorners: [number, number][]
    } | null>(null)


    const [isPanning, setIsPanning] = useState(false)
    const stageRef = useRef<any>(null);
    const isDraggingBuilding = useRef(false);


    const currentBuildings = mode === 'edit' ? draftBuildings : buildings
    const currentPipes = mode === 'edit' ? draftPipes : roads

    const L = mode === 'simulate' ? SimLayers : EditLayers


    const [scale, setScale] = useState(1)
    const [pos, setPos] = useState({ x: 0, y: 0 })

    const handleWheel = (e: any) => {
        e.evt.preventDefault()
        const stage = e.target.getStage()
        const oldScale = stage.scaleX()
        const pointer = stage.getPointerPosition()
        const scaleBy = 1.08
        const direction = e.evt.deltaY > 0 ? -1 : 1
        const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy

        const mousePointTo = {
            x: (pointer.x - stage.x()) / oldScale,
            y: (pointer.y - stage.y()) / oldScale,
        }

        stage.scale({ x: newScale, y: newScale })
        const newPos = {
            x: pointer.x - mousePointTo.x * newScale,
            y: pointer.y - mousePointTo.y * newScale,
        }
        stage.position(newPos)
        stage.batchDraw()
    }


    const [mapW, mapH] = mapSize || [2000, 2000]

    // Adjust canvas viewport, not world units
    const stageWidth = window.innerWidth
    const stageHeight = window.innerHeight

    const junctionCoords = useMemo(() => {
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
        return list
    }, [roads])

    React.useEffect(() => {
        // 1️⃣ Build base junctions from roads
        const baseJunctions: Junction[] = junctionCoords.map((coord, idx): Junction => ({
            id: idx,
            coord: coord as [number, number],
            buildingId: undefined,
        }));

        // 2️⃣ Link each building to nearest existing junction or create a new one
        const linked = [...baseJunctions];
        buildings.forEach(b => {
            const cx = b.corners.reduce((sum, c) => sum + c[0], 0) / b.corners.length;
            const cy = b.corners.reduce((sum, c) => sum + c[1], 0) / b.corners.length;
            const center: [number, number] = [cx, cy];

            let nearest = null;
            let minDist = Infinity;

            for (const j of linked) {
                const dx = j.coord[0] - cx;
                const dy = j.coord[1] - cy;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = j;
                }
            }

            if (nearest && minDist < 30) {
                nearest.buildingId = b.id;
            } else {
                linked.push({
                    id: 10_000 + b.id,
                    coord: center,
                    buildingId: b.id,
                });
            }
        });

        // 3️⃣ Now enrich pipes with start/end junction IDs (based on proximity)
        const withJunctionIds = roads.map(p => {
            const findNearest = (pt: [number, number]) => {
                let nearestId: number | undefined = undefined;
                let minDist = Infinity;
                for (const j of linked) {
                    const dx = j.coord[0] - pt[0];
                    const dy = j.coord[1] - pt[1];
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestId = j.id;
                    }
                }
                return nearestId;
            };

            return {
                ...p,
                startJunctionId: findNearest(p.start),
                endJunctionId: findNearest(p.end),
            };
        });

        // 4️⃣ Commit both to state
        setDraftJunctions(linked);
        setDraftPipes(withJunctionIds);
    }, [junctionCoords, buildings, roads]);

    const [buildingConfig, setBuildingConfig] = useState({
        size: 40,
        rotation: 0,
        district: 'residential'
    })

    const handleBuildingDrag = (id: number, newCorners: [number, number][]) => {
        // Compute new building center
        const cx = newCorners.reduce((sum, c) => sum + c[0], 0) / newCorners.length
        const cy = newCorners.reduce((sum, c) => sum + c[1], 0) / newCorners.length
        const newCenter: [number, number] = [cx, cy]
        const buildingJunction = draftJunctions.find(j => j.buildingId === id);
        if (!buildingJunction) return;

        // 1️⃣ Update building corners + center
        setDraftBuildings(prev =>
            prev.map(b =>
                b.id === id ? { ...b, corners: newCorners, center: newCenter } : b
            )
        )

        // 2️⃣ Update junction position for this building
        setDraftJunctions(prev => {
            // find old junction linked to this building
            const updated = prev.map(j =>
                j.buildingId === id ? { ...j, coord: newCenter } : j
            )
            return updated
        })

        // 3️⃣ Update all pipes connected to this building *after junctions moved*
        setDraftPipes(prev =>
            prev.map(p => {
                if (p.startBuildingId === id || p.startJunctionId === buildingJunction.id)
                    return { ...p, start: newCenter };
                if (p.endBuildingId === id || p.endJunctionId === buildingJunction.id)
                    return { ...p, end: newCenter };
                return p;
            })
        );
    }

    const handleDragStart = () => {
        isDraggingBuilding.current = true;
        setIsPanning(false);
    }

    const handleDragEnd = () => {
        isDraggingBuilding.current = false;
    }

    return (
        <>
            <Stage
                ref={stageRef}
                width={window.innerWidth}
                height={window.innerHeight}
                scaleX={scale}
                scaleY={scale}
                x={pos.x}
                y={pos.y}
                draggable={isPanning && !isDraggingBuilding.current} // 🚫 stage can't move when building dragging
                onWheel={handleWheel}
                onMouseDown={(e) => {
                    const stage = e.target.getStage();
                    if (e.target === stage) onSelect?.(null);
                    const isBackground = e.target === stage;
                    const enablePanning = isBackground && !isDraggingBuilding.current;
                    setIsPanning(enablePanning);
                    stage!.draggable(enablePanning);

                    if (enablePanning) {
                        stage.container().style.cursor = 'grabbing';
                    }
                }}
                onMouseUp={(e) => {
                    const stage = e.target.getStage();
                    stage!.draggable(false);
                    setIsPanning(false);
                    stage!.container().style.cursor = 'default';
                    setActiveResize(null);
                    setActiveRotation(null);
                    isDraggingBuilding.current = false;
                }}
                onDragMove={(e) => {
                    if (isPanning && !isDraggingBuilding.current) {
                        const newPos = e.target.position();
                        setPos({ x: newPos.x, y: newPos.y });
                    }
                }}
                onDragEnd={(e) => {
                    // ✅ Only update pos if we were panning
                    if (isPanning && !isDraggingBuilding.current) {
                        const newPos = e.target.position();
                        setPos({ x: newPos.x, y: newPos.y });
                    }
                    setIsPanning(false);
                }}
                onMouseMove={(e) => {
                    const stage = e.target.getStage();
                    const pointer = stage?.getPointerPosition();
                    if (!pointer) return;

                    const scale = stage!.scaleX();
                    const stagePos = stage!.position();
                    const worldX = (pointer.x - stagePos.x) / scale;
                    const worldY = (pointer.y - stagePos.y) / scale;

                    // ✅ handle resize globally
                    // ✅ handle resize globally
                    if (activeResize) {
                        const { bId, cornerIdx } = activeResize;
                        const b = draftBuildings.find(b => b.id === bId);
                        if (!b) return;

                        // --- Compute building center ---
                        const cx = b.corners.reduce((s, c) => s + c[0], 0) / 4;
                        const cy = b.corners.reduce((s, c) => s + c[1], 0) / 4;

                        // --- Rotation helpers ---
                        const rotation = (b.rotation ?? 0) * Math.PI / 180;
                        const cos = Math.cos(-rotation);
                        const sin = Math.sin(-rotation);

                        // --- Convert world mouse coords → local building space ---
                        const dx = worldX - cx;
                        const dy = worldY - cy;
                        const lx = dx * cos - dy * sin;
                        const ly = dx * sin + dy * cos;

                        // --- Compute new half width/height ---
                        const newHalfW = Math.abs(lx);
                        const newHalfH = Math.abs(ly);

                        // --- New rectangle corners in local coordinates ---
                        const newLocal: [number, number][] = [
                            [-newHalfW, -newHalfH],
                            [newHalfW, -newHalfH],
                            [newHalfW, newHalfH],
                            [-newHalfW, newHalfH],
                        ];

                        // --- Rotate back to world space ---
                        const newCorners = newLocal.map(([x, y]) => [
                            cx + x * Math.cos(rotation) - y * Math.sin(rotation),
                            cy + x * Math.sin(rotation) + y * Math.cos(rotation),
                        ]) as [number, number][];

                        // --- Apply new geometry ---
                        handleBuildingDrag(bId, newCorners);
                    }


                    if (activeRotation) {
                        const { bId, startAngle, baseRotation } = activeRotation
                        const b = draftBuildings.find(b => b.id === bId)
                        if (!b) return

                        const [cx, cy] = b.center

                        const dx = worldX - cx
                        const dy = worldY - cy
                        const currentAngle = Math.atan2(dy, dx)

                        let delta = currentAngle - startAngle
                        if (delta > Math.PI) delta -= 2 * Math.PI
                        if (delta < -Math.PI) delta += 2 * Math.PI

                        const newRotation = baseRotation + (delta * 180) / Math.PI

                        setDraftBuildings(prev =>
                            prev.map(bb =>
                                bb.id === bId ? { ...bb, rotation: newRotation } : bb
                            )
                        )
                    }
                }}
            >
                {showTerrain && heightMap && heightMapBounds ? (
                    <Layer listening={false}>
                        <L.TerrainLayer heightMap={heightMap} bounds={heightMapBounds} />
                    </Layer>
                ) : (
                    <Layer listening={false}>
                        <L.GridLayer bounds={heightMapBounds ?? [0, 0, 2000, 2000]} step={100} />
                    </Layer>
                )}
                <Layer listening={true}>
                    <L.BuildingsLayer
                        buildings={currentBuildings}
                        sinkFlows={sinkFlows ?? {}}
                        frame={frame ?? 0}
                        editMode={editMode}
                        onSelect={onSelect}
                        onDragMove={handleBuildingDrag}
                        onDragStart={handleDragStart}
                        onDragEnd={handleDragEnd}
                        selectedId={selected?.type === 'building' ? selected.id : null}
                        activeResize={activeResize}
                        setActiveResize={setActiveResize}
                        activeRotation={activeRotation}
                        setActiveRotation={setActiveRotation}
                    />

                </Layer>

                <Layer listening={true}>
                    {mode === 'edit' ? (
                        <EditLayers.JunctionsLayer
                            junctions={draftJunctions}
                            onSelect={onSelect}
                        />
                    ) : (
                        <SimLayers.JunctionsLayer
                            junctions={junctions}
                            junctionCoords={junctionCoords}
                            frame={frame}
                            externalGrid={externalGrid}
                            onSelect={onSelect}
                        />
                    )}
                </Layer>
                <Layer listening={true}>
                    <L.PipesLayer
                        roads={currentPipes}
                        frame={frame}
                        pipeVelocities={pipeVelocities}
                        pipeFlows={pipeFlows}
                        showFlow={showFlow}
                        onSelect={onSelect}
                    />
                </Layer>

                {mode === 'edit' && (
                    <EditOverlay
                        tool={editMode}
                        draftBuildings={draftBuildings}
                        draftPipes={draftPipes}
                        setDraftBuildings={setDraftBuildings}
                        setDraftPipes={setDraftPipes}
                        draftJunctions={draftJunctions}
                        setDraftJunctions={setDraftJunctions}
                        buildingConfig={buildingConfig}
                        onToolDone={() => setEditMode(null)}
                    />
                )}



            </Stage>
            {mode === 'edit' && (
                <Toolbox
                    currentTool={editMode}
                    onSelect={setEditMode}
                    onSave={() => console.log('TODO: send to backend', { draftBuildings, draftPipes })}
                    onReset={() => {
                        // 1️⃣ Reset drafts to original input
                        setDraftBuildings(buildings);
                        setDraftPipes(roads);

                        // 2️⃣ Rebuild junctions exactly like useEffect does
                        const baseJunctions: Junction[] = [];
                        const seen = new Set<string>();

                        roads.forEach((r, idx) => {
                            [r.start, r.end].forEach((coord) => {
                                const key = coord.join(',');
                                if (!seen.has(key)) {
                                    seen.add(key);
                                    baseJunctions.push({ id: idx * 2 + baseJunctions.length, coord });
                                }
                            });
                        });

                        const linked = [...baseJunctions];
                        buildings.forEach(b => {
                            const cx = b.corners.reduce((sum, c) => sum + c[0], 0) / b.corners.length;
                            const cy = b.corners.reduce((sum, c) => sum + c[1], 0) / b.corners.length;
                            const center: [number, number] = [cx, cy];

                            let nearest = null;
                            let minDist = Infinity;

                            for (const j of linked) {
                                const dx = j.coord[0] - cx;
                                const dy = j.coord[1] - cy;
                                const dist = Math.sqrt(dx * dx + dy * dy);
                                if (dist < minDist) {
                                    minDist = dist;
                                    nearest = j;
                                }
                            }

                            if (nearest && minDist < 30) {
                                nearest.buildingId = b.id;
                            } else {
                                linked.push({
                                    id: 10_000 + b.id,
                                    coord: center,
                                    buildingId: b.id,
                                });
                            }
                        });

                        setDraftJunctions(linked);

                        // 3️⃣ Reset tool and interaction state
                        setActiveResize(null);
                        setActiveRotation(null);
                        setEditMode(null);
                    }}
                />
            )}
            {mode === 'edit' && editMode === 'addBuilding' && (
                <BuildingSettingsPanel
                    size={buildingConfig.size}
                    rotation={buildingConfig.rotation}
                    district={buildingConfig.district}
                    onChange={(values) => setBuildingConfig({ ...buildingConfig, ...values })}
                />
            )}
        </>
    )

}
