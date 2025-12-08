'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Stage, Layer } from 'react-konva'
import * as EditLayers from './edit-layers'
import * as SimLayers from './sim-layers'
import EditOverlay from './edit-tools/EditOverlay'
import Toolbox from './edit-tools/Toolbox'
import BuildingSettingsPanel from './edit-tools/BuildingSettingsPanel'
import PipeSettingsPanel from './edit-tools/PipeSettingsPanel'
import dynamic from 'next/dynamic'
import { api } from '@/lib/api'
import BurstRiskPanel from './sim-tools/BurstRiskPanel'
import PipeSearchPanel from './sim-tools/PipeSearchPanel'
import LeakRegionLayer from './sim-layers/LeakRegionLayer'

const SideDrawer = dynamic(() => import('@/components/ui/town/SideDrawer'), {
    ssr: false,
})


export interface Road {
    id: number
    start: [number, number]
    end: [number, number]
    pipe_type?: string
    startBuildingId?: number  // optional link to buildings
    endBuildingId?: number
    startJunctionId?: number
    endJunctionId?: number
    age: number
}
export interface Building {
    id: number
    corners: [number, number][]
    center: [number, number]
    district?: string
    building_type?: string
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

export interface LeakRegion {
    pipe_id: number;
    center: [number, number];
    radius: number;
}

export interface TrilaterationPipe {
    pipe_id: number;
    x: number;
    y: number;
    r: number;
    probability: number;
}

export interface TriangulatedRegion {
    leak_point: [number, number];
    uncertainty_radius: number;
    supporting_pipes: TrilaterationPipe[];
}

export interface LeakRiskResponse {
    town_id: number;
    baseline_simulation_id: number;
    current_simulation_id: number;

    best_circle: LeakRegion;
    triangulated: TriangulatedRegion;

    top_pipes: any[];
    all_pipes: any[];

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
    showFlow?: boolean,
    pipeParameters?: Record<string, any>
    timestamps?: string[]
    hasLeakSimulation?: boolean;
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
    pipeParameters,
    timestamps,
    hasLeakSimulation
}: TownCanvasProps) {
    const [editMode, setEditMode] = useState<'addBuilding' | 'addPipe' | 'delete' | null>(null)
    const [selected, setSelected] = useState<{ type: 'road' | 'building' | 'junction'; id: number } | null>(null)
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

    const [leakDataV1, setLeakDataV1] = useState<LeakRiskResponse | null>(null);
    const [leakDataV2, setLeakDataV2] = useState<LeakRiskResponse | null>(null);

    useEffect(() => {
        async function loadBoth() {
            const id = Number(window.location.pathname.split("/").pop());

            const [v1, v2] = await Promise.all([
                api.get(`/leak-risk/${id}`).catch(() => null),
                api.get(`/leak-risk/v2/${id}`).catch(() => null),
            ]);

            if (v1?.data) setLeakDataV1(v1.data);
            if (v2?.data) setLeakDataV2(v2.data);
        }

        if (mode === "simulate" && hasLeakSimulation) {
            loadBoth();
        } else {
            setLeakDataV1(null);
            setLeakDataV2(null);
        }
    }, [mode, hasLeakSimulation]);



    const [highlightedPipeId, setHighlightedPipeId] = useState<number | null>(null);

    const [isPanning, setIsPanning] = useState(false)
    const stageRef = useRef<any>(null);
    const isDraggingBuilding = useRef(false);

    const drawerItem = useMemo(() => {
        if (!selected) return null

        // 🧮 Simulate mode: live values
        if (mode === 'simulate' && frame != null) {
            // ⚙️ Just extract timestamp array if available (we assume all have same length)
            const tsArray =
                pipeVelocities?.[Object.keys(pipeVelocities)[0]] ??
                sinkFlows?.[Object.keys(sinkFlows)[0]] ??
                junctions?.[Object.keys(junctions)[0]]

            const ts = tsArray ? `${timestamps![frame]}` : ''

            if (selected.type === 'road') {
                const vel = pipeVelocities?.[`v_mean_pipe_${selected.id}`]?.[frame] ?? 0
                const flow = pipeFlows?.[`flow_pipe_${selected.id}`]?.[frame] ?? 0
                const staticInfo = pipeParameters?.[selected.id] || {}   // ✅ restore static data
                const geo = draftPipes.find(r => r.id === selected.id) || {}
                return {
                    type: 'road',
                    id: selected.id,
                    timestamp: ts,
                    value: vel,
                    flow,
                    ...staticInfo,
                    ...geo,
                }
            }

            if (selected.type === 'building') {
                const sink = sinkFlows?.[`mdot_sink_${selected.id}`]?.[frame] ?? 0
                const geo = draftBuildings.find(b => b.id === selected.id) || {}
                return {
                    type: 'building',
                    id: selected.id,
                    timestamp: ts,
                    value: sink,
                    ...geo,
                }
            }

            if (selected.type === 'junction') {
                const pressure = junctions?.[`p_bar_junction_${selected.id}`]?.[frame] ?? 0
                const geo = draftJunctions.find(j => j.id === selected.id) || {}
                return {
                    type: 'junction',
                    id: selected.id,
                    timestamp: ts,
                    value: pressure,
                    ...geo,
                }
            }
        }

        if (mode === 'edit') {
            if (selected.type === 'road') {
                return draftPipes.find(r => r.id === selected.id)
            } else if (selected.type === 'building') {
                return draftBuildings.find(b => b.id === selected.id)
            } else if (selected.type === 'junction') {
                return draftJunctions.find(j => j.id === selected.id)
            }
        }


        return null
    }, [
        selected,
        draftBuildings,
        draftPipes,
        draftJunctions,
        mode,
        frame,
        pipeVelocities,
        pipeFlows,
        sinkFlows,
        junctions,
    ])


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
        buildingType: 'single_family',
    })

    const [pipeConfig, setPipeConfig] = useState({
        pipe_type: 'main',
        age: 0,
    })

    const getDistrictForType = (type: string): string =>
        ['factory', 'warehouse', 'processing_plant'].includes(type)
            ? 'industrial'
            : 'residential'

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

    function validateAndCleanTownData(data: { junctions: any[]; roads: any[]; buildings: any[] }) {
        const eps = 1e-3;

        // 🔹 Helper: round coordinates to nearest 0.001 to prevent float mismatches
        const roundCoord = (pt: [number, number]): [number, number] => [
            Number(pt[0].toFixed(3)),
            Number(pt[1].toFixed(3)),
        ];

        // 1️⃣ Merge duplicate junctions
        const uniqueJunctions: any[] = [];
        const junctionMap = new Map(); // oldId -> newId

        data.junctions.forEach(j => {
            // round coords first to reduce near-duplicates
            j.coord = roundCoord(j.coord);
            const existing = uniqueJunctions.find(
                u => Math.hypot(u.coord[0] - j.coord[0], u.coord[1] - j.coord[1]) < eps
            );
            if (existing) {
                junctionMap.set(j.id, existing.id);
            } else {
                uniqueJunctions.push(j);
                junctionMap.set(j.id, j.id);
            }
        });

        // 2️⃣ Rewire pipes to merged junction IDs
        const cleanedPipes = data.roads
            .map(p => {
                const start = roundCoord(p.start);
                const end = roundCoord(p.end);
                return {
                    ...p,
                    start,
                    end,
                    startJunctionId: junctionMap.get(p.startJunctionId) ?? p.startJunctionId,
                    endJunctionId: junctionMap.get(p.endJunctionId) ?? p.endJunctionId,
                };
            })
            .filter(p => p.startJunctionId !== undefined && p.endJunctionId !== undefined);

        // 3️⃣ Remove unconnected junctions
        const usedJunctions = new Set();
        cleanedPipes.forEach(p => {
            usedJunctions.add(p.startJunctionId);
            usedJunctions.add(p.endJunctionId);
        });
        data.buildings.forEach(b => {
            const j = data.junctions.find(j => j.buildingId === b.id);
            if (j) usedJunctions.add(j.id);
        });
        const connectedJunctions = uniqueJunctions.filter(j => usedJunctions.has(j.id));

        // 4️⃣ Deduplicate pipes (same endpoints)
        const uniquePipes: any[] = [];
        for (const p of cleanedPipes) {
            const dup = uniquePipes.find(
                u =>
                    (u.startJunctionId === p.startJunctionId && u.endJunctionId === p.endJunctionId) ||
                    (u.startJunctionId === p.endJunctionId && u.endJunctionId === p.startJunctionId)
            );
            if (!dup) uniquePipes.push(p);
        }

        // 5️⃣ Ensure every building has a junction
        const finalJunctions = [...connectedJunctions];
        data.buildings.forEach(b => {
            // round center
            b.center = roundCoord(b.center);
            const linked = finalJunctions.find(j => j.buildingId === b.id);
            if (!linked) {
                const cx = b.corners.reduce((s: any, c: any[]) => s + c[0], 0) / b.corners.length;
                const cy = b.corners.reduce((s: any, c: any[]) => s + c[1], 0) / b.corners.length;
                const newJ = { id: Date.now() + b.id, coord: roundCoord([cx, cy]), buildingId: b.id };
                finalJunctions.push(newJ);
            }
        });

        // 6️⃣ Final rounding pass to ensure clean export
        finalJunctions.forEach(j => (j.coord = roundCoord(j.coord)));
        uniquePipes.forEach(p => {
            p.start = roundCoord(p.start);
            p.end = roundCoord(p.end);
        });

        return {
            ...data,
            roads: uniquePipes,
            junctions: finalJunctions,
        };
    }

    function focusOnPipe(pipeId: number) {
        const pipe = draftPipes.find(p => p.id === pipeId);
        if (!pipe) {
            return;
        }

        if (!stageRef.current) {
            return;
        }

        const stage = stageRef.current.getStage?.() || stageRef.current;

        const midX = (pipe.start[0] + pipe.end[0]) / 2;
        const midY = (pipe.start[1] + pipe.end[1]) / 2;
        const zoom = 1.4;

        stage.scale({ x: zoom, y: zoom });
        stage.position({
            x: stage.width() / 2 - midX * zoom,
            y: stage.height() / 2 - midY * zoom,
        });
        stage.batchDraw();
        stage.getLayers().forEach((layer: { draw: () => any }) => layer.draw());
        stage.draw();
    }


    return (
        <>
            <Stage
                ref={stageRef}
                width={window.innerWidth}
                height={window.innerHeight}
                draggable={isPanning && !isDraggingBuilding.current} // 🚫 stage can't move when building dragging
                onWheel={handleWheel}
                onMouseDown={(e) => {
                    const stage = e.target.getStage();
                    if (e.target === stage) setSelected?.(null);
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
                        onSelect={setSelected}
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
                            onSelect={setSelected}
                        />
                    ) : (
                        <SimLayers.JunctionsLayer
                            junctions={junctions}
                            junctionCoords={junctionCoords}
                            frame={frame}
                            externalGrid={externalGrid}
                            onSelect={setSelected}
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
                        onSelect={setSelected}
                    />
                </Layer>
                <Layer listening={false}>
                    {/* V1 model (current one, red/blue/orange) */}
                    {mode === "simulate" && hasLeakSimulation && leakDataV1 && (
                        <LeakRegionLayer
                            bestCircle={leakDataV1.best_circle}
                            triangulated={leakDataV1.triangulated}
                            showSupporting={true}
                        />
                    )}

                    {/* V2 model (we render it in PURPLE so you can compare) */}
                    {mode === "simulate" && hasLeakSimulation && leakDataV2 && (
                        <LeakRegionLayer
                            bestCircle={{
                                ...leakDataV2.best_circle,
                                colorOverride: "purple",   // NEW
                            } as any}
                            triangulated={{
                                ...leakDataV2.triangulated,
                                colorOverride: "purple",   // NEW
                            } as any}
                            showSupporting={true}
                        />
                    )}
                </Layer>
                {highlightedPipeId !== null && (
                    <Layer listening={false}>
                        {(() => {
                            const pipe = draftPipes.find(p => p.id === highlightedPipeId);
                            if (!pipe) return null;
                            return (
                                <SimLayers.HighlightedPipe
                                    key={pipe.id}
                                    start={pipe.start}
                                    end={pipe.end}
                                />
                            );
                        })()}
                    </Layer>
                )}

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
                        pipeConfig={pipeConfig}
                        getDistrictForType={getDistrictForType}
                        onToolDone={() => setEditMode(null)}
                        onSelect={setSelected}
                    />
                )}



            </Stage>
            {mode === 'simulate' && (
                <>
                    <PipeSearchPanel
                        onGoToPipe={(pipeId) => {
                            setSelected({ type: 'road', id: pipeId })
                            focusOnPipe(pipeId)
                            setHighlightedPipeId(pipeId)
                            setTimeout(() => setHighlightedPipeId(null), 2500)
                        }}
                    />
                    <BurstRiskPanel
                        townId={Number(window.location.pathname.split('/').pop())}
                        onSelectPipe={(pipeId) => {
                            setSelected({ type: 'road', id: pipeId })
                            focusOnPipe(pipeId)
                            setHighlightedPipeId(pipeId)
                            setTimeout(() => setHighlightedPipeId(null), 2500)
                        }}
                    />
                </>
            )}
            {mode === 'edit' && (
                <Toolbox
                    currentTool={editMode}
                    onSelect={(tool) => {
                        setEditMode(tool);
                        setSelected(null);
                    }}
                    onSave={async () => {
                        try {
                            const params = window.location.pathname.split('/');
                            const townId = params[params.indexOf('towns') + 1];

                            const cleaned = validateAndCleanTownData({
                                roads: draftPipes,
                                buildings: draftBuildings,
                                junctions: draftJunctions,
                            });

                            const res = await api.put(`/towns/${townId}/data`, cleaned);

                            alert('Town saved successfully! Previous simulations were removed.');

                            const event = new CustomEvent('town-updated', { detail: { townId } });
                            window.dispatchEvent(event);
                        } catch (err) {
                            console.error('❌ Failed to save town:', err);
                            alert('Failed to save town data.');
                        }
                    }}
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
                    buildingType={buildingConfig.buildingType}
                    onChange={(values) => setBuildingConfig({ ...buildingConfig, ...values })}
                />
            )}
            {mode === 'edit' && editMode === 'addPipe' && (
                <PipeSettingsPanel
                    pipe_type={pipeConfig.pipe_type}
                    age={pipeConfig.age}
                    onChange={(values) => setPipeConfig({ ...pipeConfig, ...values })}
                />
            )}
            {drawerItem && (
                <SideDrawer
                    open={!!selected}
                    onOpenChange={() => setSelected(null)}
                    item={drawerItem}
                    isSim={mode === 'simulate'}
                    onUpdateBuilding={(id, values) =>
                        setDraftBuildings(prev =>
                            prev.map(b => {
                                if (b.id !== id) return b;

                                const updated = { ...b, ...values };

                                // 🧮 Automatically recalc district if building_type changes
                                if ('building_type' in values && values.building_type) {
                                    const newDistrict = getDistrictForType(values.building_type);
                                    return { ...updated, district: newDistrict };
                                }

                                return updated;
                            })
                        )
                    }
                    onUpdatePipe={(id, values) =>
                        setDraftPipes(prev => prev.map(p => p.id === id ? { ...p, ...values } : p))
                    }
                />
            )}
        </>
    )

}
