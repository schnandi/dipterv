'use client'

import Konva from 'konva'
import React, { useEffect, useState } from 'react'
import { Line, Group, Circle } from 'react-konva'

interface Building {
    id: number
    corners: [number, number][]   // absolute world coords
    center: [number, number]      // precomputed center
    rotation?: number
    district?: string
}

interface Props {
    buildings: Building[]
    editMode: 'addBuilding' | 'delete' | 'addPipe' | null
    sinkFlows: Record<string, number[]>
    frame?: number
    onSelect?: (sel: { type: 'building'; id: number }) => void
    onDragMove?: (id: number, newCorners: [number, number][]) => void
    onDragStart?: () => void
    onDragEnd?: () => void
    selectedId?: number | null

    activeResize: { bId: number; cornerIdx: number } | null
    setActiveResize: React.Dispatch<
        React.SetStateAction<{ bId: number; cornerIdx: number } | null>
    >
    activeRotation: {
        bId: number
        startAngle: number
        baseRotation: number
    } | null
    setActiveRotation: React.Dispatch<
        React.SetStateAction<{
            bId: number
            startAngle: number
            baseRotation: number
            originalCorners: [number, number][]
        } | null>
    >
}

export function BuildingsLayer({
    buildings,
    onSelect,
    onDragMove,
    onDragStart,
    onDragEnd,
    selectedId,
    activeResize,
    setActiveResize,
    activeRotation,
    setActiveRotation,
}: Props) {
    const [hoveredId, setHoveredId] = useState<number | null>(null)
    useEffect(() => {
        const stage = Konva.stages[0]
        if (stage) {
            stage.batchDraw()
        }
    }, [buildings.length])
    return (
        <>
            {buildings.map((b) => {
                const isSelected = b.id === selectedId
                const fill =
                    b.district === 'residential'
                        ? '#90CAF9'
                        : b.district === 'industrial'
                            ? '#F28B82'
                            : '#BDBDBD'

                // ✅ Points relative to center (for Konva group rotation)
                const localPoints = b.corners
                    .map(([x, y]) => [x - b.center[0], y - b.center[1]])
                    .flat()

                return (
                    <Group
                        key={b.id}
                        x={b.center[0]}
                        y={b.center[1]}
                        rotation={b.rotation ?? 0}
                        draggable={!activeResize && !activeRotation}
                        onClick={() => onSelect?.({ type: 'building', id: b.id })}
                        onDragStart={(e) => {
                            e.cancelBubble = true
                            onDragStart?.()
                        }}
                        onDragMove={(e) => {
                            if (!onDragMove) return
                            const { x, y } = e.target.position()
                            const dx = x - b.center[0]
                            const dy = y - b.center[1]
                            const moved = b.corners.map(([cx, cy]) => [cx + dx, cy + dy]) as [
                                number,
                                number
                            ][]
                            onDragMove(b.id, moved)
                            // reset transform offset so next move is relative
                            e.target.x(b.center[0])
                            e.target.y(b.center[1])
                        }}
                        onDragEnd={() => onDragEnd?.()}
                    >
                        {/* Building shape */}
                        <Line
                            points={localPoints}
                            closed
                            fill={fill}
                            stroke={
                                isSelected
                                    ? '#0078D4'
                                    : b.id === hoveredId
                                        ? '#00BFFF'
                                        : '#444'
                            }
                            strokeWidth={isSelected ? 3 : 1.5}
                            opacity={0.95}
                            onMouseEnter={() => setHoveredId(b.id)}
                            onMouseLeave={() => setHoveredId(null)}
                        />

                        {/* Selection handles */}
                        {isSelected && (
                            <>
                                {/* Corner resize handles */}
                                {b.corners.map(([x, y], idx) => (
                                    <Circle
                                        key={idx}
                                        x={x - b.center[0]}
                                        y={y - b.center[1]}
                                        radius={6}
                                        fill="white"
                                        stroke="#0078D4"
                                        strokeWidth={2}
                                        onMouseDown={(e) => {
                                            e.cancelBubble = true
                                            onDragStart?.()
                                            setActiveResize({ bId: b.id, cornerIdx: idx })
                                        }}
                                    />
                                ))}

                                {/* Rotation handle above top-right corner */}
                                {(() => {
                                    const [x1, y1] = b.corners[1]
                                    const offset = 40
                                    const handleX = x1 - b.center[0]
                                    const handleY = y1 - b.center[1] - offset
                                    return (
                                        <>
                                            <Line
                                                points={[handleX, handleY + offset, handleX, handleY]}
                                                stroke="#0078D4"
                                                strokeWidth={1.5}
                                                listening={false}
                                            />
                                            <Circle
                                                x={handleX}
                                                y={handleY}
                                                radius={6}
                                                fill="#FFA500"
                                                stroke="white"
                                                strokeWidth={1.5}
                                                onMouseDown={(e) => {
                                                    e.cancelBubble = true
                                                    const stage = e.target.getStage()
                                                    const pointer = stage?.getPointerPosition()
                                                    if (!pointer) return
                                                    const scale = stage!.scaleX()
                                                    const stagePos = stage!.position()
                                                    const worldX = (pointer.x - stagePos.x) / scale
                                                    const worldY = (pointer.y - stagePos.y) / scale
                                                    const startAngle = Math.atan2(
                                                        worldY - b.center[1],
                                                        worldX - b.center[0]
                                                    )
                                                    setActiveRotation({
                                                        bId: b.id,
                                                        startAngle,
                                                        baseRotation: b.rotation ?? 0,
                                                        originalCorners: b.corners,
                                                    } as any)
                                                }}
                                            />
                                        </>
                                    )
                                })()}
                            </>
                        )}
                    </Group>
                )
            })}
        </>
    )
}
