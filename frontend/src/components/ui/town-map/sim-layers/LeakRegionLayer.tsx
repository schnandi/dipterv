"use client";

import React from "react";
import { Circle, Group, Line, Text } from "react-konva";

export interface TrilaterationPipe {
    pipe_id: number;
    x: number;
    y: number;
    r: number;
    probability: number;
}

export interface LeakRegionProps {
    bestCircle: {
        pipe_id: number;
        center: [number, number];
        radius: number;
        colorOverride?: string;
    } | null;

    triangulated: {
        leak_point: [number, number];
        uncertainty_radius: number;
        supporting_pipes: TrilaterationPipe[];
        colorOverride?: string;
    } | null;

    showSupporting?: boolean;
}

/**
 * Draw ML leak prediction regions:
 * - Simple best-pipe circle (red)
 * - Triangulated point (blue)
 * - Uncertainty zone (blue translucent)
 * - Optional: Supporting circles (orange)
 */
export default function LeakRegionLayer({
    bestCircle,
    triangulated,
    showSupporting = true,
}: LeakRegionProps) {
    if (!bestCircle && !triangulated) return null;

    const color = bestCircle?.colorOverride || "red";
    const triColor = triangulated?.colorOverride || "blue";


    return (
        <Group listening={false}>
            {/* ------------------------- */}
            {/* 1. Simple ML circle       */}
            {/* ------------------------- */}
            {bestCircle && (
                <Group>
                    <Circle
                        x={bestCircle.center[0]}
                        y={bestCircle.center[1]}
                        radius={bestCircle.radius}
                        stroke={color}
                        strokeWidth={2}
                        opacity={0.7}
                    />
                    <Text
                        x={bestCircle.center[0] + bestCircle.radius + 8}
                        y={bestCircle.center[1]}
                        text={`Pipe ${bestCircle.pipe_id}`}
                        fontSize={16}
                        fill={color}
                    />
                </Group>
            )}

            {/* ------------------------- */}
            {/* 2. Trilaterated center    */}
            {/* ------------------------- */}
            {triangulated && (
                <Group>
                    {/* Estimated leak location */}
                    <Circle
                        x={triangulated.leak_point[0]}
                        y={triangulated.leak_point[1]}
                        radius={8}
                        fill={triColor}
                        opacity={0.9}
                    />

                    {/* Uncertainty area */}
                    <Circle
                        x={triangulated.leak_point[0]}
                        y={triangulated.leak_point[1]}
                        radius={triangulated.uncertainty_radius}
                        stroke={triColor}
                        strokeWidth={2}
                        opacity={0.3}
                        dash={[10, 10]}
                    />

                    <Text
                        x={triangulated.leak_point[0] + 12}
                        y={triangulated.leak_point[1] - 5}
                        text="Leak Estimate"
                        fontSize={16}
                        fill={triColor}
                    />

                    {/* Optional supporting circles */}
                    {showSupporting &&
                        triangulated.supporting_pipes.map((p, idx) => (
                            <Group key={idx}>
                                <Circle
                                    x={p.x}
                                    y={p.y}
                                    radius={p.r}
                                    stroke={triColor}
                                    strokeWidth={1}
                                    opacity={0.35}
                                />
                                <Circle
                                    x={p.x}
                                    y={p.y}
                                    radius={4}
                                    fill={triColor}
                                />
                                <Text
                                    x={p.x + 5}
                                    y={p.y + 5}
                                    text={`P${p.pipe_id}`}
                                    fontSize={12}
                                    fill={triColor}
                                />
                            </Group>
                        ))}
                </Group>
            )}
        </Group>
    );
}
