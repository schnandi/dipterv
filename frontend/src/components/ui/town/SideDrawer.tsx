'use client'
/* eslint-disable @typescript-eslint/no-explicit-any */
import React from 'react'
import {
    Box,
    VStack,
    HStack,
    Text,
    Button,
    CloseButton,
} from '@chakra-ui/react'
import { api } from '../../../lib/api'
import { useParams } from 'next/navigation'

type GeometryItem = Record<string, any>

interface SimulationItem {
    type: 'road' | 'building' | 'junction'
    id: number
    timestamp: string
    value: number
    flow?: number
    from_junction?: number
    to_junction?: number
    length_m?: number
    diameter_m?: number
    k_mm?: number
    material?: string | null
    max_velocity_m_per_s?: number | null

    // extra building props that may exist in the geometry merge
    district?: string
    building_type?: string
    terrain_height?: number
}

type Item = SimulationItem | GeometryItem | null

interface SideDrawerProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    item: Item
    isSim: boolean
}

/**
 * Floating transparent right-side info panel — full parity with old drawer
 */
export default function SideDrawer({
    open,
    onOpenChange,
    item,
    isSim,
}: SideDrawerProps) {
    const params = useParams()
    const townId = Array.isArray(params.townId) ? params.townId[0] : params.townId!

    const handleAddLeak = async () => {
        if (!item || (item as SimulationItem).type !== 'road') return
        const road = item as SimulationItem & { start: [number, number]; end: [number, number] }
        const [x1, y1] = road.start
        const [x2, y2] = road.end
        const leakJunction: [number, number] = [(x1 + x2) / 2, (y1 + y2) / 2]

        await api.post(`/towns/${townId}/roads/${road.id}/leak`, {
            original_road_id: road.id,
            leak_junction: leakJunction,
            rate_kg_per_s: 0.01,
        })

        onOpenChange(false)
    }

    if (!open || !item) return null

    const simItem = item as SimulationItem
    const isRoad = simItem.type === 'road'
    const isBuilding = simItem.type === 'building'
    const isJunction = simItem.type === 'junction'

    return (
        <Box
            position="fixed"
            top="100px"
            right="30px"
            bg="rgba(255,255,255,0.85)"
            backdropFilter="blur(1px)"
            border="1px solid rgba(0,0,0,0.1)"
            borderRadius="2xl"
            boxShadow="xl"
            p="5"
            w="340px"
            maxH="75vh"
            overflowY="auto"
            zIndex={40}
            transition="all 0.25s ease"
        >
            <HStack justify="space-between" align="center" mb="2">
                <Text fontWeight="bold" fontSize="lg" color="gray.800">
                    {isSim
                        ? isRoad
                            ? `Pipe ${simItem.id}`
                            : isBuilding
                                ? `Building ${simItem.id}`
                                : `Junction ${simItem.id}`
                        : 'Details'}
                </Text>
                <CloseButton onClick={() => onOpenChange(false)} size="sm" />
            </HStack>

            <Box height="1px" bg="gray.300" my="3" />

            <VStack align="stretch" gap="2" fontSize="sm" color="gray.800">
                {isSim ? (
                    <>
                        {/* ⏱ Timestamp and main metric */}
                        <Text><b>Time:</b> {simItem.timestamp}</Text>
                        <Text>
                            <b>
                                {isRoad
                                    ? 'Velocity'
                                    : isBuilding
                                        ? 'Flow'
                                        : 'Pressure'}
                                :
                            </b>{' '}
                            {simItem.value.toFixed(6)}{' '}
                            {isJunction ? 'bar' : isBuilding ? 'kg/s' : 'm/s'}
                        </Text>

                        {/* 💧 Building extra info */}
                        {isBuilding && (
                            <>
                                <Text>
                                    <b>Volumetric:</b>{' '}
                                    {(simItem.value * 3600).toFixed(2)} L/h
                                </Text>

                                {/* 🏘️ Include geometry metadata merged from backend */}
                                {simItem.building_type && (
                                    <Text><b>Type:</b> {simItem.building_type}</Text>
                                )}
                                {simItem.district && (
                                    <Text><b>District:</b> {simItem.district}</Text>
                                )}
                                {typeof simItem.terrain_height === 'number' && (
                                    <Text>
                                        <b>Terrain Height:</b>{' '}
                                        {simItem.terrain_height.toFixed(2)} m
                                    </Text>
                                )}
                            </>
                        )}

                        {/* 🧮 Pipe extra info */}
                        {isRoad && (
                            <>
                                <Text><b>Flow:</b> {simItem.flow?.toFixed(6)} kg/s</Text>
                                <Text><b>From Junction:</b> {simItem.from_junction}</Text>
                                <Text><b>To Junction:</b> {simItem.to_junction}</Text>
                                <Text><b>Length (m):</b> {simItem.length_m?.toFixed(2)}</Text>
                                <Text><b>Diameter (m):</b> {simItem.diameter_m}</Text>
                                <Text><b>Roughness (k_mm):</b> {simItem.k_mm?.toFixed(6)}</Text>
                                {simItem.material && (
                                    <Text><b>Material:</b> {simItem.material}</Text>
                                )}
                                {simItem.max_velocity_m_per_s && (
                                    <Text>
                                        <b>Max Velocity:</b> {simItem.max_velocity_m_per_s} m/s
                                    </Text>
                                )}
                                <Button
                                    mt={3}
                                    size="sm"
                                    colorScheme="red"
                                    onClick={handleAddLeak}
                                >
                                    Add Leak
                                </Button>
                            </>
                        )}
                    </>
                ) : (
                    /* ✏️ Edit mode */
                    Object.entries(item as GeometryItem).map(([key, val]) => (
                        <Text key={key}>
                            <b>{key}:</b> {JSON.stringify(val)}
                        </Text>
                    ))
                )}
            </VStack>
        </Box>
    )
}
