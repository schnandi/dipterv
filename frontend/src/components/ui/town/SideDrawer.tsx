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
    NativeSelect,
    Input,
} from '@chakra-ui/react'
import { api } from '../../../lib/api'
import { useParams } from 'next/navigation'
import { Building, Road } from '../town-map/TownCanvas'

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
    age?: number | null
    pipe_type?: string | null
    mean_pressure_bar?: number | null
    std_pressure_bar?: number | null
    k_mm?: number
    material?: string | null
    max_velocity_m_per_s?: number | null
    district?: string
    building_type?: string
    terrain_height?: number
}

type Item = SimulationItem | Building | Road | null

interface SideDrawerProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    item: Item
    isSim: boolean
    onUpdateBuilding?: (id: number, values: Partial<Building>) => void
    onUpdatePipe?: (id: number, values: Partial<Road>) => void
}

/**
 * Floating transparent right-side info panel — full parity with old drawer
 */
export default function SideDrawer({
    open,
    onOpenChange,
    item,
    isSim,
    onUpdateBuilding,
    onUpdatePipe,
}: SideDrawerProps) {
    const params = useParams()
    const townId = Array.isArray(params.townId) ? params.townId[0] : params.townId!

    const handleAddLeak = async () => {
        if (!item || (item as SimulationItem).type !== 'road') return
        const road = item as SimulationItem & { start: [number, number]; end: [number, number] }
        const [x1, y1] = road.start
        const [x2, y2] = road.end
        const leakJunction: [number, number] = [(x1 + x2) / 2, (y1 + y2) / 2]

        const rate = getLeakRate(leakSize)

        try {
            await api.post(`/towns/${townId}/roads/${road.id}/leak`, {
                original_road_id: road.id,
                leak_junction: leakJunction,
                rate_kg_per_s: rate,
            })

            //trigger UI refresh just like edit save
            const event = new CustomEvent('town-updated', { detail: { townId } })
            window.dispatchEvent(event)

            //optional feedback
            alert('Leak added successfully!')

            onOpenChange(false)
        } catch (err) {
            console.error('Failed to add leak:', err)
            alert('Failed to add leak.')
        }
    }

    function getLeakRate(size: typeof leakSize): number {
        switch (size) {
            case 'small': return 0.005
            case 'medium': return 0.08
            case 'big': return 0.2
            case 'very_big': return 5.0
            default: return 0.01
        }
    }

    if (!open || !item) return null

    const isSimItem = (obj: any): obj is SimulationItem => 'timestamp' in obj
    const isBuilding = (obj: any): obj is Building => 'corners' in obj
    const isRoad = (obj: any): obj is Road => 'start' in obj && 'end' in obj
    const isJunction = (obj: any): obj is { id: number; coord: [number, number] } =>
        'coord' in obj && !('start' in obj)

    const [leakSize, setLeakSize] = React.useState<'small' | 'medium' | 'big' | 'very_big'>('medium')

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
                        ? isSimItem(item)
                            ? item.type === 'road'
                                ? `Pipe ${item.id}`
                                : item.type === 'building'
                                    ? `Building ${item.id}`
                                    : `Junction ${item.id}`
                            : 'Details'
                        : 'Details'}
                </Text>
                <CloseButton onClick={() => onOpenChange(false)} size="sm" />
            </HStack>

            <Box height="1px" bg="gray.300" my="3" />

            <VStack align="stretch" gap="2" fontSize="sm" color="gray.800">
                {isSim && isSimItem(item) ? (
                    <>
                        {/*Timestamp and main metric */}
                        <Text>
                            <b>Time:</b> {item.timestamp}
                        </Text>
                        <Text>
                            <b>
                                {item.type === 'road'
                                    ? 'Velocity'
                                    : item.type === 'building'
                                        ? 'Flow'
                                        : 'Pressure'}
                                :
                            </b>{' '}
                            {item.value.toFixed(6)}{' '}
                            {item.type === 'junction'
                                ? 'bar'
                                : item.type === 'building'
                                    ? 'kg/s'
                                    : 'm/s'}
                        </Text>

                        {/*Building extra info */}
                        {item.type === 'building' && (
                            <>
                                <Text>
                                    <b>Volumetric:</b> {(item.value * 3600).toFixed(2)} L/h
                                </Text>
                                {item.building_type && <Text><b>Type:</b> {item.building_type}</Text>}
                                {item.district && <Text><b>District:</b> {item.district}</Text>}
                                {typeof item.terrain_height === 'number' && (
                                    <Text>
                                        <b>Terrain Height:</b> {item.terrain_height.toFixed(2)} m
                                    </Text>
                                )}
                            </>
                        )}

                        {/*Pipe extra info */}
                        {item.type === 'road' && (
                            <>
                                <Text><b>Flow:</b> {item.flow?.toFixed(6)} kg/s</Text>
                                <Text><b>From Junction:</b> {item.from_junction}</Text>
                                <Text><b>To Junction:</b> {item.to_junction}</Text>
                                <Text><b>Length (m):</b> {item.length_m?.toFixed(2)}</Text>
                                <Text><b>Diameter (m):</b> {item.diameter_m}</Text>
                                <Text><b>Roughness (k_mm):</b> {item.k_mm?.toFixed(6)}</Text>
                                {item.pipe_type && <Text><b>Pipe Type:</b> {item.pipe_type}</Text>}
                                {typeof item.age === 'number' && <Text><b>Age:</b> {item.age} years</Text>}

                                {item.material && <Text><b>Material:</b> {item.material}</Text>}
                                {item.max_velocity_m_per_s && (
                                    <Text>
                                        <b>Max Velocity:</b> {item.max_velocity_m_per_s} m/s
                                    </Text>
                                )}
                                {typeof item.mean_pressure_bar === 'number' && (
                                    <Text><b>Mean Pressure:</b> {item.mean_pressure_bar.toFixed(3)} bar</Text>
                                )}
                                {typeof item.std_pressure_bar === 'number' && (
                                    <Text><b>Pressure Std Dev:</b> {item.std_pressure_bar.toFixed(3)} bar</Text>
                                )}
                                <VStack align="stretch" mt={3}>
                                    <Text fontWeight="semibold" fontSize="sm">Leak Size</Text>
                                    <NativeSelect.Root size="sm">
                                        <NativeSelect.Field
                                            value={leakSize}
                                            onChange={(e) => setLeakSize(e.target.value as 'small' | 'medium' | 'big' | 'very_big')}
                                        >
                                            <option value="small">Small (0.005 kg/s)</option>
                                            <option value="medium">Medium (0.08 kg/s)</option>
                                            <option value="big">Big (0.2 kg/s)</option>
                                            <option value="very_big">Very Big (5.0 kg/s)</option>
                                        </NativeSelect.Field>
                                        <NativeSelect.Indicator />
                                    </NativeSelect.Root>

                                    <Button
                                        size="sm"
                                        colorScheme="red"
                                        onClick={handleAddLeak}
                                    >
                                        Add Leak
                                    </Button>
                                </VStack>
                            </>
                        )}
                    </>
                ) : (
                    <>
                        {isBuilding(item) && (
                            <>
                                <VStack align="stretch" gap={3}>
                                    <Text fontSize="sm">Building Type</Text>
                                    <NativeSelect.Root size="sm">
                                        <NativeSelect.Field
                                            value={item.building_type ?? 'single_family'}
                                            onChange={(e) =>
                                                onUpdateBuilding?.(item.id, { building_type: e.target.value })
                                            }
                                        >
                                            <option value="single_family">Single Family</option>
                                            <option value="apartment">Apartment</option>
                                            <option value="restaurant">Restaurant</option>
                                            <option value="office">Office</option>
                                            <option value="factory">Factory</option>
                                            <option value="warehouse">Warehouse</option>
                                            <option value="processing_plant">Processing Plant</option>
                                        </NativeSelect.Field>
                                        <NativeSelect.Indicator />
                                    </NativeSelect.Root>

                                    <Text fontSize="sm">Rotation (°)</Text>
                                    <Input
                                        type="number"
                                        value={item.rotation ?? 0}
                                        onChange={(e) =>
                                            onUpdateBuilding?.(item.id, {
                                                rotation: parseFloat(e.target.value),
                                            })
                                        }
                                    />
                                </VStack>
                            </>
                        )}

                        {isRoad(item) && (
                            <>
                                <VStack align="stretch" gap={3}>
                                    <Text fontSize="sm">Pipe Type</Text>
                                    <NativeSelect.Root size="sm">
                                        <NativeSelect.Field
                                            value={item.pipe_type ?? 'main'}
                                            onChange={(e) =>
                                                onUpdatePipe?.(item.id, { pipe_type: e.target.value })
                                            }
                                        >
                                            <option value="main">Main</option>
                                            <option value="side">Side</option>
                                            <option value="building connection">
                                                Building Connection
                                            </option>
                                        </NativeSelect.Field>
                                        <NativeSelect.Indicator />
                                    </NativeSelect.Root>

                                    <Text fontSize="sm">Age (years)</Text>
                                    <Input
                                        type="number"
                                        min={0}
                                        max={50}
                                        value={item.age ?? 0}
                                        onChange={(e) =>
                                            onUpdatePipe?.(item.id, {
                                                age: parseInt(e.target.value),
                                            })
                                        }
                                    />
                                </VStack>
                            </>
                        )}

                        {isJunction(item) && (
                            <Text color="gray.600" fontSize="sm" fontStyle="italic">
                                Junctions currently cannot be edited.
                            </Text>
                        )}
                    </>
                )}
            </VStack>
        </Box>
    )
}
