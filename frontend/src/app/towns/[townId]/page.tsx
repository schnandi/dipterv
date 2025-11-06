/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, { useState } from 'react'
import dynamic from 'next/dynamic'
import useSWR, { useSWRConfig } from 'swr'
import { useRouter, useParams } from 'next/navigation'
import {
    Box,
    Flex,
    Heading,
    Button,
    ButtonGroup,
    Spinner,
    Text,
    HStack,
    IconButton,
    Switch,
    VStack,
} from '@chakra-ui/react'
import { LuArrowLeft } from 'react-icons/lu'
import { api } from '../../../lib/api'
import TownCanvas from '@/components/ui/town-map/TownCanvas'
import SimulationControls from '../../../components/ui/town/SimulationControls'

// dynamically import drawer so it never SSR’s
const SideDrawer = dynamic(
    () => import('../../../components/ui/town/SideDrawer'),
    { ssr: false }
)

interface TownData {
    name: string
    seed: number
    data: {
        map_size: [number, number]
        height_map_resolution: [number, number]
        height_map: number[][]
        height_map_bounds: [number, number, number, number]
        roads: any[]
        buildings: any[]
    }
}


interface SimulationSummary {
    id: number
    title: string
    town_id: number
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

interface SimulationFull extends SimulationSummary {
    details: {
        timestamps: string[]
        junction_pressures: Record<string, number[]>
        pipe_velocities: Record<string, number[]>
        pipe_flows: Record<string, number[]>
        sink_flows: Record<string, number[]>
        pipe_parameters: Record<string, {
            name: string
            from_junction: number
            to_junction: number
            length_m: number
            diameter_m: number
            k_mm: number
            material?: string | null
            max_velocity_m_per_s?: number | null
        }>
        external_grid: ExternalGrid
        pumps: PumpInfo[]
    }
}

export default function TownDetailPage() {
    const router = useRouter()
    const params = useParams()
    const townId = Array.isArray(params.townId) ? params.townId[0] : params.townId!

    // local UI state
    const [frame, setFrame] = useState(0)
    const [isPlaying, setPlaying] = useState(false)
    const [speed, setSpeed] = useState(1)
    const [selected, setSelected] = useState<{ type: 'road' | 'building' | 'junction'; id: number } | null>(null)
    const [viewMode, setViewMode] = useState<'edit' | 'simulate'>('edit')
    const [isRegenerating, setRegenerating] = useState(false);
    const [showTerrain, setShowTerrain] = useState(false)
    const [showFlow, setShowFlow] = useState(false)

    // fetch town geometry
    const { data: townData, error, isLoading } = useSWR<TownData>(
        `/towns/${townId}`, (url: string) => api.get(url).then(r => r.data)
    )

    // fetch simulation summaries
    const { data: sims, isLoading: simsLoading } = useSWR<SimulationSummary[]>(
        `/simulations/town/${townId}`, (url: string) => api.get(url).then(r => r.data)
    )
    const sim = sims?.[0] ?? null

    // fetch full simulation details
    const { data: simDetail, isLoading: simDetailLoading } = useSWR<SimulationFull>(
        sim ? `/simulations/${sim.id}` : null,
        (url: string) => api.get(url).then(r => r.data)
    )

    // create simulation helper
    const { mutate } = useSWRConfig()
    const createSim = async () => {
        setRegenerating(true);
        await api.post('/simulations/', {
            title: `Sim for town ${townId}`,
            town_id: Number(townId),
        });
        mutate(`/simulations/town/${townId}`);
        setRegenerating(false);
    };

    React.useEffect(() => {
        const handleTownUpdated = (e: CustomEvent) => {
            if (String(e.detail.townId) === String(townId)) {
                // 🧹 Clear simulation list + detail from cache
                mutate(`/simulations/town/${townId}`);
                mutate(`/simulations/${sims?.[0]?.id}`);
            }
        };

        window.addEventListener('town-updated', handleTownUpdated as EventListener);
        return () => window.removeEventListener('town-updated', handleTownUpdated as EventListener);
    }, [mutate, townId, sims]);

    if (isLoading) return <Spinner />
    if (error) return <Text color="red.500">Failed to load town.</Text>

    return (
        <Flex direction="column" h="100vh" overflow="hidden">
            <Flex p="4" gap="12" align="center">
                <IconButton
                    aria-label="Back to towns"
                    variant="ghost"
                    onClick={() => router.push('/towns')}
                >
                    <LuArrowLeft />
                </IconButton>

                <Heading size="xl">
                    {townData?.seed && `${townData.name} (seed ${townData.seed})`}
                </Heading>

                {simsLoading || isRegenerating ? (
                    <Spinner size="sm" marginEnd="auto" />
                ) : sim ? (
                    <HStack marginEnd="auto">
                        <Box w={2} h={2} bg="green.500" borderRadius="full" />
                        <Text>Simulation is ready</Text>
                        <Button size="sm" colorScheme="yellow" onClick={createSim}>
                            Regenerate
                        </Button>
                    </HStack>
                ) : (
                    <Button colorScheme="green" onClick={createSim} marginEnd="auto">
                        Generate Simulation
                    </Button>
                )}

                <ButtonGroup>
                    <Button
                        variant={viewMode === 'edit' ? 'solid' : 'outline'}
                        onClick={() => setViewMode('edit')}
                    >
                        Edit
                    </Button>
                    <Button
                        variant={viewMode === 'simulate' ? 'solid' : 'outline'}
                        onClick={() => setViewMode('simulate')}
                    >
                        Simulate
                    </Button>
                </ButtonGroup>
            </Flex>

            <Box flex="1" position="relative">
                <Box
                    position="absolute"
                    top="4"
                    left="5"
                    bg="gray.100"
                    p="3"
                    borderRadius="md"
                    boxShadow="md"
                    zIndex="1"
                >
                    <VStack align="stretch">
                        <HStack justify="space-between">
                            <Switch.Root
                                size="sm"
                                checked={showTerrain}
                                colorPalette="teal"
                                onCheckedChange={checked => setShowTerrain(checked.checked)}
                            >
                                <Switch.HiddenInput />
                                <Switch.Control />
                                <Switch.Label color="black">Show Terrain</Switch.Label>
                            </Switch.Root>
                        </HStack>
                        <HStack justify="space-between">
                            <Switch.Root
                                size="sm"
                                checked={showFlow}
                                colorPalette="teal"
                                onCheckedChange={checked => setShowFlow(checked.checked)}
                            >
                                <Switch.HiddenInput />
                                <Switch.Control />
                                <Switch.Label color="black">Show Pipe Flow</Switch.Label>
                            </Switch.Root>
                        </HStack>
                    </VStack>
                </Box>

                {simDetailLoading ? (
                    <Flex h="100%" align="center" justify="center">
                        <Spinner size="xl" />
                    </Flex>
                ) : (
                    <TownCanvas
                        mode={viewMode}
                        roads={townData!.data.roads}
                        buildings={townData!.data.buildings}
                        junctions={simDetail?.details.junction_pressures || {}}
                        pipeVelocities={simDetail?.details.pipe_velocities}
                        pipeFlows={simDetail?.details.pipe_flows}
                        sinkFlows={simDetail?.details.sink_flows}
                        frame={frame}
                        mapSize={townData!.data.map_size}
                        heightMap={townData!.data.height_map}
                        heightMapBounds={townData!.data.height_map_bounds}
                        externalGrid={simDetail?.details.external_grid}
                        pumps={simDetail?.details.pumps}
                        showTerrain={showTerrain}
                        showFlow={showFlow}
                        pipeParameters={simDetail?.details.pipe_parameters}
                        timestamps={simDetail?.details.timestamps}
                    />
                )}
            </Box>
            {viewMode === 'simulate' && simDetail && (
                <Box
                    position="fixed"
                    bottom="16px"
                    left="50%"
                    transform="translateX(-50%)"
                    zIndex={20}
                >
                    <SimulationControls
                        frame={frame}
                        setFrame={setFrame}
                        maxFrame={simDetail.details.timestamps.length - 1}
                        isPlaying={isPlaying}
                        setPlaying={setPlaying}
                        speed={speed}
                        setSpeed={setSpeed}
                        timestamps={simDetail.details.timestamps}
                    />
                </Box>
            )}
        </Flex>
    )
}
