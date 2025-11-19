'use client'

import { useEffect, useState } from 'react'
import {
    Box,
    VStack,
    HStack,
    Text,
    Button,
    Spinner,
    Icon,
} from '@chakra-ui/react'
import {
    FaExclamationTriangle,
    FaBrain,
    FaChevronDown,
    FaChevronUp,
    FaFlask,
} from 'react-icons/fa'
import { api } from '@/lib/api'

interface PipeRisk {
    pipe_id: number
    age: number
    burst_risk: number
    mean_flow: number
    mean_pressure: number
}

interface Props {
    townId: number
    onSelectPipe: (pipeId: number) => void
}

export default function BurstRiskPanel({ townId, onSelectPipe }: Props) {
    const [data, setData] = useState<PipeRisk[]>([])
    const [young, setYoung] = useState<PipeRisk[]>([])
    const [loading, setLoading] = useState(true)
    const [openMain, setOpenMain] = useState(true)
    const [openYoung, setOpenYoung] = useState(true)

    useEffect(() => {
        setLoading(true)
        api
            .get(`/burst-risk/${townId}`)
            .then(res => {
                setData(res.data.pipe_risk_summary || [])
                setYoung(res.data.concerning_young_pipes || [])
                setLoading(false)
            })
            .catch(err => {
                console.error('Failed to fetch burst risks:', err)
                setLoading(false)
            })
    }, [townId])

    const renderPipeList = (list: PipeRisk[]) => (
        <VStack align="stretch" gap="2" mt="2">
            {list.slice(0, 20).map(pipe => {
                const color =
                    pipe.burst_risk > 0.55
                        ? 'red.500'
                        : pipe.burst_risk > 0.4
                            ? 'orange.500'
                            : 'gray.600'

                return (
                    <Box
                        key={pipe.pipe_id}
                        borderWidth="1px"
                        borderColor="gray.100"
                        borderRadius="lg"
                        p="2"
                        _hover={{ bg: 'gray.50', cursor: 'pointer' }}
                        onClick={() => onSelectPipe(Number(pipe.pipe_id))}
                    >
                        <HStack justify="space-between">
                            <Text fontSize="sm" fontWeight="medium">
                                Pipe #{pipe.pipe_id}
                            </Text>
                            <Text fontSize="sm" color={color} fontWeight="semibold">
                                {(pipe.burst_risk * 100).toFixed(1)}%
                            </Text>
                        </HStack>
                        <HStack justify="space-between" mt="1">
                            <Text fontSize="xs" color="gray.500">
                                Age: {pipe.age}
                            </Text>
                            <Text fontSize="xs" color="gray.500">
                                Flow: {pipe.mean_flow.toFixed(2)} kg/s
                            </Text>
                        </HStack>
                    </Box>
                )
            })}

            {list.length === 0 && !loading && (
                <HStack justify="center" color="gray.500" py="3">
                    <Icon as={FaExclamationTriangle} />
                    <Text fontSize="sm">No pipes found</Text>
                </HStack>
            )}
        </VStack>
    )

    return (
        <Box
            position="fixed"
            left="20px"
            top="200px"
            bg="bg.surface"
            p="3"
            borderRadius="2xl"
            boxShadow="lg"
            zIndex={30}
            w="280px"
            maxH="70vh"
            overflowY="auto"
        >
            <VStack align="stretch" gap="6">
                {/* 🧠 General Burst Risk Section */}
                <Box>
                    <HStack justify="space-between">
                        <HStack gap="2">
                            <Icon as={FaBrain} color="teal.500" boxSize="18px" />
                            <Text fontWeight="semibold" color="teal.700">
                                General Burst Risk
                            </Text>
                        </HStack>
                        <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setOpenMain(!openMain)}
                        >
                            <HStack gap="1">
                                <Text>{openMain ? 'Hide' : 'Show'}</Text>
                                <Icon
                                    as={openMain ? FaChevronUp : FaChevronDown}
                                    boxSize="12px"
                                />
                            </HStack>
                        </Button>
                    </HStack>

                    {openMain && (
                        <Box mt="2">
                            {loading ? (
                                <HStack justify="center" py={4}>
                                    <Spinner size="sm" />
                                    <Text fontSize="sm" color="gray.500">
                                        Loading...
                                    </Text>
                                </HStack>
                            ) : (
                                renderPipeList(data)
                            )}
                        </Box>
                    )}
                </Box>

                {/* ⚠️ Concerning Young Pipes Section */}
                <Box>
                    <HStack justify="space-between">
                        <HStack gap="2">
                            <Icon as={FaFlask} color="orange.500" boxSize="18px" />
                            <Text fontWeight="semibold" color="orange.700">
                                Concerning Young Pipes
                            </Text>
                        </HStack>
                        <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setOpenYoung(!openYoung)}
                        >
                            <HStack gap="1">
                                <Text>{openYoung ? 'Hide' : 'Show'}</Text>
                                <Icon
                                    as={openYoung ? FaChevronUp : FaChevronDown}
                                    boxSize="12px"
                                />
                            </HStack>
                        </Button>
                    </HStack>

                    {openYoung && (
                        <Box mt="2">
                            {loading ? (
                                <HStack justify="center" py={4}>
                                    <Spinner size="sm" />
                                    <Text fontSize="sm" color="gray.500">
                                        Loading...
                                    </Text>
                                </HStack>
                            ) : (
                                renderPipeList(young)
                            )}
                        </Box>
                    )}
                </Box>
            </VStack>
        </Box>
    )
}
