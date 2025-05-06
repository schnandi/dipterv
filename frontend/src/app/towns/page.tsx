/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'
import React, { useState } from 'react'
import useSWR from 'swr'
import {
    Box,
    Button,
    Flex,
    Heading,
    SimpleGrid,
    Spinner,
    Text,
    Input,
    Dialog,
    Portal,
    CloseButton,
    Field,
} from '@chakra-ui/react'
import { toaster } from '@/components/ui/toaster'
import { api } from '@/lib/api'
import TownCard from '@/components/ui/town/TownCard'

interface Town {
    id: number
    seed: number
    name?: string
}

export default function TownListPage() {
    const { data, error, isLoading, mutate } = useSWR<Town[]>(
        '/towns/',
        (url: string) => api.get(url).then(r => r.data)
    )

    // Dialog state
    const [name, setName] = useState('')
    const [isGenerating, setIsGenerating] = useState(false)

    const createTown = async () => {
        setIsGenerating(true)
        try {
            await api.post('/towns/generate', { name })
            toaster.create({ description: 'Town created', type: 'success' })
            mutate()
            setName('')
        } catch (err: any) {
            toaster.create({ description: err.message || 'Failed to generate', type: 'error' })
        } finally {
            setIsGenerating(false)
        }
    }

    const renameTown = async (t: Town) => {
        const newName = window.prompt('New name', t.name || '')
        if (newName == null) return
        try {
            await api.put(`/towns/${t.id}`, { name: newName })
            toaster.create({ description: 'Renamed', type: 'success' })
            mutate()
        } catch (err: any) {
            toaster.create({ description: err.message || 'Rename failed', type: 'error' })
        }
    }

    const deleteTown = async (t: Town) => {
        if (!window.confirm(`Delete ${t.name || `Town #${t.id}`}?`)) return
        try {
            await api.delete(`/towns/${t.id}`)
            toaster.create({ description: 'Deleted', type: 'info' })
            mutate()
        } catch (err: any) {
            toaster.create({ description: err.message || 'Delete failed', type: 'error' })
        }
    }

    if (isLoading) return <Spinner />
    if (error) return <Text color="red.500">Couldn’t load towns</Text>

    return (
        <Box p={6}>
            <Flex justify="space-between" align="center" mb={4}>
                <Heading size="lg">Towns</Heading>

                <Dialog.Root placement="center">
                    <Dialog.Trigger asChild>
                        <Button colorScheme="teal">+ New Town</Button>
                    </Dialog.Trigger>

                    <Portal>
                        <Dialog.Backdrop />
                        <Dialog.Positioner>
                            <Dialog.Content>

                                <Flex justify="space-between" align="center" mb="4">
                                    <Heading size="md">Generate Town</Heading>
                                    <Dialog.CloseTrigger asChild>
                                        <CloseButton />
                                    </Dialog.CloseTrigger>
                                </Flex>

                                <Field.Root>
                                    <Field.Label>Name (optional)</Field.Label>
                                    <Input
                                        placeholder="E.g. Springfield"
                                        value={name}
                                        onChange={e => setName(e.target.value)}
                                    />
                                </Field.Root>

                                <Flex justify="flex-end" mt="6" gap="2">
                                    <Dialog.CloseTrigger asChild>
                                        <Button variant="outline">Cancel</Button>
                                    </Dialog.CloseTrigger>
                                    <Dialog.CloseTrigger asChild>
                                        <Button
                                            colorScheme="teal"
                                            loading={isGenerating}
                                            onClick={createTown}
                                        >
                                            Generate
                                        </Button>
                                    </Dialog.CloseTrigger>
                                </Flex>
                            </Dialog.Content>
                        </Dialog.Positioner>
                    </Portal>
                </Dialog.Root>
            </Flex>

            {data && data.length
                ? (
                    <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} gap={4}>
                        {data.map(town => (
                            <Box key={town.id} borderWidth="1px" borderRadius="md" p={4}>
                                <TownCard town={town} />

                                <Flex mt={3} gap={2}>
                                    <Button size="sm" variant="outline" onClick={() => renameTown(town)}>
                                        Rename
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        colorScheme="red"
                                        onClick={() => deleteTown(town)}
                                    >
                                        Delete
                                    </Button>
                                </Flex>
                            </Box>
                        ))}
                    </SimpleGrid>
                )
                : <Text>No towns yet. Click “+ New Town” to start.</Text>
            }
        </Box>
    )
}
