'use client'

import { useState } from 'react'
import {
    Box,
    HStack,
    Input,
    Button,
} from '@chakra-ui/react'
import { FaSearch } from 'react-icons/fa'

interface Props {
    onGoToPipe: (pipeId: number) => void
}

export default function PipeSearchPanel({ onGoToPipe }: Props) {
    const [pipeId, setPipeId] = useState('')


    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        const id = Number(pipeId)
        onGoToPipe(id)
        setPipeId('')
    }

    return (
        <Box
            position="fixed"
            left="20px"
            top="150px"
            bg="bg.surface"
            p="2"
            borderRadius="xl"
            boxShadow="md"
            zIndex={35}
            w="200px"
        >
            <form onSubmit={handleSubmit}>
                <HStack>

                    <Input
                        size="sm"
                        placeholder="Go to pipe ID..."
                        value={pipeId}
                        onChange={(e) => setPipeId(e.target.value)}
                    />
                    <Button
                        size="sm"
                        aria-label="Go"
                        type="submit"
                        bg="white"
                        color="black"
                    />
                </HStack>
            </form>
        </Box>
    )
}
