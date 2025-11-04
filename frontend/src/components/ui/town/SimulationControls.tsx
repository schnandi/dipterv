'use client'

import React, { useEffect } from 'react'
import {
    Box,
    Text,
    Button,
    IconButton,
    HStack,
    Slider,
    NativeSelect,
} from '@chakra-ui/react'
import { FaPlay, FaPause } from 'react-icons/fa'

export interface SimulationControlsProps {
    frame: number
    setFrame: (f: number) => void
    maxFrame: number
    isPlaying: boolean
    setPlaying: (b: boolean) => void
    speed: number
    setSpeed: (s: number) => void
    timestamps: string[]
}

export default function SimulationControls({
    frame,
    setFrame,
    maxFrame,
    isPlaying,
    setPlaying,
    speed,
    setSpeed,
    timestamps
}: SimulationControlsProps) {
    // Advance frames when playing
    useEffect(() => {
        if (!isPlaying) return
        const id = setInterval(() => {
            setFrame(frame < maxFrame ? frame + 1 : 0)
        }, 500 / speed)
        return () => clearInterval(id)
    }, [isPlaying, speed, frame, maxFrame, setFrame])

    // Arrow key navigation
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'ArrowRight') setFrame(Math.min(frame + 1, maxFrame))
            if (e.key === 'ArrowLeft') setFrame(Math.max(frame - 1, 0))
            if (e.code === 'Space') {
                setPlaying(!isPlaying)
                e.preventDefault()
            }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [frame, maxFrame, setFrame, setPlaying, isPlaying])

    return (
        <Box
            position="absolute"
            bottom="4"
            left="50%"
            transform="translateX(-50%)"
            p="4"
            bg="white"
            shadow="md"
            borderRadius="md"
            w="320px"
            color="black"
        >
            <Text mb="2" color="black">
                Frame {frame} / {maxFrame} — {timestamps[frame]}
            </Text>

            <Slider.Root
                value={[frame]}
                max={maxFrame}
                onValueChange={(details) => setFrame(details.value[0])}
            >
                <Slider.Control>
                    <Slider.Track>
                        <Slider.Range />
                    </Slider.Track>
                    <Slider.Thumbs />
                </Slider.Control>
            </Slider.Root>

            <HStack mt="4" justify="space-between">
                <IconButton
                    aria-label={isPlaying ? 'Pause' : 'Play'}
                    onClick={() => setPlaying(!isPlaying)}
                    size="sm"
                    variant="outline"
                    color="black"
                    borderColor="gray.300"
                    bg="white"
                    _hover={{ bg: 'gray.100' }}
                    _active={{ bg: 'gray.200' }}
                >
                    {isPlaying ? <FaPause /> : <FaPlay />}
                </IconButton>

                <NativeSelect.Root size="sm" width="20">
                    <NativeSelect.Field
                        value={String(speed)}
                        onChange={(e) => setSpeed(Number(e.currentTarget.value))}
                        bg="white"
                        color="black"
                        _focus={{ borderColor: 'black' }}
                    >
                        <option value="0.5">0.5×</option>
                        <option value="1">1×</option>
                        <option value="2">2×</option>
                        <option value="4">4×</option>
                    </NativeSelect.Field>
                    <NativeSelect.Indicator color="black" />
                </NativeSelect.Root>

                <Button
                    size="sm"
                    bg="white"
                    color="black"
                    borderWidth="1px"
                    borderColor="gray.300"
                    _hover={{ bg: 'gray.100' }}
                    _active={{ bg: 'gray.200' }}
                    onClick={() => setFrame(0)}
                >
                    Restart
                </Button>
            </HStack>
        </Box>
    )
}
