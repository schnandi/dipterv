'use client'
import { Box, VStack, Input, Text, NativeSelect } from '@chakra-ui/react'

interface Props {
    pipe_type: string
    age: number
    onChange: (values: { pipe_type?: string; age?: number }) => void
}

export default function PipeSettingsPanel({ pipe_type, age, onChange }: Props) {
    return (
        <Box
            position="fixed"
            top="150px"
            right="40px"
            bg="white"
            p="4"
            borderRadius="md"
            shadow="md"
            zIndex={30}
            w="220px"
        >
            <VStack align="stretch" gap={3}>
                <Text fontWeight="bold" fontSize="sm">
                    Pipe Type
                </Text>
                <NativeSelect.Root size="sm">
                    <NativeSelect.Field
                        value={pipe_type}
                        onChange={(e) => onChange({ pipe_type: e.target.value })}
                    >
                        <option value="main">Main</option>
                        <option value="side">Side</option>
                        <option value="building connection">Building Connection</option>
                    </NativeSelect.Field>
                    <NativeSelect.Indicator />
                </NativeSelect.Root>

                <Text fontWeight="bold" fontSize="sm">
                    Age (years)
                </Text>
                <Input
                    type="number"
                    min={0}
                    max={50}
                    value={age}
                    onChange={(e) => onChange({ age: parseInt(e.target.value) })}
                />
            </VStack>
        </Box>
    )
}
