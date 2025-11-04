'use client'
import { Box, VStack, Input, Text, NativeSelect } from '@chakra-ui/react'

interface Props {
    size: number
    rotation: number
    district: string
    onChange: (values: { size?: number; rotation?: number; district?: string }) => void
}

export default function BuildingSettingsPanel({
    size,
    rotation,
    district,
    onChange,
}: Props) {
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
                    Building Size
                </Text>
                <Input
                    type="number"
                    value={size}
                    onChange={(e) => onChange({ size: parseFloat(e.target.value) })}
                />

                <Text fontWeight="bold" fontSize="sm">
                    Rotation (°)
                </Text>
                <Input
                    type="number"
                    value={rotation}
                    onChange={(e) => onChange({ rotation: parseFloat(e.target.value) })}
                />

                <Text fontWeight="bold" fontSize="sm">
                    District
                </Text>
                <NativeSelect.Root size="sm">
                    <NativeSelect.Field
                        value={district}
                        onChange={(e) => onChange({ district: e.target.value })}
                    >
                        <option value="residential">Residential</option>
                        <option value="industrial">Industrial</option>
                    </NativeSelect.Field>
                    <NativeSelect.Indicator />
                </NativeSelect.Root>

            </VStack>
        </Box>
    )
}
