'use client'
import { Box, VStack, Input, Text, NativeSelect } from '@chakra-ui/react'

interface Props {
    size: number
    buildingType: string
    onChange: (values: { size?: number; rotation?: number; buildingType?: string }) => void
}

export default function BuildingSettingsPanel({
    size,
    buildingType,
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
            w="240px"
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
                    Building Type
                </Text>
                <NativeSelect.Root size="sm">
                    <NativeSelect.Field
                        value={buildingType}
                        onChange={(e) => onChange({ buildingType: e.target.value })}
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
            </VStack>
        </Box>
    )
}
