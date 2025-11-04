'use client'

import { Box, VStack, HStack, Icon, Text, Button } from '@chakra-ui/react'
import { FaHome, FaProjectDiagram, FaTrash, FaSave, FaUndo } from 'react-icons/fa'

interface Props {
    currentTool: 'addBuilding' | 'addPipe' | 'delete' | null
    onSelect: (tool: 'addBuilding' | 'addPipe' | 'delete' | null) => void
    onSave: () => void
    onReset: () => void
}

export default function Toolbox({ currentTool, onSelect, onSave, onReset }: Props) {
    const makeButton = (
        label: string,
        icon: React.ElementType,
        active: boolean,
        color?: string,
        onClick?: () => void
    ) => (
        <Button
            onClick={onClick}
            variant={active ? 'solid' : 'outline'}
            colorPalette={active ? color ?? 'blue' : undefined}
            justifyContent="flex-start"
            w="150px"
            rounded="lg"
            size="md"
        >
            <HStack gap="3">
                <Icon as={icon} boxSize="16px" />
                <Text fontSize="sm">{label}</Text>
            </HStack>
        </Button>
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
        >
            <VStack align="stretch" gap="3">
                {makeButton(
                    'Add Building',
                    FaHome,
                    currentTool === 'addBuilding',
                    'blue',
                    () => onSelect(currentTool === 'addBuilding' ? null : 'addBuilding')
                )}
                {makeButton(
                    'Add Pipe',
                    FaProjectDiagram,
                    currentTool === 'addPipe',
                    'blue',
                    () => onSelect(currentTool === 'addPipe' ? null : 'addPipe')
                )}
                {makeButton(
                    'Delete',
                    FaTrash,
                    currentTool === 'delete',
                    'red',
                    () => onSelect(currentTool === 'delete' ? null : 'delete')
                )}
                {makeButton('Save', FaSave, false, 'green', onSave)}
                {makeButton('Reset', FaUndo, false, 'red', onReset)}
            </VStack>

        </Box>
    )
}
