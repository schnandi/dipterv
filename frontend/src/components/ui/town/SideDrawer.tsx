/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React from 'react'
import {
    Drawer,
    Portal,
    CloseButton,
    Stack,
    Text,
} from '@chakra-ui/react'

/**
 * When in “edit” mode, we’ll pass in the raw geometry object:
 *   e.g. { id: 5, start: [x,y], end: [x,y], pipe_type: 'main', … }
 *
 * When in “simulate” mode, we’ll pass in:
 *   { type: 'road'|'building'|'junction', id, timestamp, value, ...staticInfo }
 */
type GeometryItem = Record<string, any>

interface SimulationItem {
    type: 'road' | 'building' | 'junction'
    id: number
    timestamp: string
    value: number

    // merged-in static pipe info (only for roads)
    name?: string
    from_junction?: number
    to_junction?: number
    length_m?: number
    diameter_m?: number
    k_mm?: number
    material?: string | null
    max_velocity_m_per_s?: number | null
}

type Item = SimulationItem | GeometryItem | null

interface SideDrawerProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    item: Item
    isSim: boolean
}

export default function SideDrawer({
    open,
    onOpenChange,
    item,
    isSim
}: SideDrawerProps) {

    return (
        <Drawer.Root
            open={open}
            onOpenChange={evt => onOpenChange(evt.open)}
            placement="end"
            size="sm"
        >
            <Portal>
                <Drawer.Backdrop />
                <Drawer.Positioner>
                    <Drawer.Content>
                        {/* close button */}
                        <Drawer.CloseTrigger asChild>
                            <CloseButton
                                size="sm"
                                position="absolute"
                                top="4px"
                                right="4px"
                            />
                        </Drawer.CloseTrigger>

                        {/* header */}
                        <Drawer.Header>
                            <Text as="h2" fontSize="lg" fontWeight="semibold">
                                {item && isSim
                                    ? // simulation mode: show “Pipe 3” / “Building 5” / “Junction 2”
                                    `${(item as SimulationItem).type === 'road'
                                        ? 'Pipe'
                                        : (item as SimulationItem).type === 'building'
                                            ? 'Building'
                                            : 'Junction'} ${(item as SimulationItem).id}`
                                    : // edit mode: generic title
                                    'Details'}
                            </Text>
                        </Drawer.Header>

                        {/* body */}
                        <Drawer.Body>
                            {item ? (
                                isSim ? (
                                    // ── SIMULATION MODE ──
                                    <Stack>
                                        {/* Time + main value */}
                                        <Text>
                                            <Text as="span" fontWeight="bold">
                                                Time:
                                            </Text>{' '}
                                            {(item as SimulationItem).timestamp}
                                        </Text>
                                        <Text>
                                            <Text as="span" fontWeight="bold">
                                                {((item as SimulationItem).type === 'road'
                                                    ? 'Velocity'
                                                    : (item as SimulationItem).type === 'building'
                                                        ? 'Flow'
                                                        : 'Pressure')}
                                                :
                                            </Text>{' '}
                                            {(item as SimulationItem).value.toFixed(8)}{' '}
                                            {((item as SimulationItem).type === 'junction'
                                                ? 'bar'
                                                : (item as SimulationItem).type === 'building'
                                                    ? 'kg/s'
                                                    : 'm/s')}
                                        </Text>
                                        {(item as SimulationItem).type === 'building' && (
                                            <>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">Volumetric:</Text>{' '}
                                                    {((item as SimulationItem).value * 3600).toFixed(2)} L/h
                                                </Text>
                                            </>
                                        )}

                                        {/* ── EXTRA FOR PIPES ── */}
                                        {(item as SimulationItem).type === 'road' && (
                                            <>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">
                                                        From Junction:
                                                    </Text>{' '}
                                                    {(item as SimulationItem).from_junction}
                                                </Text>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">
                                                        To Junction:
                                                    </Text>{' '}
                                                    {(item as SimulationItem).to_junction}
                                                </Text>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">
                                                        Length (m):
                                                    </Text>{' '}
                                                    {(item as SimulationItem).length_m?.toFixed(4)}
                                                </Text>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">
                                                        Diameter (m):
                                                    </Text>{' '}
                                                    {(item as SimulationItem).diameter_m}
                                                </Text>
                                                <Text>
                                                    <Text as="span" fontWeight="bold">
                                                        Roughness (k_mm):
                                                    </Text>{' '}
                                                    {(item as SimulationItem).k_mm?.toFixed(6)}
                                                </Text>
                                                {(item as SimulationItem).material != null && (
                                                    <Text>
                                                        <Text as="span" fontWeight="bold">
                                                            Material:
                                                        </Text>{' '}
                                                        {(item as SimulationItem).material}
                                                    </Text>
                                                )}
                                                {(item as SimulationItem).max_velocity_m_per_s != null && (
                                                    <Text>
                                                        <Text as="span" fontWeight="bold">
                                                            Max Velocity (m/s):
                                                        </Text>{' '}
                                                        {(item as SimulationItem).max_velocity_m_per_s}
                                                    </Text>
                                                )}
                                            </>
                                        )}
                                        {Object.entries(item as Record<string, any>)
                                            .filter(([k]) =>
                                                !['type', 'id', 'timestamp', 'value',
                                                    'from_junction', 'to_junction', 'length_m', 'diameter_m', 'k_mm'
                                                ].includes(k)
                                            )
                                            .map(([key, val]) => (
                                                <Text key={key}>
                                                    <Text as="span" fontWeight="bold">{key}:</Text>{' '}
                                                    {JSON.stringify(val)}
                                                </Text>
                                            ))}
                                    </Stack>
                                ) : (
                                    // ── EDIT MODE ──
                                    <Stack>
                                        {Object.entries(item as GeometryItem).map(
                                            ([key, val]) => (
                                                <Text key={key}>
                                                    <Text as="span" fontWeight="bold">
                                                        {key}:
                                                    </Text>{' '}
                                                    {JSON.stringify(val)}
                                                </Text>
                                            )
                                        )}
                                    </Stack>
                                )
                            ) : (
                                <Text>No selection</Text>
                            )}
                        </Drawer.Body>
                    </Drawer.Content>
                </Drawer.Positioner>
            </Portal>
        </Drawer.Root>
    )
}
