'use client';
import React from 'react';
import { Box, Heading, Text, BoxProps } from '@chakra-ui/react';
import { useRouter } from 'next/navigation';

interface Town {
    id: number;
    seed: number;
    name?: string;
}

export default function TownCard({
    town,
    ...boxProps
}: { town: Town } & BoxProps) {
    const router = useRouter();
    return (
        <Box
            as="div"
            borderWidth="1px"
            borderRadius="md"
            p={4}
            _hover={{ bg: 'gray.50', cursor: 'pointer' }}
            onClick={() => router.push(`/towns/${town.id}`)}
            {...boxProps}
        >
            <Heading size="md" mb={2}>
                {town.name || `Town #${town.id}`}
            </Heading>
            <Text fontSize="sm" color="gray.600">
                Seed: {town.seed}
            </Text>
        </Box>
    );
}
