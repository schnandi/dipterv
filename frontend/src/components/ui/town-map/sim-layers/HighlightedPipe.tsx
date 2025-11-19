import React from "react";
import { Line } from "react-konva";

export function HighlightedPipe({
    start,
    end,
}: {
    start: [number, number];
    end: [number, number];
}) {
    const [opacity, setOpacity] = React.useState(1);

    React.useEffect(() => {
        let dir = -1;
        const id = setInterval(() => {
            setOpacity((prev) => {
                let next = prev + dir * 0.1;
                if (next <= 0.2 || next >= 1) dir *= -1;
                return Math.max(0.2, Math.min(1, next));
            });
        }, 50);

        return () => clearInterval(id);
    }, []);

    return (
        <Line
            points={[start[0], start[1], end[0], end[1]]}
            stroke="yellow"
            strokeWidth={6}
            opacity={opacity}
            lineCap="round"
            shadowColor="yellow"
            shadowBlur={15}
        />
    );
}
