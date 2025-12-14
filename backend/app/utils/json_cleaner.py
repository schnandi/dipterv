import math


def clean_json(value):
    """
    Recursively clean NaN/Infinity from a JSON-like structure.
    Converts them to None so PostgreSQL JSON accepts it.
    """
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_json(v) for v in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value
