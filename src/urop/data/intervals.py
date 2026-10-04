"""Validation of sorted, disjoint participant intervals."""
import numpy as np

def validate_spans(spans):
    values = np.asarray(spans, dtype=float).reshape(-1, 2)
    if (not np.isfinite(values).all() or np.any(values[:, 0] < 0)
            or np.any(values[:, 1] <= values[:, 0])
            or np.any(values[1:, 0] < values[:-1, 1])):
        raise ValueError('spans must be finite, positive, sorted and nonoverlapping')
    return values
