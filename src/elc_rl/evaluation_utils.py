"""Shared numerical helpers for the physics controller evaluators."""

from __future__ import annotations

import numpy as np


GAIN_MARGIN_CAP_DB = 120.0


def zero_crossing_locations(
    log_frequency: np.ndarray, values: np.ndarray
) -> list[tuple[float, int, float]]:
    """Locate sign changes using linear interpolation on a log-frequency grid."""

    signs = np.signbit(values)
    indices = np.flatnonzero(signs[:-1] != signs[1:])
    crossings: list[tuple[float, int, float]] = []
    for index in indices:
        y0, y1 = values[index : index + 2]
        fraction = float(-y0 / (y1 - y0))
        log_value = float(
            log_frequency[index]
            + fraction * (log_frequency[index + 1] - log_frequency[index])
        )
        crossings.append((10.0**log_value, int(index), fraction))
    return crossings


def interpolate_pair(values: np.ndarray, index: int, fraction: float) -> float:
    """Interpolate a value between two adjacent samples."""

    return float(values[index] + fraction * (values[index + 1] - values[index]))
