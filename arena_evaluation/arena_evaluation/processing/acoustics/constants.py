"""Acoustic model constants shared by the solver wrapper, the metric and the plot."""

from typing import Final

ACOUSTIC_REFERENCE_PRESSURE_PA: Final[float] = 20e-6
ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2: Final[float] = ACOUSTIC_REFERENCE_PRESSURE_PA * ACOUSTIC_REFERENCE_PRESSURE_PA

# Propagation model: a wall crossing pays the full loss once, the microphone sits 1 m
# from the robot so closer targets never exceed free-space at that distance.
WALL_TRANSMISSION_LOSS_DB: Final[float] = 47.0
MIC_DISTANCE_M: Final[float] = 1.0

# Door model: a door pays its loss spread over the cells of its thickness; a door above
# OPEN_PROGRESS_THRESHOLD (0 closed, 1 open) transmits fully.
DOOR_TRANSMISSION_LOSS_DB: Final[float] = 25.0
DOOR_OPEN_PROGRESS_THRESHOLD: Final[float] = 0.15

# The barrier-aware exposure is re-solved only after the robot or a pedestrian moved.
POSITION_EVALUATION_THRESHOLD_M: Final[float] = 0.5

# The exposure solves one wavefront over the barrier grid per re-solved sample, so its cost grows with the grid's
# cells times the samples: a grid above MAX_EXPOSURE_GRID_CELLS is not solved (its values are NULL, `grid_too_large`).
# ARENA_ACOUSTIC_MAX_GRID_CELLS overrides the limit, 0 lifts it.
MAX_EXPOSURE_GRID_CELLS: Final[int] = 4_000_000
MAX_EXPOSURE_GRID_CELLS_ENV: Final[str] = "ARENA_ACOUSTIC_MAX_GRID_CELLS"

NANOSECONDS_PER_SECOND: Final[float] = 1e9

# The recorded topic every acoustic metric and the field renderer read.
ROBOT_ACOUSTICS_TOPIC: Final[str] = "robot_acoustics"

# ABI of the C surface `processing/acoustics/solver.cpp` exports
ACOUSTIC_SOLVER_ABI: Final[int] = 1
ACOUSTIC_SOLVER_ABI_SYMBOL: Final[str] = "arena_acoustic_solver_abi"
ACOUSTIC_SOLVER_SYMBOLS: Final[tuple[str, ...]] = (
    "solve_acoustic_field",
    "solve_acoustic_grid",
)
