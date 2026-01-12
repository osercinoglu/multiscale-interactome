import multiprocessing
from typing import Final

# Fixed project defaults (mirrors current notebook usage)
DEFAULT_ALPHA: Final[float] = 0.8595436247434408
DEFAULT_MAX_ITER: Final[int] = 1000
DEFAULT_TOL: Final[float] = 1e-6

# Keys must match successor_type values produced by MSI.weight_graph()
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "down_biological_function": 4.4863053901688685,
    "indication": 3.541889556309463,
    "biological_function": 6.583155399238509,
    "up_biological_function": 2.09685000906964,
    "protein": 4.396695660380823,
    "drug": 3.2071696595616364,
}


def default_num_cores() -> int:
    """Safe default: ~40% fewer than available cores."""
    return max(1, int(multiprocessing.cpu_count() * 0.6))
