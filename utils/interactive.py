"""Interactive selection utilities for diffusion profile computation."""

from __future__ import annotations

import os
import re


def parse_selection(selection: str, max_value: int) -> list[int]:
    """Parse user selection string into list of indices (1-based).

    Supports:
    - 'all' -> all indices
    - Single numbers: '5'
    - Comma-separated: '1,3,5'
    - Ranges: '1-10'
    - Mixed: '1-5,8,12-15'

    Args:
        selection: User input string
        max_value: Maximum valid index (inclusive)

    Returns:
        Sorted list of 1-based indices

    Raises:
        ValueError: If selection is invalid or contains out-of-range values
    """
    selection = selection.strip().lower()

    if selection == "all":
        return list(range(1, max_value + 1))

    if not selection:
        raise ValueError("Empty selection")

    indices = set()
    parts = selection.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for range (e.g., '1-10')
        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise ValueError(f"Invalid range: {start}-{end} (start > end)")
            if start < 1:
                raise ValueError(f"Invalid range start: {start} (must be >= 1)")
            if end > max_value:
                raise ValueError(f"Invalid range end: {end} (max is {max_value})")
            indices.update(range(start, end + 1))
        else:
            # Single number
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(f"Invalid selection: '{part}'")

            if idx < 1 or idx > max_value:
                raise ValueError(f"Index {idx} out of range (1-{max_value})")
            indices.add(idx)

    if not indices:
        raise ValueError("No valid indices in selection")

    return sorted(indices)


def select_runs_interactively(
    available_runs: list[str],
    pipeline_name: str = "configurations",
) -> list[str]:
    """Display available runs and let user select a subset.

    Args:
        available_runs: List of run identifiers (e.g., TSV filenames)
        pipeline_name: Name to display for the pipeline

    Returns:
        List of selected run identifiers
    """
    if not available_runs:
        print(f"No {pipeline_name} available.")
        return []

    print(f"\nAvailable {pipeline_name} ({len(available_runs)} total):")
    for i, run in enumerate(available_runs, 1):
        # Display just the filename without full path
        display_name = os.path.basename(run) if os.path.sep in run else run
        print(f"  [{i:3d}] {display_name}")

    print("\nEnter selection:")
    print("  - 'all' to compute all")
    print("  - Comma-separated numbers (e.g., '1,3,5')")
    print("  - Range (e.g., '1-10')")
    print("  - Mixed (e.g., '1-5,8,12-15')")
    print("  - 'q' or 'quit' to cancel")

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            selection = input("> ").strip()

            if selection.lower() in ("q", "quit", "exit"):
                print("Selection cancelled.")
                return []

            indices = parse_selection(selection, len(available_runs))
            selected = [available_runs[i - 1] for i in indices]

            print(f"\nSelected {len(selected)} configuration(s).")
            return selected

        except ValueError as e:
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"Error: {e}. Please try again ({remaining} attempts remaining).")
            else:
                print(f"Error: {e}. Maximum attempts reached.")
                return []

    return []
