"""Logging configuration for diffusion profile computation."""

from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logging(
    output_dir: str,
    log_name: str = "diffusion_profiles",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    logger_name: str = "diffusion_profiles",
) -> logging.Logger:
    """Configure logging to both console and file in output directory.

    Creates log file: {output_dir}/{log_name}_{timestamp}.log

    Args:
        output_dir: Directory to store log file
        log_name: Base name for log file
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        logger_name: Name for the logger instance (default: diffusion_profiles)

    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{log_name}_{timestamp}.log")

    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers (in case of re-initialization)
    logger.handlers.clear()

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler.setFormatter(file_format)

    # Console handler (info and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    """Get the diffusion_profiles logger.

    Returns the existing logger if already configured, or a basic one if not.
    """
    return logging.getLogger("diffusion_profiles")
