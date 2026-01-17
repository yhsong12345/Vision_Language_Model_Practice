"""
Core Framework Centralized Logging Configuration

Provides consistent logging setup across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from rich.logging import RichHandler
from rich.console import Console


# Default format for file logging
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console for rich output
console = Console()


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True,
    module_name: str = "vla",
) -> logging.Logger:
    """
    Set up logging configuration for the VLA framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If provided, logs will be written to file.
        use_rich: Whether to use rich formatting for console output
        module_name: Name of the logger to configure

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="training.log")
        >>> logger.info("Training started")
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt=DATE_FORMAT,
        )
        console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt=FILE_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training progress with structured output.

    Provides methods for logging training metrics, progress bars,
    and experiment tracking.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: str = "INFO",
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        self.logger = setup_logging(
            level=level,
            log_file=log_file,
            module_name=f"vla.{experiment_name}",
        )

        self._step = 0
        self._epoch = 0

    def set_step(self, step: int) -> None:
        """Set the current training step."""
        self._step = step

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self._epoch = epoch

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names to values
            prefix: Optional prefix for metric names (e.g., "train", "val")
        """
        prefix_str = f"{prefix}/" if prefix else ""
        metrics_str = " | ".join(
            f"{prefix_str}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix_str}{k}: {v}"
            for k, v in metrics.items()
        )
        self.logger.info(f"[Epoch {self._epoch} Step {self._step}] {metrics_str}")

    def log_hyperparameters(self, hparams: dict) -> None:
        """Log hyperparameters at the start of training."""
        self.logger.info("=" * 60)
        self.logger.info("Hyperparameters:")
        for k, v in hparams.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 60)

    def log_model_summary(self, total_params: int, trainable_params: int) -> None:
        """Log model parameter summary."""
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)


def silence_transformers_logging() -> None:
    """Silence verbose logging from transformers and other libraries."""
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


def enable_debug_logging() -> None:
    """Enable debug logging for all VLA modules."""
    setup_logging(level="DEBUG")
