"""
Logging utilities for Judgement RL.

This module provides a centralized logging system with proper formatting,
file output, and different log levels for different components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logger(
    name: str = "judgement_rl",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_colors: bool = True,
    use_json: bool = False,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        log_dir: Directory for log files
        use_colors: Whether to use colored output in console
        use_json: Whether to use JSON formatting for file output
        include_timestamp: Whether to include timestamp in console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors:
        console_format = "%(levelname)s - %(name)s - %(message)s"
        console_formatter = ColoredFormatter(console_format)
    else:
        if include_timestamp:
            console_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        else:
            console_format = "%(levelname)s - %(name)s - %(message)s"
        console_formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(level)

        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            file_formatter = logging.Formatter(file_format)

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "judgement_rl") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, message: str, **context):
    """Log a message with additional context."""
    extra_fields = context
    logger.info(message, extra={"extra_fields": extra_fields})


class TrainingLogger:
    """Specialized logger for training sessions."""

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create experiment-specific log file
        if experiment_name is None:
            experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_file = f"{experiment_name}.log"

        # Set up logger
        self.logger = setup_logger(
            name=f"training.{experiment_name}",
            log_file=self.log_file,
            log_dir=str(self.log_dir),
            use_colors=True,
            use_json=False,
        )

        # Set up external logging
        self.tensorboard_writer = None
        self.wandb_run = None

        if use_tensorboard:
            self._setup_tensorboard()

        if use_wandb:
            self._setup_wandb()

    def _setup_tensorboard(self):
        """Set up TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = self.log_dir / "tensorboard" / self.experiment_name
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tensorboard_dir))
            self.logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")
        except ImportError:
            self.logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )

    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project="judgement-rl",
                name=self.experiment_name,
                config={},
                dir=str(self.log_dir),
            )
            self.logger.info("Weights & Biases logging enabled")
        except ImportError:
            self.logger.warning(
                "Weights & Biases not available. Install with: pip install wandb"
            )

    def log_episode(self, episode: int, reward: float, **metrics):
        """Log episode metrics."""
        message = f"Episode {episode}: reward={reward:.2f}"
        if metrics:
            metric_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
            message += f", {metric_str}"

        self.logger.info(message)

        # Log to external tools
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("episode/reward", reward, episode)
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"episode/{key}", value, episode)

        if self.wandb_run:
            wandb.log({"episode": episode, "reward": reward, **metrics})

    def log_training_step(self, step: int, **metrics):
        """Log training step metrics."""
        message = f"Training step {step}"
        if metrics:
            metric_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
            message += f": {metric_str}"

        self.logger.info(message)

        # Log to external tools
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"training/{key}", value, step)

        if self.wandb_run:
            wandb.log({"training_step": step, **metrics})

    def log_evaluation(self, episode: int, **metrics):
        """Log evaluation metrics."""
        message = f"Evaluation at episode {episode}"
        if metrics:
            metric_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
            message += f": {metric_str}"

        self.logger.info(message)

        # Log to external tools
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"evaluation/{key}", value, episode)

        if self.wandb_run:
            wandb.log({"evaluation_episode": episode, **metrics})

    def close(self):
        """Close the logger and external tools."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        if self.wandb_run:
            self.wandb_run.finish()

        self.logger.info("Training logger closed")


# Convenience functions
def setup_training_logger(**kwargs) -> TrainingLogger:
    """Set up a training logger with default settings."""
    return TrainingLogger(**kwargs)


def log_experiment_start(logger: logging.Logger, config: Dict[str, Any]):
    """Log experiment start with configuration."""
    logger.info("Starting new experiment")
    logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")


def log_experiment_end(logger: logging.Logger, results: Dict[str, Any]):
    """Log experiment end with results."""
    logger.info("Experiment completed")
    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
