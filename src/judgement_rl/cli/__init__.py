"""
Command-line interface for Judgement RL.

This package provides command-line tools for training, evaluation,
and interaction with the Judgement RL system.
"""

from .train import main as train_main
from .evaluate import main as evaluate_main
from .gui import main as gui_main
from .monitor import main as monitor_main

__all__ = [
    "train_main",
    "evaluate_main",
    "gui_main",
    "monitor_main",
]
