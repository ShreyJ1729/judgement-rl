"""
Configuration management for Judgement RL.

This module provides dataclass-based configuration for all components
of the Judgement RL system.
"""

from .config import (
    TrainingConfig,
    EnvironmentConfig,
    AgentConfig,
    MonitoringConfig,
    EvaluationConfig,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_ENV_CONFIG,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_MONITORING_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
)

__all__ = [
    "TrainingConfig",
    "EnvironmentConfig",
    "AgentConfig",
    "MonitoringConfig",
    "EvaluationConfig",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_ENV_CONFIG",
    "DEFAULT_AGENT_CONFIG",
    "DEFAULT_MONITORING_CONFIG",
    "DEFAULT_EVALUATION_CONFIG",
]
