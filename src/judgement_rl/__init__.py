"""
Judgement RL - Reinforcement Learning for the Judgement Card Game

A comprehensive implementation of reinforcement learning for the Judgement
card game using PPO (Proximal Policy Optimization) with self-play training.
"""

__version__ = "2.0.0"

# Core components
from .environment.judgement_env import JudgementEnv
from .utils.state_encoder import StateEncoder
from .agents.agent import PPOAgent, SelfPlayTrainer
from .agents.heuristic_agent import HeuristicAgent

# Configuration
from .config import (
    EnvironmentConfig,
    AgentConfig,
    TrainingConfig,
    MonitoringConfig,
    EvaluationConfig,
    GUIConfig,
    DEFAULT_ENV_CONFIG,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_MONITORING_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_GUI_CONFIG,
)

# Utilities
from .utils.logging import (
    setup_logger,
    get_logger,
    log_with_context,
    TrainingLogger,
    setup_training_logger,
    log_experiment_start,
    log_experiment_end,
)

# Public API
__all__ = [
    # Core components
    "JudgementEnv",
    "StateEncoder",
    "PPOAgent",
    "SelfPlayTrainer",
    "HeuristicAgent",
    # Configuration
    "EnvironmentConfig",
    "AgentConfig",
    "TrainingConfig",
    "MonitoringConfig",
    "EvaluationConfig",
    "GUIConfig",
    "DEFAULT_ENV_CONFIG",
    "DEFAULT_AGENT_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_MONITORING_CONFIG",
    "DEFAULT_EVALUATION_CONFIG",
    "DEFAULT_GUI_CONFIG",
    # Utilities
    "setup_logger",
    "get_logger",
    "log_with_context",
    "TrainingLogger",
    "setup_training_logger",
    "log_experiment_start",
    "log_experiment_end",
]
