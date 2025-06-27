#!/usr/bin/env python3
"""
Configuration file for Judgement card game training.
Contains all hyperparameters and training variables.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Environment settings
    num_players: int = 4
    max_cards: int = 7

    # Agent hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256

    # Training parameters
    num_episodes: int = 100
    episodes_per_update: int = 10
    batch_size: int = 64
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    # Self-play specific
    num_agents: int = 4
    policy_noise: float = 0.01  # Noise when copying best agent policy

    # Monitoring
    use_monitor: bool = True
    monitor_update_interval: float = 1.0
    monitor_max_points: int = 1000

    # Model saving
    save_interval: int = 100  # Save model every N episodes
    models_dir: str = "models"

    # Evaluation
    eval_interval: int = 50  # Evaluate every N episodes
    eval_games: int = 100

    # Random seeds
    random_seed: int = 42

    # Logging
    log_interval: int = 10  # Print progress every N episodes


@dataclass
class EnvironmentConfig:
    """Configuration for game environment."""

    # Game rules
    num_players: int = 4
    max_cards: int = 7

    # Scoring
    exact_bid_bonus: int = 10
    bid_penalty_multiplier: int = 10

    # Trump suits rotation
    trump_suits: List[str] = None

    def __post_init__(self):
        if self.trump_suits is None:
            self.trump_suits = ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]


@dataclass
class AgentConfig:
    """Configuration for PPO agent."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3

    # PPO parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Memory
    memory_size: int = 10000

    # Exploration
    initial_epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_AGENT_CONFIG = AgentConfig()
