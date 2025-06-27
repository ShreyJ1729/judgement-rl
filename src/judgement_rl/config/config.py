"""
Configuration classes for Judgement RL.

This module defines all configuration dataclasses used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration for the game environment."""

    # Game rules
    num_players: int = 4
    max_cards: int = 7

    # Scoring parameters
    exact_bid_bonus: int = 10
    bid_penalty_multiplier: int = 10

    # Trump suits rotation
    trump_suits: List[str] = field(
        default_factory=lambda: ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]
    )

    # Game flow
    skip_one_card_round: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_players < 2:
            raise ValueError("num_players must be at least 2")
        if self.max_cards < 1:
            raise ValueError("max_cards must be at least 1")
        if self.exact_bid_bonus < 0:
            raise ValueError("exact_bid_bonus must be non-negative")


@dataclass
class AgentConfig:
    """Configuration for PPO agent."""

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"

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

    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Environment settings
    num_players: int = 4
    max_cards: int = 7

    # Training parameters
    num_episodes: int = 1000
    episodes_per_update: int = 10
    batch_size: int = 64
    num_epochs: int = 4

    # Exploration
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    # Self-play specific
    use_self_play: bool = True
    num_agents: int = 4
    policy_noise: float = 0.01

    # Model saving
    save_interval: int = 100
    models_dir: str = "models"
    save_best_only: bool = True

    # Evaluation
    eval_interval: int = 50
    eval_games: int = 100

    # Random seeds
    random_seed: int = 42

    # Logging
    log_interval: int = 10
    verbose: bool = True

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if self.episodes_per_update <= 0:
            raise ValueError("episodes_per_update must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring."""

    # Monitoring settings
    enabled: bool = True
    update_interval: float = 1.0  # seconds
    max_points: int = 1000

    # Metrics to track
    track_rewards: bool = True
    track_losses: bool = True
    track_actions: bool = True
    track_epsilon: bool = True

    # Visualization
    plot_interval: int = 10
    save_plots: bool = True
    plots_dir: str = "plots"

    # Logging
    log_to_file: bool = True
    log_dir: str = "logs"

    # External tools
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "judgement-rl"
    wandb_entity: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Evaluation settings
    num_games: int = 100
    num_opponents: int = 3

    # Opponent types
    opponent_types: List[str] = field(
        default_factory=lambda: ["random", "heuristic", "trained"]
    )

    # Metrics
    track_win_rate: bool = True
    track_average_score: bool = True
    track_bid_accuracy: bool = True
    track_trick_win_rate: bool = True

    # Output
    save_results: bool = True
    results_dir: str = "evaluation_results"
    verbose: bool = True

    # Comparison
    compare_models: bool = False
    baseline_model: Optional[str] = None


@dataclass
class GUIConfig:
    """Configuration for GUI interface."""

    # Window settings
    window_width: int = 1200
    window_height: int = 800
    window_title: str = "Judgement RL - Play Against AI"

    # Game settings
    default_ai_model: str = "models/best_agent.pth"
    ai_difficulty: str = "medium"  # "easy", "medium", "hard"

    # Display settings
    card_width: int = 80
    card_height: int = 120
    animation_speed: float = 0.5

    # Features
    show_probabilities: bool = True
    show_ai_thinking: bool = True
    auto_play: bool = False


# Default configurations
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_AGENT_CONFIG = AgentConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MONITORING_CONFIG = MonitoringConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_GUI_CONFIG = GUIConfig()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def save_config_to_file(config: Any, config_path: str):
    """Save configuration to a YAML file."""
    import yaml
    from dataclasses import asdict

    config_dict = asdict(config)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Any, override_config: Dict[str, Any]) -> Any:
    """Merge a base configuration with override values."""
    import copy

    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if hasattr(merged, key):
            setattr(merged, key, value)

    return merged
