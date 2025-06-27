"""
Command-line interface for training Judgement RL agents.

This module provides a comprehensive CLI for training single agents
and self-play training with proper configuration management.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent, SelfPlayTrainer
from judgement_rl.config import (
    EnvironmentConfig,
    AgentConfig,
    TrainingConfig,
    DEFAULT_ENV_CONFIG,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_TRAINING_CONFIG,
)
from judgement_rl.utils.logging import TrainingLogger, setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Judgement RL agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training mode
    parser.add_argument(
        "--mode",
        choices=["single", "self-play"],
        default="single",
        help="Training mode: single agent or self-play",
    )

    # Environment settings
    parser.add_argument(
        "--num-players",
        type=int,
        default=DEFAULT_ENV_CONFIG.num_players,
        help="Number of players in the game",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=DEFAULT_ENV_CONFIG.max_cards,
        help="Maximum number of cards per round",
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_episodes,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.episodes_per_update,
        help="Episodes per training update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_epochs,
        help="Number of training epochs per update",
    )

    # Agent parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_AGENT_CONFIG.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_AGENT_CONFIG.hidden_dim,
        help="Hidden dimension for neural network",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=DEFAULT_AGENT_CONFIG.memory_size,
        help="Memory buffer size",
    )

    # Self-play specific
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_agents,
        help="Number of agents for self-play",
    )

    # Exploration
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.epsilon,
        help="Exploration epsilon",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=DEFAULT_AGENT_CONFIG.epsilon_decay,
        help="Epsilon decay rate",
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=DEFAULT_AGENT_CONFIG.min_epsilon,
        help="Minimum epsilon value",
    )

    # Model saving
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.save_interval,
        help="Save model every N episodes",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_TRAINING_CONFIG.models_dir,
        help="Directory to save models",
    )
    parser.add_argument(
        "--model-name", type=str, default="trained_agent.pth", help="Model filename"
    )

    # Logging and monitoring
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory for log files"
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Experiment name for logging"
    )
    parser.add_argument(
        "--use-tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.random_seed,
        help="Random seed for reproducibility",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_TRAINING_CONFIG.device,
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    return parser.parse_args()


def setup_environment(args) -> tuple:
    """Set up environment and encoder."""
    env = JudgementEnv(num_players=args.num_players, max_cards=args.max_cards)

    encoder = StateEncoder(num_players=args.num_players, max_cards=args.max_cards)

    return env, encoder


def setup_agent(encoder, args) -> PPOAgent:
    """Set up PPO agent."""
    return PPOAgent(
        state_encoder=encoder,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        memory_size=args.memory_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )


def setup_self_play_trainer(env, encoder, args) -> SelfPlayTrainer:
    """Set up self-play trainer."""
    return SelfPlayTrainer(
        env=env,
        state_encoder=encoder,
        num_agents=args.num_agents,
        learning_rate=args.learning_rate,
    )


def train_single_agent(args, logger):
    """Train a single agent."""
    logger.info("Setting up single agent training...")

    # Set up components
    env, encoder = setup_environment(args)
    agent = setup_agent(encoder, args)

    logger.info(f"Training for {args.episodes} episodes")
    logger.info(f"Episodes per update: {args.episodes_per_update}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Training loop
    current_epsilon = args.epsilon

    for episode in range(args.episodes):
        # Collect experience
        agent.collect_experience(env, num_episodes=1, epsilon=current_epsilon)

        # Log episode
        if len(agent.memory) > 0:
            recent_rewards = agent.training_stats["episode_rewards"][-1:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            logger.log_episode(episode + 1, avg_reward, epsilon=current_epsilon)

        # Train periodically
        if (episode + 1) % args.episodes_per_update == 0:
            agent.train(batch_size=args.batch_size, num_epochs=args.num_epochs)

            # Log training metrics
            stats = agent.get_training_stats()
            if stats["policy_losses"]:
                logger.log_training_step(
                    episode + 1,
                    policy_loss=stats["policy_losses"][-1],
                    value_loss=(
                        stats["value_losses"][-1] if stats["value_losses"] else 0.0
                    ),
                    entropy_loss=(
                        stats["entropy_losses"][-1] if stats["entropy_losses"] else 0.0
                    ),
                )

        # Save model periodically
        if (episode + 1) % args.save_interval == 0:
            model_path = Path(args.models_dir) / f"agent_episode_{episode + 1}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save_model(str(model_path))
            logger.info(f"Saved model to {model_path}")

        # Decay epsilon
        current_epsilon = max(args.min_epsilon, current_epsilon * args.epsilon_decay)

    # Save final model
    final_model_path = Path(args.models_dir) / args.model_name
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(final_model_path))
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    return agent


def train_self_play(args, logger):
    """Train with self-play."""
    logger.info("Setting up self-play training...")

    # Set up components
    env, encoder = setup_environment(args)
    trainer = setup_self_play_trainer(env, encoder, args)

    logger.info(f"Training with {args.num_agents} agents for {args.episodes} episodes")
    logger.info(f"Episodes per update: {args.episodes_per_update}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Training loop
    current_epsilon = args.epsilon

    for episode in range(args.episodes):
        # Train episode
        trainer.train_episode(epsilon=current_epsilon)

        # Log episode
        stats = trainer.get_training_stats()
        if stats["episode_rewards"]:
            recent_reward = stats["episode_rewards"][-1]
            logger.log_episode(episode + 1, recent_reward, epsilon=current_epsilon)

        # Train periodically
        if (episode + 1) % args.episodes_per_update == 0:
            trainer.train(
                num_episodes=1,
                episodes_per_update=1,
                epsilon=current_epsilon,
                batch_size=args.batch_size,
            )

        # Save best agent periodically
        if (episode + 1) % args.save_interval == 0:
            model_path = Path(args.models_dir) / f"best_agent_episode_{episode + 1}.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_best_agent(str(model_path))
            logger.info(f"Saved best agent to {model_path}")

        # Decay epsilon
        current_epsilon = max(args.min_epsilon, current_epsilon * args.epsilon_decay)

    # Save final best agent
    final_model_path = Path(args.models_dir) / args.model_name
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_best_agent(str(final_model_path))
    logger.info(f"Self-play training completed. Best agent saved to {final_model_path}")

    return trainer


def main():
    """Main training function."""
    args = parse_args()

    # Set up logging
    if args.experiment_name is None:
        args.experiment_name = f"training_{args.mode}_{args.episodes}episodes"

    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
    )

    # Set up console logger
    console_logger = setup_logger(
        name="console", level="INFO" if args.verbose else "WARNING", use_colors=True
    )

    try:
        # Log configuration
        logger.logger.info("Starting training with configuration:")
        logger.logger.info(f"Mode: {args.mode}")
        logger.logger.info(f"Episodes: {args.episodes}")
        logger.logger.info(f"Players: {args.num_players}")
        logger.logger.info(f"Max cards: {args.max_cards}")
        logger.logger.info(f"Learning rate: {args.learning_rate}")
        logger.logger.info(f"Hidden dim: {args.hidden_dim}")

        # Set random seed
        import torch
        import numpy as np

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Train based on mode
        if args.mode == "single":
            agent = train_single_agent(args, logger)
            console_logger.info("Single agent training completed successfully")
        else:  # self-play
            trainer = train_self_play(args, logger)
            console_logger.info("Self-play training completed successfully")

        # Log final statistics
        logger.logger.info("Training completed successfully")

    except Exception as e:
        logger.logger.error(f"Training failed: {e}")
        console_logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
