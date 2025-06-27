#!/usr/bin/env python3
"""
Training script for PPO agent with self-play for the Judgement card game.
Uses configuration file for hyperparameters and tqdm for progress tracking.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time

from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from agent import PPOAgent, SelfPlayTrainer
from realtime_monitor import create_monitor_and_callback
from config import TrainingConfig, EnvironmentConfig, AgentConfig


def plot_training_stats(stats: dict, save_path: str = None):
    """Plot training statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    if "episode_rewards" in stats and stats["episode_rewards"]:
        axes[0, 0].plot(stats["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

    # Policy losses
    if "policy_losses" in stats and stats["policy_losses"]:
        axes[0, 1].plot(stats["policy_losses"])
        axes[0, 1].set_title("Policy Loss")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)

    # Value losses
    if "value_losses" in stats and stats["value_losses"]:
        axes[1, 0].plot(stats["value_losses"])
        axes[1, 0].set_title("Value Loss")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)

    # Agent performances
    if "agent_performances" in stats and stats["agent_performances"]:
        agent_performances = np.array(stats["agent_performances"])
        for i in range(agent_performances.shape[1]):
            axes[1, 1].plot(agent_performances[:, i], label=f"Agent {i}")
        axes[1, 1].set_title("Individual Agent Performances")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()


def evaluate_agent(agent: PPOAgent, env: JudgementEnv, num_games: int = 100) -> dict:
    """Evaluate the agent's performance."""
    total_rewards = []
    total_tricks_won = []
    total_declarations_met = []

    for _ in range(num_games):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action (no exploration during evaluation)
            action, _, _ = agent.select_action(state, legal_actions, epsilon=0.0)

            # Take step
            next_state, reward, done = env.step(current_player, action)
            episode_reward += reward
            state = next_state

        # Calculate additional metrics
        tricks_won = env.tricks_won[0]  # Assuming agent is player 0
        declaration = env.declarations[0]
        declaration_met = 1 if tricks_won == declaration else 0

        total_rewards.append(episode_reward)
        total_tricks_won.append(tricks_won)
        total_declarations_met.append(declaration_met)

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_tricks_won": np.mean(total_tricks_won),
        "declaration_success_rate": np.mean(total_declarations_met),
    }


def train_selfplay(config: TrainingConfig):
    """Train using self-play with real-time monitoring."""
    print("=== Training with Self-Play ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=config.num_players, max_cards=config.max_cards)
    state_encoder = StateEncoder(
        num_players=config.num_players, max_cards=config.max_cards
    )

    # Initialize self-play trainer
    trainer = SelfPlayTrainer(
        env=env,
        state_encoder=state_encoder,
        num_agents=config.num_agents,
        learning_rate=config.learning_rate,
    )

    # Initialize real-time monitor if requested
    monitor = None
    callback = None
    if config.use_monitor:
        agent_names = [f"Self-Play Agent {i}" for i in range(config.num_agents)]
        monitor, callback = create_monitor_and_callback(
            agent_names, max_points=config.monitor_max_points
        )
        monitor.start_monitoring()
        print("Real-time monitoring enabled. Close the plot window to stop monitoring.")

    # Create models directory
    os.makedirs(config.models_dir, exist_ok=True)

    # Training loop with tqdm progress bar
    print(f"Training for {config.num_episodes} episodes...")
    start_time = time.time()

    epsilon = config.epsilon

    with tqdm(total=config.num_episodes, desc="Training Progress") as pbar:
        for episode in range(config.num_episodes):
            # Train one episode
            episode_rewards = trainer.train_episode(epsilon=epsilon)

            # Update monitor for each agent
            if callback:
                for i, agent in enumerate(trainer.agents):
                    agent_name = f"Self-Play Agent {i}"

                    # Extract episode metrics for this agent
                    episode_metrics = {
                        "episode_reward": (
                            episode_rewards[i] if i < len(episode_rewards) else 0
                        ),
                        "declarations": (
                            env.declarations.copy()
                            if hasattr(env, "declarations")
                            else []
                        ),
                        "tricks_won": (
                            env.tricks_won.copy() if hasattr(env, "tricks_won") else []
                        ),
                        "total_tricks": (
                            env.round_cards if hasattr(env, "round_cards") else 0
                        ),
                        "round_count": 1,
                    }
                    callback.on_episode_end(agent_name, agent, env, episode_metrics)

            # Train agents periodically
            if (episode + 1) % config.episodes_per_update == 0:
                for i, agent in enumerate(trainer.agents):
                    agent_name = f"Self-Play Agent {i}"
                    loss_data = agent.train(batch_size=config.batch_size)
                    if loss_data and callback:
                        callback.on_training_step(agent_name, agent, loss_data)

            # Update best agent based on recent performance
            if (episode + 1) % config.episodes_per_update == 0:
                recent_performances = trainer.training_stats["agent_performances"][
                    -config.episodes_per_update :
                ]
                avg_performances = [
                    np.mean([perf[i] for perf in recent_performances])
                    for i in range(config.num_agents)
                ]
                trainer.current_best_agent = np.argmax(avg_performances)

                # Copy best agent's policy to other agents (with noise)
                best_agent = trainer.agents[trainer.current_best_agent]
                for i, agent in enumerate(trainer.agents):
                    if i != trainer.current_best_agent:
                        for param, best_param in zip(
                            agent.policy_net.parameters(),
                            best_agent.policy_net.parameters(),
                        ):
                            param.data = (
                                best_param.data
                                + torch.randn_like(param.data) * config.policy_noise
                            )

            # Decay epsilon
            epsilon = max(config.min_epsilon, epsilon * config.epsilon_decay)

            # Save model periodically
            if (episode + 1) % config.save_interval == 0:
                model_path = os.path.join(
                    config.models_dir, f"selfplay_agent_episode_{episode + 1}.pth"
                )
                trainer.save_best_agent(model_path)

            # Evaluate periodically
            if (episode + 1) % config.eval_interval == 0:
                best_agent = trainer.agents[trainer.current_best_agent]
                eval_results = evaluate_agent(
                    best_agent, env, num_games=config.eval_games
                )
                pbar.set_postfix(
                    {
                        "Best Agent": trainer.current_best_agent,
                        "Avg Reward": f"{eval_results['avg_reward']:.2f}",
                        "Success Rate": f"{eval_results['declaration_success_rate']:.2f}",
                        "Epsilon": f"{epsilon:.3f}",
                    }
                )

            pbar.update(1)

    training_time = time.time() - start_time
    print(f"\nSelf-play training completed in {training_time:.2f} seconds")

    # Save final model
    final_model_path = os.path.join(config.models_dir, "selfplay_best_agent.pth")
    trainer.save_best_agent(final_model_path)
    print(f"Best agent saved to {final_model_path}")

    # Save monitoring data if available
    if monitor:
        monitor.save_metrics(
            os.path.join(config.models_dir, "selfplay_training_metrics.json")
        )
        monitor.stop_monitoring()

    # Plot training stats
    plot_training_stats(
        trainer.get_training_stats(),
        os.path.join(config.models_dir, "selfplay_training.png"),
    )

    # Final evaluation
    print("\n=== Final Evaluation ===")
    best_agent = trainer.agents[trainer.current_best_agent]
    eval_results = evaluate_agent(best_agent, env, num_games=config.eval_games)
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.3f}")

    return trainer


def main():
    """Main training function with command-line options."""
    parser = argparse.ArgumentParser(
        description="Train Judgement card game agents with self-play"
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Disable real-time monitoring"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate for the agent"
    )
    parser.add_argument(
        "--num-agents", type=int, default=4, help="Number of agents for self-play"
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_agents=args.num_agents,
        use_monitor=not args.no_monitor,
    )

    print("Judgement Card Game - Self-Play PPO Training")
    print("=" * 50)
    print(f"Episodes: {config.num_episodes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of agents: {config.num_agents}")
    print(f"Real-time monitoring: {'Enabled' if config.use_monitor else 'Disabled'}")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Train with self-play
    trainer = train_selfplay(config)

    print(
        "\nTraining completed! Check the 'models' directory for saved models and plots."
    )
    if config.use_monitor:
        print("Training metrics have been saved to JSON files in the models directory.")


if __name__ == "__main__":
    main()
