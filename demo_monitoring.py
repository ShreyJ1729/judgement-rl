#!/usr/bin/env python3
"""
Demo script for real-time training monitoring.
Shows how to use the monitoring system with different training scenarios.
"""

import torch
import numpy as np
import time
from realtime_monitor import create_monitor_and_callback, RealtimeMonitor
from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from agent import PPOAgent


def demo_single_agent_monitoring():
    """
    Demo single agent training with real-time monitoring.
    """
    print("=== Demo: Single Agent Training with Real-time Monitoring ===")

    # Initialize environment and agent
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    agent = PPOAgent(
        state_encoder=state_encoder,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=256,
    )

    # Create monitor and callback
    monitor, callback = create_monitor_and_callback(["Demo Agent"])
    monitor.start_monitoring()

    print("Real-time monitoring started. Watch the graphs update!")
    print("Training for 100 episodes...")

    # Training loop
    num_episodes = 100
    episodes_per_update = 5
    epsilon = 0.2

    for episode in range(num_episodes):
        # Collect experience
        episode_data = agent.collect_experience(env, num_episodes=1, epsilon=epsilon)

        # Update monitor with episode data
        if episode_data:
            callback.on_episode_end("Demo Agent", agent, env, episode_data)

        # Train periodically
        if (episode + 1) % episodes_per_update == 0:
            loss_data = agent.train(batch_size=32)
            if loss_data:
                callback.on_training_step("Demo Agent", agent, loss_data)

        # Reduce exploration over time
        epsilon = max(0.01, epsilon * 0.995)

        # Print progress
        if (episode + 1) % 20 == 0:
            recent_rewards = agent.training_stats["episode_rewards"][-5:]
            avg_reward = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1}: Avg reward = {avg_reward:.2f}, Epsilon = {epsilon:.3f}"
            )

    # Save metrics
    monitor.save_metrics("demo_single_agent_metrics.json")
    monitor.stop_monitoring()

    print("Demo completed! Check demo_single_agent_metrics.json for saved metrics.")


def demo_multi_agent_monitoring():
    """
    Demo multi-agent training with real-time monitoring.
    """
    print("\n=== Demo: Multi-Agent Training with Real-time Monitoring ===")

    # Initialize environment
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Create multiple agents with different configurations
    agents = {}
    agent_configs = [
        ("Conservative Agent", {"learning_rate": 1e-4, "entropy_coef": 0.005}),
        ("Aggressive Agent", {"learning_rate": 5e-4, "entropy_coef": 0.02}),
        ("Balanced Agent", {"learning_rate": 3e-4, "entropy_coef": 0.01}),
    ]

    for agent_name, config in agent_configs:
        agents[agent_name] = PPOAgent(
            state_encoder=state_encoder,
            learning_rate=config["learning_rate"],
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=config["entropy_coef"],
            max_grad_norm=0.5,
            hidden_dim=256,
        )

    # Create monitor and callback
    agent_names = list(agents.keys())
    monitor, callback = create_monitor_and_callback(agent_names)
    monitor.start_monitoring()

    print("Real-time monitoring started for multiple agents!")
    print("Training for 50 episodes...")

    # Training loop
    num_episodes = 50
    episodes_per_update = 5
    epsilon = 0.3

    for episode in range(num_episodes):
        # Train each agent
        for agent_name, agent in agents.items():
            # Collect experience
            episode_data = agent.collect_experience(
                env, num_episodes=1, epsilon=epsilon
            )

            # Update monitor
            if episode_data:
                callback.on_episode_end(agent_name, agent, env, episode_data)

            # Train periodically
            if (episode + 1) % episodes_per_update == 0:
                loss_data = agent.train(batch_size=32)
                if loss_data:
                    callback.on_training_step(agent_name, agent, loss_data)

        # Reduce exploration over time
        epsilon = max(0.01, epsilon * 0.99)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Epsilon = {epsilon:.3f}")

    # Save metrics
    monitor.save_metrics("demo_multi_agent_metrics.json")
    monitor.stop_monitoring()

    print(
        "Multi-agent demo completed! Check demo_multi_agent_metrics.json for saved metrics."
    )


def demo_custom_metrics():
    """
    Demo custom metrics tracking.
    """
    print("\n=== Demo: Custom Metrics Tracking ===")

    # Create a custom monitor
    monitor = RealtimeMonitor(max_points=500, update_interval=0.5)
    monitor.add_agent("Custom Agent")
    monitor.start_monitoring()

    print("Custom monitoring started. Simulating training data...")

    # Simulate training data with custom metrics
    for i in range(100):
        # Simulate episode data
        episode_data = {
            "episode_reward": np.random.normal(10, 5),
            "declarations": [2, 3, 1, 2],
            "tricks_won": [2, 3, 1, 2],
            "total_tricks": 4,
            "round_count": 1,
        }

        # Simulate training step data
        loss_data = {
            "policy_loss": np.random.exponential(0.1),
            "value_loss": np.random.exponential(0.05),
            "entropy_loss": np.random.exponential(0.01),
        }

        # Update monitor directly
        monitor.update_metrics(
            "Custom Agent",
            {
                "episode_rewards": episode_data["episode_reward"],
                "bidding_accuracy": 0.75 + np.random.normal(0, 0.1),
                "avg_score_per_round": episode_data["episode_reward"],
                "policy_loss": loss_data["policy_loss"],
                "value_loss": loss_data["value_loss"],
                "entropy_loss": loss_data["entropy_loss"],
                "avg_bid_error": np.random.exponential(0.5),
                "trick_win_rate": 0.5 + np.random.normal(0, 0.15),
                "declaration_success_rate": 0.6 + np.random.normal(0, 0.1),
                "exploration_rate": max(0.01, 0.3 * (0.99**i)),
            },
        )

        time.sleep(0.1)  # Simulate training time

    # Save metrics
    monitor.save_metrics("demo_custom_metrics.json")
    monitor.stop_monitoring()

    print("Custom metrics demo completed!")


def main():
    """
    Main demo function.
    """
    print("Judgement Card Game - Real-time Monitoring Demo")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Demo 1: Single agent monitoring
        demo_single_agent_monitoring()

        # Demo 2: Multi-agent monitoring
        demo_multi_agent_monitoring()

        # Demo 3: Custom metrics
        demo_custom_metrics()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("Check the generated JSON files for detailed metrics.")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo failed with error: {e}")


if __name__ == "__main__":
    main()
