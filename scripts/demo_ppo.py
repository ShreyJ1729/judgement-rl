#!/usr/bin/env python3
"""
Demonstration script for PPO agent with self-play for the Judgement card game.
This shows how to use the implementation from Step 6.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent, SelfPlayTrainer


def quick_demo():
    """
    Quick demonstration of the PPO agent.
    """
    print("Judgement Card Game - PPO Agent Demo")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    print(
        f"Environment initialized with {env.num_players} players, max {env.max_cards} cards"
    )
    print(f"State dimension: {state_encoder.get_state_dim()}")
    print(f"Action dimension: {state_encoder.get_action_dim(7)}")

    # Initialize PPO agent
    agent = PPOAgent(
        state_encoder=state_encoder,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=128,  # Smaller for demo
    )

    print("\n=== Quick Training Demo ===")
    print("Training agent for 50 episodes...")

    start_time = time.time()

    # Quick training
    for episode in range(50):
        # Collect experience
        agent.collect_experience(env, num_episodes=1, epsilon=0.3)

        # Train every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.train(batch_size=32, num_epochs=2)

            # Print progress
            if (episode + 1) % 10 == 0:
                recent_rewards = agent.training_stats["episode_rewards"][-5:]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1}: Avg reward = {avg_reward:.2f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Test the trained agent
    print("\n=== Testing Trained Agent ===")

    test_rewards = []
    for game in range(10):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action (no exploration during testing)
            action, _, _ = agent.select_action(state, legal_actions, epsilon=0.0)

            # Take step
            next_state, reward, done = env.step(current_player, action)

            episode_reward += reward
            state = next_state

        test_rewards.append(episode_reward)
        print(f"Game {game + 1}: Reward = {episode_reward:.2f}")

    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage test reward: {avg_test_reward:.2f}")

    return agent


def selfplay_demo():
    """
    Demonstration of self-play training.
    """
    print("\n=== Self-Play Training Demo ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize self-play trainer
    trainer = SelfPlayTrainer(
        env=env, state_encoder=state_encoder, num_agents=4, learning_rate=3e-4
    )

    print("Training with self-play for 30 episodes...")
    start_time = time.time()

    # Quick self-play training
    trainer.train(num_episodes=30, episodes_per_update=5, epsilon=0.3, batch_size=32)

    training_time = time.time() - start_time
    print(f"Self-play training completed in {training_time:.2f} seconds")

    # Test the best agent
    print(f"\nBest agent: {trainer.current_best_agent}")
    best_agent = trainer.agents[trainer.current_best_agent]

    test_rewards = []
    for game in range(10):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action using best agent
            action, _, _ = best_agent.select_action(state, legal_actions, epsilon=0.0)

            # Take step
            next_state, reward, done = env.step(current_player, action)

            episode_reward += reward
            state = next_state

        test_rewards.append(episode_reward)
        print(f"Game {game + 1}: Reward = {episode_reward:.2f}")

    avg_test_reward = np.mean(test_rewards)
    print(f"\nBest agent average test reward: {avg_test_reward:.2f}")

    return trainer


def interactive_demo():
    """
    Interactive demonstration where you can watch the agent play.
    """
    print("\n=== Interactive Demo ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize agent
    agent = PPOAgent(state_encoder)

    # Quick training
    print("Training agent for 20 episodes...")
    for episode in range(20):
        agent.collect_experience(env, num_episodes=1, epsilon=0.3)
        if (episode + 1) % 5 == 0:
            agent.train(batch_size=32, num_epochs=1)

    print("Training complete! Watch the agent play...")

    # Play a game with detailed output
    state = env.reset()
    print(f"\nStarting new game with {env.round_cards} cards, trump: {env.trump}")
    print("Player hands:")
    for i, hand in enumerate(env.hands):
        print(f"Player {i}: {hand}")

    done = False
    while not done:
        current_player = env.current_player
        legal_actions = env.get_legal_actions(current_player)

        print(f"\n--- Player {current_player}'s turn ---")
        print(f"Hand: {env.hands[current_player]}")
        print(f"Legal actions: {legal_actions}")

        # Select action
        action, action_prob, state_value = agent.select_action(
            state, legal_actions, epsilon=0.0
        )

        print(
            f"Agent selects action {action} (prob: {action_prob:.3f}, value: {state_value:.3f})"
        )

        # Take step
        next_state, reward, done = env.step(current_player, action)

        print(f"Reward: {reward}")
        if done:
            print(f"Game finished! Final rewards: {env.tricks_won}")

        state = next_state

    print("\nInteractive demo complete!")


def main():
    """
    Main demonstration function.
    """
    print("Judgement Card Game - PPO Agent Demonstrations")
    print("=" * 60)

    try:
        # Quick demo
        agent = quick_demo()

        # Self-play demo
        trainer = selfplay_demo()

        # Interactive demo
        interactive_demo()

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("\nKey features demonstrated:")
        print("- PPO agent training with experience collection")
        print("- Self-play training with multiple agents")
        print("- Action selection with legal action masking")
        print("- State encoding and neural network inference")
        print("- Model evaluation and performance testing")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
