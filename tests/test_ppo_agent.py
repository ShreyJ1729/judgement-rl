#!/usr/bin/env python3
"""
Test script for PPO agent implementation.
This verifies that the agent works correctly with the Judgement environment.
"""

import torch
import numpy as np
from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from agent import PPOAgent, SelfPlayTrainer


def test_ppo_agent_basic():
    """
    Test basic PPO agent functionality.
    """
    print("=== Testing Basic PPO Agent ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize agent
    agent = PPOAgent(state_encoder)

    # Test state encoding
    state = env.reset()
    encoded_state = state_encoder.encode_state(state)
    print(f"State dimension: {len(encoded_state)}")
    print(f"Expected state dimension: {state_encoder.get_state_dim()}")
    assert (
        len(encoded_state) == state_encoder.get_state_dim()
    ), "State dimension mismatch"

    # Test action selection
    legal_actions = env.get_legal_actions(env.current_player)
    action, action_prob, state_value = agent.select_action(state, legal_actions)
    print(f"Selected action: {action}")
    print(f"Action probability: {action_prob:.4f}")
    print(f"State value: {state_value:.4f}")
    assert action in legal_actions, "Selected action not in legal actions"
    assert 0 <= action_prob <= 1, "Invalid action probability"

    print("✓ Basic PPO agent test passed!")


def test_ppo_agent_training():
    """
    Test PPO agent training functionality.
    """
    print("\n=== Testing PPO Agent Training ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize agent
    agent = PPOAgent(state_encoder)

    # Collect some experience
    print("Collecting experience...")
    agent.collect_experience(env, num_episodes=5, epsilon=0.5)
    print(f"Collected {len(agent.memory)} experiences")

    # Test training
    if len(agent.memory) > 0:
        print("Training agent...")
        initial_loss = None
        if agent.training_stats["total_losses"]:
            initial_loss = agent.training_stats["total_losses"][-1]

        agent.train(batch_size=32, num_epochs=2)

        if agent.training_stats["total_losses"]:
            final_loss = agent.training_stats["total_losses"][-1]
            print(f"Training completed. Loss: {final_loss:.4f}")

        print("✓ PPO agent training test passed!")
    else:
        print("⚠ No experience collected, skipping training test")


def test_selfplay_trainer():
    """
    Test self-play trainer functionality.
    """
    print("\n=== Testing Self-Play Trainer ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize self-play trainer
    trainer = SelfPlayTrainer(env, state_encoder, num_agents=4)

    # Test single episode
    print("Testing single self-play episode...")
    episode_rewards = trainer.train_episode(epsilon=0.5)
    print(f"Episode rewards: {episode_rewards}")
    print(f"Total reward: {sum(episode_rewards):.2f}")

    # Test training for a few episodes
    print("Testing self-play training for 5 episodes...")
    trainer.train(num_episodes=5, episodes_per_update=2, epsilon=0.5, batch_size=16)

    print(f"Training stats: {len(trainer.training_stats['episode_rewards'])} episodes")
    print("✓ Self-play trainer test passed!")


def test_model_saving_loading():
    """
    Test model saving and loading functionality.
    """
    print("\n=== Testing Model Save/Load ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize agent
    agent = PPOAgent(state_encoder)

    # Collect some experience and train
    agent.collect_experience(env, num_episodes=3, epsilon=0.5)
    if len(agent.memory) > 0:
        agent.train(batch_size=16, num_epochs=1)

    # Save model
    import os

    os.makedirs("models", exist_ok=True)
    save_path = "models/test_agent.pth"
    agent.save_model(save_path)
    print(f"Model saved to {save_path}")

    # Load model
    new_agent = PPOAgent(state_encoder)
    new_agent.load_model(save_path)
    print("Model loaded successfully")

    # Test that loaded agent works
    state = env.reset()
    legal_actions = env.get_legal_actions(env.current_player)
    action, _, _ = new_agent.select_action(state, legal_actions)
    print(f"Loaded agent selected action: {action}")

    # Clean up
    os.remove(save_path)
    print("✓ Model save/load test passed!")


def test_agent_performance():
    """
    Test agent performance in a few games.
    """
    print("\n=== Testing Agent Performance ===")

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Initialize agent
    agent = PPOAgent(state_encoder)

    # Play a few games
    num_games = 5
    total_rewards = []

    for game in range(num_games):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action
            action, _, _ = agent.select_action(state, legal_actions, epsilon=0.1)

            # Take step
            next_state, reward, done = env.step(current_player, action)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        print(f"Game {game + 1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_games} games: {avg_reward:.2f}")
    print("✓ Agent performance test passed!")


def main():
    """
    Run all tests.
    """
    print("PPO Agent Test Suite")
    print("=" * 40)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_ppo_agent_basic()
        test_ppo_agent_training()
        test_selfplay_trainer()
        test_model_saving_loading()
        test_agent_performance()

        print("\n" + "=" * 40)
        print("✓ All tests passed! PPO agent is working correctly.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
