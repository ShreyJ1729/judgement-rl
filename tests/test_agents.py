"""
Tests for the agent modules.

This module tests the PPO agent, heuristic agent, and self-play trainer,
including network architecture, training, and action selection.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from judgement_rl.agents.agent import (
    PPOAgent,
    SelfPlayTrainer,
    PolicyNetwork,
    PPOMemory,
)
from judgement_rl.agents.heuristic_agent import HeuristicAgent


class TestPolicyNetwork:
    """Test suite for PolicyNetwork class."""

    def test_initialization(self, state_encoder, agent_config):
        """Test policy network initialization."""
        state_dim = state_encoder.get_state_dim()
        max_action_dim = state_encoder.get_action_dim(state_encoder.max_cards)

        network = PolicyNetwork(
            state_dim=state_dim,
            max_action_dim=max_action_dim,
            hidden_dim=agent_config.hidden_dim,
        )

        assert network.state_dim == state_dim
        assert network.max_action_dim == max_action_dim
        assert network.hidden_dim == agent_config.hidden_dim

    def test_forward_pass(self, state_encoder, agent_config, sample_encoded_state):
        """Test forward pass through the network."""
        state_dim = state_encoder.get_state_dim()
        max_action_dim = state_encoder.get_action_dim(state_encoder.max_cards)

        network = PolicyNetwork(
            state_dim=state_dim,
            max_action_dim=max_action_dim,
            hidden_dim=agent_config.hidden_dim,
        )

        # Convert to tensor
        state_tensor = torch.FloatTensor(sample_encoded_state).unsqueeze(0)

        # Forward pass
        action_logits, state_value = network(state_tensor)

        # Check output shapes
        assert action_logits.shape == (1, max_action_dim)
        assert state_value.shape == (1, 1)

        # Check output types
        assert isinstance(action_logits, torch.Tensor)
        assert isinstance(state_value, torch.Tensor)

    def test_action_probabilities(
        self, state_encoder, agent_config, sample_encoded_state
    ):
        """Test action probability calculation."""
        state_dim = state_encoder.get_state_dim()
        max_action_dim = state_encoder.get_action_dim(state_encoder.max_cards)

        network = PolicyNetwork(
            state_dim=state_dim,
            max_action_dim=max_action_dim,
            hidden_dim=agent_config.hidden_dim,
        )

        state_tensor = torch.FloatTensor(sample_encoded_state).unsqueeze(0)
        legal_actions = [0, 1, 2]  # Sample legal actions

        # Get action probabilities
        action_probs = network.get_action_probs(state_tensor, legal_actions)

        # Check output
        assert action_probs.shape == (1, max_action_dim)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(action_probs >= 0)

        # Check that illegal actions have zero probability
        illegal_actions = set(range(max_action_dim)) - set(legal_actions)
        for action in illegal_actions:
            assert action_probs[0, action] < 1e-6


class TestPPOMemory:
    """Test suite for PPOMemory class."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = PPOMemory(max_size=1000)
        assert memory.max_size == 1000
        assert len(memory) == 0

    def test_add_experience(self):
        """Test adding experiences to memory."""
        memory = PPOMemory(max_size=10)

        # Add experience
        state = np.random.randn(10)
        action = 1
        reward = 1.0
        next_state = np.random.randn(10)
        action_prob = 0.5
        state_value = 0.8
        done = False
        legal_actions = [0, 1, 2]

        memory.add(
            state,
            action,
            reward,
            next_state,
            action_prob,
            state_value,
            done,
            legal_actions,
        )

        assert len(memory) == 1
        assert memory.states[0] is state
        assert memory.actions[0] == action
        assert memory.rewards[0] == reward

    def test_memory_overflow(self):
        """Test memory overflow behavior."""
        memory = PPOMemory(max_size=3)

        # Add more experiences than max_size
        for i in range(5):
            memory.add(
                np.random.randn(10),
                i,
                1.0,
                np.random.randn(10),
                0.5,
                0.8,
                False,
                [0, 1, 2],
            )

        # Should only keep the last 3
        assert len(memory) == 3
        assert memory.actions[0] == 2  # First item should be from i=2

    def test_clear_memory(self):
        """Test clearing memory."""
        memory = PPOMemory(max_size=10)

        # Add some experiences
        for i in range(3):
            memory.add(
                np.random.randn(10),
                i,
                1.0,
                np.random.randn(10),
                0.5,
                0.8,
                False,
                [0, 1, 2],
            )

        assert len(memory) == 3

        # Clear memory
        memory.clear()
        assert len(memory) == 0


class TestPPOAgent:
    """Test suite for PPOAgent class."""

    def test_initialization(self, ppo_agent, agent_config):
        """Test PPO agent initialization."""
        assert ppo_agent.state_encoder is not None
        assert ppo_agent.state_dim > 0
        assert ppo_agent.max_action_dim > 0
        assert ppo_agent.learning_rate == agent_config.learning_rate
        assert ppo_agent.gamma == 0.99
        assert ppo_agent.clip_epsilon == 0.2

    def test_select_action(self, ppo_agent, sample_game_state):
        """Test action selection."""
        legal_actions = [0, 1, 2, 3]

        # Test with epsilon=0 (no exploration)
        action, prob, value = ppo_agent.select_action(
            sample_game_state, legal_actions, epsilon=0.0
        )

        assert action in legal_actions
        assert 0 <= prob <= 1
        assert isinstance(value, float)

        # Test with epsilon=1 (full exploration)
        action, prob, value = ppo_agent.select_action(
            sample_game_state, legal_actions, epsilon=1.0
        )

        assert action in legal_actions
        assert prob == 1.0 / len(legal_actions)  # Uniform probability

    def test_collect_experience(self, ppo_agent, mock_environment):
        """Test experience collection."""
        initial_memory_size = len(ppo_agent.memory)

        # Collect experience
        ppo_agent.collect_experience(mock_environment, num_episodes=1, epsilon=0.1)

        # Memory should have increased
        assert len(ppo_agent.memory) > initial_memory_size

    def test_compute_gae_returns(self, ppo_agent):
        """Test GAE return computation."""
        rewards = [1.0, 0.5, -0.5, 2.0]
        values = [0.8, 0.6, 0.4, 0.2]
        dones = [False, False, False, True]

        returns = ppo_agent.compute_gae_returns(rewards, values, dones)

        assert len(returns) == len(rewards)
        assert all(isinstance(r, float) for r in returns)

        # Returns should be reasonable
        assert all(-10 < r < 10 for r in returns)

    def test_training(self, ppo_agent, sample_training_data):
        """Test training process."""
        if len(sample_training_data) == 0:
            pytest.skip("No training data available")

        # Get initial training stats
        initial_stats = ppo_agent.get_training_stats()
        initial_losses = len(initial_stats["policy_losses"])

        # Train
        ppo_agent.train(batch_size=4, num_epochs=1)

        # Check that training occurred
        updated_stats = ppo_agent.get_training_stats()
        assert len(updated_stats["policy_losses"]) > initial_losses

    def test_save_load_model(self, ppo_agent, temp_dir):
        """Test model saving and loading."""
        model_path = create_test_model_path(temp_dir, "test_model.pth")

        # Save model
        ppo_agent.save_model(str(model_path))
        assert model_path.exists()

        # Create new agent and load model
        new_agent = PPOAgent(ppo_agent.state_encoder)
        new_agent.load_model(str(model_path))

        # Check that models are similar (not exact due to random initialization)
        # This is a basic check - in practice you'd compare specific weights
        assert new_agent.state_dim == ppo_agent.state_dim
        assert new_agent.max_action_dim == ppo_agent.max_action_dim

    def test_training_stats(self, ppo_agent):
        """Test training statistics tracking."""
        stats = ppo_agent.get_training_stats()

        required_keys = [
            "episode_rewards",
            "policy_losses",
            "value_losses",
            "entropy_losses",
            "total_losses",
        ]

        assert all(key in stats for key in required_keys)
        assert all(isinstance(stats[key], list) for key in required_keys)


class TestHeuristicAgent:
    """Test suite for HeuristicAgent class."""

    def test_initialization(self, heuristic_agent):
        """Test heuristic agent initialization."""
        assert heuristic_agent is not None

    def test_select_action(self, heuristic_agent, sample_game_state):
        """Test heuristic action selection."""
        legal_actions = [0, 1, 2, 3]

        action, prob, value = heuristic_agent.select_action(
            sample_game_state, legal_actions
        )

        assert action in legal_actions
        assert 0 <= prob <= 1
        assert isinstance(value, float)

    def test_bidding_strategy(self, heuristic_agent, sample_game_state):
        """Test bidding strategy."""
        # Test bidding phase
        sample_game_state["phase"] = "bidding"
        legal_actions = [0, 1, 2, 3, 4, 5, 6, 7]

        action, _, _ = heuristic_agent.select_action(sample_game_state, legal_actions)

        # Should bid a reasonable amount
        assert 0 <= action <= 7

    def test_playing_strategy(self, heuristic_agent, sample_game_state):
        """Test playing strategy."""
        # Test playing phase
        sample_game_state["phase"] = "playing"
        legal_actions = [0, 1, 2]

        action, _, _ = heuristic_agent.select_action(sample_game_state, legal_actions)

        assert action in legal_actions


class TestSelfPlayTrainer:
    """Test suite for SelfPlayTrainer class."""

    def test_initialization(self, self_play_trainer, agent_config):
        """Test self-play trainer initialization."""
        assert self_play_trainer.env is not None
        assert self_play_trainer.state_encoder is not None
        assert len(self_play_trainer.agents) == 2  # For test config
        assert self_play_trainer.learning_rate == agent_config.learning_rate

    def test_train_episode(self, self_play_trainer):
        """Test single episode training."""
        initial_stats = self_play_trainer.get_training_stats()
        initial_episodes = len(initial_stats["episode_rewards"])

        # Train one episode
        self_play_trainer.train_episode(epsilon=0.1)

        # Check that episode was completed
        updated_stats = self_play_trainer.get_training_stats()
        assert len(updated_stats["episode_rewards"]) > initial_episodes

    def test_training(self, self_play_trainer, training_config):
        """Test full training process."""
        initial_stats = self_play_trainer.get_training_stats()
        initial_episodes = len(initial_stats["episode_rewards"])

        # Train for a few episodes
        self_play_trainer.train(
            num_episodes=training_config.num_episodes,
            episodes_per_update=training_config.episodes_per_update,
            epsilon=training_config.epsilon,
            batch_size=training_config.batch_size,
        )

        # Check that training occurred
        updated_stats = self_play_trainer.get_training_stats()
        assert len(updated_stats["episode_rewards"]) > initial_episodes

    def test_save_best_agent(self, self_play_trainer, temp_dir):
        """Test saving the best agent."""
        model_path = create_test_model_path(temp_dir, "best_agent.pth")

        # Save best agent
        self_play_trainer.save_best_agent(str(model_path))
        assert model_path.exists()

    def test_agent_competition(self, self_play_trainer):
        """Test agent competition mechanism."""
        # Train a few episodes to establish some performance
        self_play_trainer.train(num_episodes=5, episodes_per_update=2, epsilon=0.1)

        # Check that there's a current best agent
        assert self_play_trainer.current_best_agent is not None
        assert 0 <= self_play_trainer.current_best_agent < len(self_play_trainer.agents)


class TestAgentIntegration:
    """Integration tests for agents with environment."""

    def test_ppo_agent_full_game(self, ppo_agent, environment):
        """Test PPO agent playing a full game."""
        state = environment.reset()

        done = False
        steps = 0
        max_steps = 50

        while not done and steps < max_steps:
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action, prob, value = ppo_agent.select_action(
                    state, legal_actions, epsilon=0.0
                )
                state, reward, done = environment.step(current_player, action)
                steps += 1
            else:
                break

        # Game should complete or reach max steps
        assert done or steps < max_steps

    def test_heuristic_agent_full_game(self, heuristic_agent, environment):
        """Test heuristic agent playing a full game."""
        state = environment.reset()

        done = False
        steps = 0
        max_steps = 50

        while not done and steps < max_steps:
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action, prob, value = heuristic_agent.select_action(
                    state, legal_actions
                )
                state, reward, done = environment.step(current_player, action)
                steps += 1
            else:
                break

        # Game should complete or reach max steps
        assert done or steps < max_steps

    def test_agent_comparison(self, ppo_agent, heuristic_agent, environment):
        """Test comparing different agent types."""
        # Play games with both agents
        ppo_rewards = []
        heuristic_rewards = []

        for _ in range(3):  # Play 3 games each
            # PPO agent game
            state = environment.reset()
            done = False
            ppo_reward = 0

            while not done:
                current_player = environment.current_player
                legal_actions = environment.get_legal_actions(current_player)

                if legal_actions:
                    action, _, _ = ppo_agent.select_action(
                        state, legal_actions, epsilon=0.0
                    )
                    state, reward, done = environment.step(current_player, action)
                    ppo_reward += reward
                else:
                    break

            ppo_rewards.append(ppo_reward)

            # Heuristic agent game
            state = environment.reset()
            done = False
            heuristic_reward = 0

            while not done:
                current_player = environment.current_player
                legal_actions = environment.get_legal_actions(current_player)

                if legal_actions:
                    action, _, _ = heuristic_agent.select_action(state, legal_actions)
                    state, reward, done = environment.step(current_player, action)
                    heuristic_reward += reward
                else:
                    break

            heuristic_rewards.append(heuristic_reward)

        # Both should complete games
        assert len(ppo_rewards) == 3
        assert len(heuristic_rewards) == 3
