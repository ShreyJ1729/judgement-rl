"""
Integration tests for Judgement RL.

This module tests the complete system integration, including full training
workflows, end-to-end scenarios, and system-wide functionality.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent, SelfPlayTrainer
from judgement_rl.agents.heuristic_agent import HeuristicAgent
from judgement_rl.config import (
    EnvironmentConfig,
    AgentConfig,
    TrainingConfig,
    DEFAULT_ENV_CONFIG,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_TRAINING_CONFIG,
)
from judgement_rl.utils.logging import TrainingLogger


class TestFullTrainingWorkflow:
    """Test complete training workflows."""

    def test_single_agent_training(self, temp_dir):
        """Test complete single agent training workflow."""
        # Set up components
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Set up logger
        logger = TrainingLogger(
            log_dir=str(temp_dir), experiment_name="single_agent_test"
        )

        # Training loop
        num_episodes = 10
        episodes_per_update = 2

        for episode in range(num_episodes):
            # Collect experience
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)

            # Log episode
            if len(agent.memory) > 0:
                recent_rewards = agent.training_stats["episode_rewards"][-1:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                logger.log_episode(episode + 1, avg_reward)

            # Train periodically
            if (episode + 1) % episodes_per_update == 0:
                agent.train(batch_size=16, num_epochs=1)

                # Log training metrics
                stats = agent.get_training_stats()
                if stats["policy_losses"]:
                    logger.log_training_step(
                        episode + 1,
                        policy_loss=stats["policy_losses"][-1],
                        value_loss=(
                            stats["value_losses"][-1] if stats["value_losses"] else 0.0
                        ),
                    )

        # Check that training occurred
        stats = agent.get_training_stats()
        assert len(stats["episode_rewards"]) > 0
        assert len(stats["policy_losses"]) > 0

        # Save model
        model_path = temp_dir / "trained_agent.pth"
        agent.save_model(str(model_path))
        assert model_path.exists()

        logger.close()

    def test_self_play_training(self, temp_dir):
        """Test complete self-play training workflow."""
        # Set up components
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        trainer = SelfPlayTrainer(
            env=env,
            state_encoder=encoder,
            num_agents=2,  # Small for testing
            learning_rate=1e-3,
        )

        # Set up logger
        logger = TrainingLogger(log_dir=str(temp_dir), experiment_name="self_play_test")

        # Training loop
        num_episodes = 10
        episodes_per_update = 2

        for episode in range(num_episodes):
            # Train episode
            trainer.train_episode(epsilon=0.1)

            # Log episode
            stats = trainer.get_training_stats()
            if stats["episode_rewards"]:
                recent_reward = stats["episode_rewards"][-1]
                logger.log_episode(episode + 1, recent_reward)

            # Train periodically
            if (episode + 1) % episodes_per_update == 0:
                trainer.train(
                    num_episodes=1, episodes_per_update=1, epsilon=0.1, batch_size=16
                )

        # Check that training occurred
        stats = trainer.get_training_stats()
        assert len(stats["episode_rewards"]) > 0

        # Save best agent
        model_path = temp_dir / "best_agent.pth"
        trainer.save_best_agent(str(model_path))
        assert model_path.exists()

        logger.close()

    def test_agent_evaluation(self, temp_dir):
        """Test agent evaluation workflow."""
        # Train an agent first
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Quick training
        for episode in range(5):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 2 == 0:
                agent.train(batch_size=16, num_epochs=1)

        # Evaluate against random play
        num_eval_games = 10
        eval_rewards = []

        for game in range(num_eval_games):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                current_player = env.current_player
                legal_actions = env.get_legal_actions(current_player)

                if legal_actions:
                    # Use trained agent for player 0, random for others
                    if current_player == 0:
                        action, _, _ = agent.select_action(
                            state, legal_actions, epsilon=0.0
                        )
                    else:
                        action = np.random.choice(legal_actions)

                    state, reward, done = env.step(current_player, action)
                    if current_player == 0:
                        episode_reward += reward
                else:
                    break

            eval_rewards.append(episode_reward)

        # Check evaluation results
        assert len(eval_rewards) == num_eval_games
        avg_reward = np.mean(eval_rewards)
        assert isinstance(avg_reward, float)
        assert -1000 < avg_reward < 1000  # Reasonable range

    def test_model_persistence(self, temp_dir):
        """Test complete model save/load workflow."""
        # Train an agent
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Quick training
        for episode in range(3):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 2 == 0:
                agent.train(batch_size=16, num_epochs=1)

        # Save model
        model_path = temp_dir / "persistence_test.pth"
        agent.save_model(str(model_path))
        assert model_path.exists()

        # Load model in new agent
        new_agent = PPOAgent(encoder)
        new_agent.load_model(str(model_path))

        # Test that both agents produce similar outputs
        state = env.reset()
        legal_actions = env.get_legal_actions(0)

        if legal_actions:
            action1, prob1, value1 = agent.select_action(
                state, legal_actions, epsilon=0.0
            )
            action2, prob2, value2 = new_agent.select_action(
                state, legal_actions, epsilon=0.0
            )

            # Actions should be the same (deterministic)
            assert action1 == action2
            assert abs(prob1 - prob2) < 1e-6
            assert abs(value1 - value2) < 1e-6


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    def test_complete_game_with_trained_agent(self, temp_dir):
        """Test a complete game with a trained agent."""
        # Train an agent
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Quick training
        for episode in range(3):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 2 == 0:
                agent.train(batch_size=16, num_epochs=1)

        # Play a complete game
        state = env.reset()
        game_history = []
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            if legal_actions:
                # Use trained agent for player 0, random for others
                if current_player == 0:
                    action, prob, value = agent.select_action(
                        state, legal_actions, epsilon=0.0
                    )
                else:
                    action = np.random.choice(legal_actions)
                    prob = 1.0 / len(legal_actions)
                    value = 0.0

                # Record action
                game_history.append(
                    {
                        "player": current_player,
                        "action": action,
                        "prob": prob,
                        "value": value,
                        "legal_actions": legal_actions.copy(),
                    }
                )

                # Take step
                state, reward, done = env.step(current_player, action)
                steps += 1
            else:
                break

        # Check game completion
        assert done or steps < max_steps
        assert len(game_history) > 0

        # Check that all hands are empty
        assert all(len(hand) == 0 for hand in env.hands)

        # Check game history consistency
        for entry in game_history:
            assert entry["action"] in entry["legal_actions"]
            assert 0 <= entry["prob"] <= 1
            assert isinstance(entry["value"], float)

    def test_agent_vs_agent_competition(self, temp_dir):
        """Test competition between different agent types."""
        # Create different agents
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)

        # Train a PPO agent
        ppo_agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        for episode in range(3):
            ppo_agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 2 == 0:
                ppo_agent.train(batch_size=16, num_epochs=1)

        # Create heuristic agent
        heuristic_agent = HeuristicAgent()

        # Play games between agents
        num_games = 5
        ppo_wins = 0
        heuristic_wins = 0

        for game in range(num_games):
            state = env.reset()
            done = False
            ppo_score = 0
            heuristic_score = 0
            steps = 0
            max_steps = 100

            while not done and steps < max_steps:
                current_player = env.current_player
                legal_actions = env.get_legal_actions(current_player)

                if legal_actions:
                    # Assign agents to players
                    if current_player == 0:
                        action, _, _ = ppo_agent.select_action(
                            state, legal_actions, epsilon=0.0
                        )
                    elif current_player == 1:
                        action, _, _ = heuristic_agent.select_action(
                            state, legal_actions
                        )
                    else:
                        action = np.random.choice(legal_actions)

                    state, reward, done = env.step(current_player, action)

                    # Track scores
                    if current_player == 0:
                        ppo_score += reward
                    elif current_player == 1:
                        heuristic_score += reward

                    steps += 1
                else:
                    break

            # Determine winner
            if ppo_score > heuristic_score:
                ppo_wins += 1
            elif heuristic_score > ppo_score:
                heuristic_wins += 1

        # Check that games were played
        assert ppo_wins + heuristic_wins <= num_games
        assert ppo_wins >= 0
        assert heuristic_wins >= 0

    def test_configuration_integration(self):
        """Test integration with configuration system."""
        # Create custom configurations
        env_config = EnvironmentConfig(
            num_players=3, max_cards=5, exact_bid_bonus=15, bid_penalty_multiplier=8
        )

        agent_config = AgentConfig(
            hidden_dim=128, learning_rate=1e-4, gamma=0.95, clip_epsilon=0.15
        )

        training_config = TrainingConfig(
            num_episodes=20, episodes_per_update=4, batch_size=32, epsilon=0.05
        )

        # Create components with custom configs
        env = JudgementEnv(
            num_players=env_config.num_players, max_cards=env_config.max_cards
        )

        encoder = StateEncoder(
            num_players=env_config.num_players, max_cards=env_config.max_cards
        )

        agent = PPOAgent(
            state_encoder=encoder,
            learning_rate=agent_config.learning_rate,
            hidden_dim=agent_config.hidden_dim,
            gamma=agent_config.gamma,
            clip_epsilon=agent_config.clip_epsilon,
        )

        # Test that configurations are applied
        assert env.num_players == env_config.num_players
        assert env.max_cards == env_config.max_cards
        assert agent.learning_rate == agent_config.learning_rate
        assert agent.gamma == agent_config.gamma
        assert agent.clip_epsilon == agent_config.clip_epsilon

        # Quick training test
        for episode in range(training_config.num_episodes):
            agent.collect_experience(
                env, num_episodes=1, epsilon=training_config.epsilon
            )
            if (episode + 1) % training_config.episodes_per_update == 0:
                agent.train(batch_size=training_config.batch_size, num_epochs=1)

        # Check that training occurred
        stats = agent.get_training_stats()
        assert len(stats["episode_rewards"]) > 0


class TestSystemRobustness:
    """Test system robustness and error handling."""

    def test_memory_management(self):
        """Test memory management under load."""
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder,
            learning_rate=1e-3,
            hidden_dim=64,
            memory_size=100,  # Small memory for testing
        )

        # Collect many experiences to test memory overflow
        for episode in range(20):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)

            # Memory should not exceed max size
            assert len(agent.memory) <= agent.memory.max_size

        # Train to clear memory
        agent.train(batch_size=16, num_epochs=1)

        # Memory should be cleared after training
        assert len(agent.memory) == 0

    def test_error_recovery(self):
        """Test system recovery from errors."""
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Collect some experience
        agent.collect_experience(env, num_episodes=1, epsilon=0.1)

        # Simulate training with insufficient data
        if len(agent.memory) > 0:
            # This should not crash
            agent.train(batch_size=len(agent.memory) + 10, num_epochs=1)

        # System should still be functional
        state = env.reset()
        legal_actions = env.get_legal_actions(0)

        if legal_actions:
            action, prob, value = agent.select_action(state, legal_actions, epsilon=0.0)
            assert action in legal_actions
            assert 0 <= prob <= 1

    def test_concurrent_access(self):
        """Test concurrent access to shared resources."""
        import threading
        import time

        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Function for concurrent access
        def collect_experience():
            for _ in range(3):
                agent.collect_experience(env, num_episodes=1, epsilon=0.1)
                time.sleep(0.01)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=collect_experience)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # System should still be functional
        assert len(agent.memory) > 0

        # Should be able to train
        agent.train(batch_size=16, num_epochs=1)


class TestPerformance:
    """Test system performance characteristics."""

    @pytest.mark.slow
    def test_training_performance(self, temp_dir):
        """Test training performance with larger scale."""
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=128, memory_size=5000
        )

        import time

        start_time = time.time()

        # Larger training run
        num_episodes = 50
        for episode in range(num_episodes):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 5 == 0:
                agent.train(batch_size=64, num_epochs=2)

        end_time = time.time()
        training_time = end_time - start_time

        # Check performance
        assert training_time < 60  # Should complete within 60 seconds
        assert len(agent.training_stats["episode_rewards"]) > 0

        # Check memory usage
        assert len(agent.memory) <= agent.memory.max_size

    def test_inference_performance(self):
        """Test inference performance."""
        env = JudgementEnv(num_players=4, max_cards=7)
        encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_encoder=encoder, learning_rate=1e-3, hidden_dim=64, memory_size=1000
        )

        # Quick training
        for episode in range(2):
            agent.collect_experience(env, num_episodes=1, epsilon=0.1)
            if (episode + 1) % 2 == 0:
                agent.train(batch_size=16, num_epochs=1)

        # Test inference speed
        import time

        state = env.reset()
        legal_actions = env.get_legal_actions(0)

        if legal_actions:
            start_time = time.time()

            # Multiple inference calls
            for _ in range(100):
                action, prob, value = agent.select_action(
                    state, legal_actions, epsilon=0.0
                )

            end_time = time.time()
            inference_time = end_time - start_time

            # Should be fast
            assert inference_time < 5.0  # 100 inferences in under 5 seconds
            assert action in legal_actions
