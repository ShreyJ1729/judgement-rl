"""
Tests for utility modules.

This module tests the state encoder, logging utilities, and other helper functions.
"""

import pytest
import numpy as np
import torch
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.utils.logging import (
    setup_logger,
    get_logger,
    log_with_context,
    TrainingLogger,
    setup_training_logger,
    ColoredFormatter,
    JSONFormatter,
)


class TestStateEncoder:
    """Test suite for StateEncoder class."""

    def test_initialization(self, state_encoder, env_config):
        """Test state encoder initialization."""
        assert state_encoder.num_players == env_config.num_players
        assert state_encoder.max_cards == env_config.max_cards
        assert state_encoder.state_dim > 0
        assert state_encoder.action_dim > 0

    def test_get_state_dim(self, state_encoder):
        """Test state dimension calculation."""
        state_dim = state_encoder.get_state_dim()
        assert isinstance(state_dim, int)
        assert state_dim > 0

        # State dimension should be reasonable
        assert state_dim < 10000  # Not unreasonably large

    def test_get_action_dim(self, state_encoder):
        """Test action dimension calculation."""
        for num_cards in range(1, 8):
            action_dim = state_encoder.get_action_dim(num_cards)
            assert isinstance(action_dim, int)
            assert action_dim > 0
            assert action_dim >= num_cards + 1  # At least bidding actions

    def test_encode_state(self, state_encoder, sample_game_state):
        """Test state encoding."""
        encoded_state = state_encoder.encode_state(sample_game_state)

        # Check output type and shape
        assert isinstance(encoded_state, np.ndarray)
        assert encoded_state.shape[0] == state_encoder.get_state_dim()
        assert len(encoded_state.shape) == 1  # 1D array

        # Check for NaN values
        assert not np.isnan(encoded_state).any()

        # Check for reasonable values
        assert np.all(np.isfinite(encoded_state))

    def test_encode_hand(self, state_encoder):
        """Test hand encoding."""
        hand = ["A of Spades", "K of Hearts", "Q of Diamonds"]
        encoded_hand = state_encoder._encode_hand(hand)

        assert isinstance(encoded_hand, np.ndarray)
        assert len(encoded_hand) > 0
        assert not np.isnan(encoded_hand).any()

    def test_encode_card(self, state_encoder):
        """Test individual card encoding."""
        card = "A of Spades"
        encoded_card = state_encoder._encode_card(card)

        assert isinstance(encoded_card, np.ndarray)
        assert len(encoded_card) > 0
        assert not np.isnan(encoded_card).any()

    def test_encode_trump(self, state_encoder):
        """Test trump suit encoding."""
        trump_suits = ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]

        for trump in trump_suits:
            encoded_trump = state_encoder._encode_trump(trump)
            assert isinstance(encoded_trump, np.ndarray)
            assert len(encoded_trump) > 0
            assert not np.isnan(encoded_trump).any()

    def test_encode_declarations(self, state_encoder):
        """Test declarations encoding."""
        declarations = [2, 1, 0, 1]  # Sample bids
        encoded_declarations = state_encoder._encode_declarations(declarations)

        assert isinstance(encoded_declarations, np.ndarray)
        assert len(encoded_declarations) > 0
        assert not np.isnan(encoded_declarations).any()

    def test_encode_tricks_won(self, state_encoder):
        """Test tricks won encoding."""
        tricks_won = [1, 2, 0, 1]
        encoded_tricks = state_encoder._encode_tricks_won(tricks_won)

        assert isinstance(encoded_tricks, np.ndarray)
        assert len(encoded_tricks) > 0
        assert not np.isnan(encoded_tricks).any()

    def test_encode_current_trick(self, state_encoder):
        """Test current trick encoding."""
        current_trick = [(0, "A of Spades"), (1, "K of Hearts")]
        encoded_trick = state_encoder._encode_current_trick(current_trick)

        assert isinstance(encoded_trick, np.ndarray)
        assert len(encoded_trick) > 0
        assert not np.isnan(encoded_trick).any()

    def test_state_consistency(self, state_encoder, environment):
        """Test that encoded states are consistent throughout a game."""
        state = environment.reset()

        encoded_states = []
        for _ in range(5):  # Test a few steps
            encoded_state = state_encoder.encode_state(state)
            encoded_states.append(encoded_state)

            # Take a step
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action = legal_actions[0]
                state, reward, done = environment.step(current_player, action)
                if done:
                    break

        # All encoded states should have the same dimension
        assert all(enc.shape == encoded_states[0].shape for enc in encoded_states)

    def test_edge_cases(self, state_encoder):
        """Test edge cases in state encoding."""
        # Test empty hand
        empty_state = {
            "hand": [],
            "trump": "No Trump",
            "declarations": [None, None, None, None],
            "current_trick": [],
            "tricks_won": [0, 0, 0, 0],
            "round_cards": 7,
            "phase": "bidding",
            "current_player": 0,
        }

        encoded_state = state_encoder.encode_state(empty_state)
        assert isinstance(encoded_state, np.ndarray)
        assert not np.isnan(encoded_state).any()

        # Test with None declarations
        none_declarations_state = {
            "hand": ["A of Spades"],
            "trump": "No Trump",
            "declarations": [None, None, None, None],
            "current_trick": [],
            "tricks_won": [0, 0, 0, 0],
            "round_cards": 7,
            "phase": "bidding",
            "current_player": 0,
        }

        encoded_state = state_encoder.encode_state(none_declarations_state)
        assert isinstance(encoded_state, np.ndarray)
        assert not np.isnan(encoded_state).any()


class TestLogging:
    """Test suite for logging utilities."""

    def test_setup_logger(self, temp_dir):
        """Test logger setup."""
        log_file = "test.log"
        logger = setup_logger(
            name="test_logger",
            level=logging.INFO,
            log_file=log_file,
            log_dir=str(temp_dir),
            use_colors=True,
        )

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

        # Check that log file was created
        log_path = temp_dir / log_file
        assert log_path.exists()

    def test_get_logger(self):
        """Test getting existing logger."""
        logger = get_logger("test_get_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger"

    def test_log_with_context(self, temp_dir):
        """Test logging with context."""
        logger = setup_logger(
            name="context_logger", log_file="context.log", log_dir=str(temp_dir)
        )

        # Log with context
        log_with_context(logger, "Test message", episode=1, reward=10.5)

        # Check log file contains the message
        log_path = temp_dir / "context.log"
        with open(log_path, "r") as f:
            log_content = f.read()
            assert "Test message" in log_content

    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = JSONFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should be valid JSON
        import json

        parsed = json.loads(formatted)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_training_logger(self, temp_dir):
        """Test training logger."""
        logger = TrainingLogger(
            log_dir=str(temp_dir),
            experiment_name="test_experiment",
            use_tensorboard=False,
            use_wandb=False,
        )

        assert logger.experiment_name == "test_experiment"
        assert logger.logger is not None

        # Test logging methods
        logger.log_episode(1, 10.5, loss=0.1, accuracy=0.8)
        logger.log_training_step(1, policy_loss=0.05, value_loss=0.03)
        logger.log_evaluation(10, win_rate=0.7, avg_score=15.2)

        # Check that log file was created
        log_path = temp_dir / "test_experiment.log"
        assert log_path.exists()

        # Clean up
        logger.close()

    def test_setup_training_logger(self, temp_dir):
        """Test training logger setup function."""
        logger = setup_training_logger(
            log_dir=str(temp_dir), experiment_name="setup_test"
        )

        assert isinstance(logger, TrainingLogger)
        assert logger.experiment_name == "setup_test"

        logger.close()

    def test_logger_with_external_tools(self, temp_dir):
        """Test logger with external tools (mocked)."""
        # Test with tensorboard (should handle missing dependency gracefully)
        logger = TrainingLogger(
            log_dir=str(temp_dir),
            experiment_name="tensorboard_test",
            use_tensorboard=True,
            use_wandb=False,
        )

        # Should not crash even if tensorboard is not available
        logger.log_episode(1, 10.0)
        logger.close()

    def test_logger_file_output(self, temp_dir):
        """Test logger file output."""
        logger = TrainingLogger(log_dir=str(temp_dir), experiment_name="file_test")

        # Log some messages
        logger.log_episode(1, 10.0)
        logger.log_episode(2, 15.0)

        # Check log file
        log_path = temp_dir / "file_test.log"
        assert log_path.exists()

        with open(log_path, "r") as f:
            content = f.read()
            assert "Episode 1" in content
            assert "Episode 2" in content
            assert "reward=10.0" in content
            assert "reward=15.0" in content

        logger.close()

    def test_logger_levels(self, temp_dir):
        """Test different logging levels."""
        logger = setup_logger(
            name="level_test",
            level=logging.WARNING,
            log_file="level_test.log",
            log_dir=str(temp_dir),
        )

        # These should not appear in the log
        logger.debug("Debug message")
        logger.info("Info message")

        # These should appear
        logger.warning("Warning message")
        logger.error("Error message")

        # Check log file
        log_path = temp_dir / "level_test.log"
        with open(log_path, "r") as f:
            content = f.read()
            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Warning message" in content
            assert "Error message" in content


class TestUtilsIntegration:
    """Integration tests for utilities."""

    def test_encoder_with_environment(self, state_encoder, environment):
        """Test state encoder with real environment."""
        state = environment.reset()

        # Encode state
        encoded_state = state_encoder.encode_state(state)

        # Play a few steps and encode each state
        for _ in range(3):
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action = legal_actions[0]
                state, reward, done = environment.step(current_player, action)

                # Encode new state
                encoded_state = state_encoder.encode_state(state)
                assert isinstance(encoded_state, np.ndarray)
                assert not np.isnan(encoded_state).any()

                if done:
                    break

    def test_logger_with_training(self, temp_dir, ppo_agent, mock_environment):
        """Test logger with actual training."""
        # Set up logger
        logger = TrainingLogger(log_dir=str(temp_dir), experiment_name="training_test")

        # Collect some experience
        ppo_agent.collect_experience(mock_environment, num_episodes=1, epsilon=0.1)

        # Log training
        if len(ppo_agent.memory) > 0:
            logger.log_episode(1, 10.0)

            # Train
            ppo_agent.train(batch_size=4, num_epochs=1)

            # Log training step
            stats = ppo_agent.get_training_stats()
            if stats["policy_losses"]:
                logger.log_training_step(1, policy_loss=stats["policy_losses"][-1])

        logger.close()

        # Check that log file was created
        log_path = temp_dir / "training_test.log"
        assert log_path.exists()

    def test_encoder_consistency_across_games(self, state_encoder, environment):
        """Test that encoder produces consistent output across different games."""
        encoded_states = []

        for _ in range(3):  # Test 3 different games
            state = environment.reset()
            encoded_state = state_encoder.encode_state(state)
            encoded_states.append(encoded_state)

        # All states should have the same dimension
        assert all(enc.shape == encoded_states[0].shape for enc in encoded_states)

        # States should be different (different games)
        # But this is not guaranteed due to randomness, so we just check they're valid
        assert all(not np.isnan(enc).any() for enc in encoded_states)
