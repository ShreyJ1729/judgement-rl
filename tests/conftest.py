"""
Pytest configuration and fixtures for Judgement RL tests.

This module provides common fixtures and test configuration used across
all test modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def env_config():
    """Default environment configuration for tests."""
    return EnvironmentConfig(
        num_players=4, max_cards=7, exact_bid_bonus=10, bid_penalty_multiplier=10
    )


@pytest.fixture
def agent_config():
    """Default agent configuration for tests."""
    return AgentConfig(
        hidden_dim=64, learning_rate=1e-3, memory_size=1000  # Smaller for faster tests
    )


@pytest.fixture
def training_config():
    """Default training configuration for tests."""
    return TrainingConfig(
        num_episodes=10,  # Small for fast tests
        episodes_per_update=2,
        batch_size=16,
        num_epochs=1,
    )


@pytest.fixture
def environment(env_config):
    """Create a test environment instance."""
    return JudgementEnv(
        num_players=env_config.num_players, max_cards=env_config.max_cards
    )


@pytest.fixture
def state_encoder(env_config):
    """Create a test state encoder instance."""
    return StateEncoder(
        num_players=env_config.num_players, max_cards=env_config.max_cards
    )


@pytest.fixture
def ppo_agent(state_encoder, agent_config):
    """Create a test PPO agent instance."""
    return PPOAgent(
        state_encoder=state_encoder,
        learning_rate=agent_config.learning_rate,
        hidden_dim=agent_config.hidden_dim,
        memory_size=agent_config.memory_size,
    )


@pytest.fixture
def heuristic_agent():
    """Create a test heuristic agent instance."""
    return HeuristicAgent()


@pytest.fixture
def self_play_trainer(environment, state_encoder, agent_config):
    """Create a test self-play trainer instance."""
    return SelfPlayTrainer(
        env=environment,
        state_encoder=state_encoder,
        num_agents=2,  # Smaller for faster tests
        learning_rate=agent_config.learning_rate,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_game_state(environment):
    """Create a sample game state for testing."""
    state = environment.reset()
    return state


@pytest.fixture
def sample_encoded_state(state_encoder, sample_game_state):
    """Create a sample encoded state for testing."""
    return state_encoder.encode_state(sample_game_state)


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""

    class MockEnvironment:
        def __init__(self):
            self.num_players = 4
            self.max_cards = 7
            self.current_player = 0
            self.phase = "bidding"
            self.round_cards = 7
            self.trump = "No Trump"
            self.hands = [["A of Spades", "K of Hearts"] for _ in range(4)]
            self.declarations = [None] * 4
            self.tricks_won = [0] * 4
            self.current_trick = []

        def reset(self):
            return self._get_state(0)

        def step(self, player_idx, action):
            if self.phase == "bidding":
                self.declarations[player_idx] = action
                if all(d is not None for d in self.declarations):
                    self.phase = "playing"
                return self._get_state((player_idx + 1) % 4), 0, False
            else:
                return self._get_state((player_idx + 1) % 4), 1.0, True

        def get_legal_actions(self, player_idx):
            if self.phase == "bidding":
                return list(range(self.round_cards + 1))
            else:
                return list(range(len(self.hands[player_idx])))

        def _get_state(self, player_idx):
            return {
                "hand": self.hands[player_idx].copy(),
                "trump": self.trump,
                "declarations": self.declarations.copy(),
                "current_trick": self.current_trick.copy(),
                "tricks_won": self.tricks_won.copy(),
                "round_cards": self.round_cards,
                "phase": self.phase,
                "current_player": self.current_player,
            }

    return MockEnvironment()


@pytest.fixture
def sample_training_data(ppo_agent, mock_environment):
    """Create sample training data for testing."""
    # Collect some experience
    ppo_agent.collect_experience(mock_environment, num_episodes=1, epsilon=0.1)
    return ppo_agent.memory


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")


# Helper functions for tests
def assert_tensor_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are equal within tolerance."""
    if isinstance(tensor1, torch.Tensor):
        tensor1 = tensor1.detach().numpy()
    if isinstance(tensor2, torch.Tensor):
        tensor2 = tensor2.detach().numpy()

    np.testing.assert_allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def assert_dict_structure(dict1, dict2):
    """Assert that two dictionaries have the same structure."""
    assert set(dict1.keys()) == set(dict2.keys())
    for key in dict1:
        if isinstance(dict1[key], dict):
            assert_dict_structure(dict1[key], dict2[key])
        elif isinstance(dict1[key], list):
            assert len(dict1[key]) == len(dict2[key])


def create_test_model_path(temp_dir, model_name="test_model.pth"):
    """Create a test model file path."""
    return temp_dir / model_name
