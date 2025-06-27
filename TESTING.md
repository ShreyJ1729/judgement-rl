# Testing Guide

This guide provides comprehensive information about testing in the Judgement RL project.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Categories](#test-categories)
- [Test Fixtures](#test-fixtures)
- [Best Practices](#best-practices)
- [Debugging Tests](#debugging-tests)
- [Continuous Integration](#continuous-integration)

## Testing Overview

The Judgement RL project uses **pytest** as the primary testing framework. The testing strategy focuses on:

- **Unit tests**: Testing individual components in isolation
- **Integration tests**: Testing component interactions
- **Performance tests**: Testing system performance characteristics
- **Regression tests**: Ensuring existing functionality continues to work

### Key Testing Principles

1. **Test-driven development**: Write tests before implementing features
2. **Comprehensive coverage**: Aim for high test coverage across all components
3. **Fast execution**: Tests should run quickly to enable rapid development
4. **Reliable results**: Tests should be deterministic and not flaky
5. **Clear documentation**: Tests should serve as documentation for expected behavior

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_environment.py      # Environment tests
├── test_agents.py          # Agent tests
├── test_utils.py           # Utility tests
├── test_integration.py     # Integration tests
├── test_config.py          # Configuration tests
├── test_cli.py             # CLI tests
└── data/                   # Test data files
    ├── sample_games/
    └── test_models/
```

### Test File Naming Convention

- Test files should be named `test_<module_name>.py`
- Test functions should be named `test_<function_name>_<scenario>`
- Test classes should be named `Test<ClassName>`

### Test Organization

Each test file should be organized as follows:

```python
"""
Tests for <module_name> module.

This module contains tests for the <module_name> functionality.
"""

import pytest
from judgement_rl import <module_components>

# Test fixtures (if module-specific)

# Unit tests

# Integration tests

# Performance tests (if applicable)
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=judgement_rl

# Run tests with coverage and generate HTML report
pytest --cov=judgement_rl --cov-report=html
```

### Running Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_environment.py

# Run tests matching a pattern
pytest -k "test_reset"

# Run tests with specific markers
pytest -m "unit"
pytest -m "integration"
pytest -m "not slow"

# Run tests in a specific class
pytest tests/test_agents.py::TestPPOAgent
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only GUI tests
pytest -m gui

# Exclude slow tests
pytest -m "not slow"
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run tests on specific number of workers
pytest -n 4
```

### Debugging Tests

```bash
# Run tests with print statements visible
pytest -s

# Run tests and stop on first failure
pytest -x

# Run tests and show local variables on failure
pytest -l

# Run tests with maximum verbosity
pytest -vvv
```

## Writing Tests

### Basic Test Structure

```python
def test_function_name():
    """Test description of what is being tested."""
    # Arrange: Set up test data and conditions
    input_data = [1, 2, 3]
    expected_result = 6

    # Act: Execute the function being tested
    result = sum(input_data)

    # Assert: Verify the result
    assert result == expected_result
```

### Testing with Parameters

```python
import pytest

@pytest.mark.parametrize("input_data,expected", [
    ([1, 2, 3], 6),
    ([0, 0, 0], 0),
    ([1], 1),
    ([], 0),
])
def test_sum_function(input_data, expected):
    """Test sum function with various inputs."""
    result = sum(input_data)
    assert result == expected
```

### Testing Exceptions

```python
import pytest

def test_division_by_zero():
    """Test that division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

def test_invalid_input_type():
    """Test that invalid input types raise TypeError."""
    with pytest.raises(TypeError):
        process_data("not_a_number")
```

### Testing with Mocking

```python
from unittest.mock import Mock, patch

def test_function_with_external_dependency():
    """Test function that depends on external service."""
    # Mock the external dependency
    with patch('module.external_service') as mock_service:
        mock_service.return_value = "mocked_result"

        result = function_under_test()

        assert result == "expected_result"
        mock_service.assert_called_once()
```

## Test Categories

### Unit Tests

Unit tests verify that individual components work correctly in isolation.

```python
def test_agent_action_selection():
    """Test that agent selects valid actions."""
    agent = PPOAgent(state_dim=10, action_dim=4)
    state = torch.randn(1, 10)
    legal_actions = [0, 1, 2]

    action, prob, value = agent.select_action(state, legal_actions)

    assert action in legal_actions
    assert 0 <= prob <= 1
    assert isinstance(value, float)
```

### Integration Tests

Integration tests verify that components work together correctly.

```python
def test_agent_environment_interaction():
    """Test that agent can interact with environment."""
    env = JudgementEnv(num_players=4, max_cards=7)
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=7)

    obs, info = env.reset()
    done = False

    while not done:
        legal_actions = env.get_legal_actions()
        action, _, _ = agent.select_action(obs, legal_actions)
        obs, reward, done, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(done, bool)
```

### Performance Tests

Performance tests verify that components meet performance requirements.

```python
import time

@pytest.mark.performance
def test_agent_inference_speed():
    """Test that agent inference is fast enough."""
    agent = PPOAgent(state_dim=100, action_dim=10)
    state = torch.randn(1, 100)
    legal_actions = list(range(10))

    start_time = time.time()
    for _ in range(1000):
        agent.select_action(state, legal_actions)
    end_time = time.time()

    avg_time = (end_time - start_time) / 1000
    assert avg_time < 0.001  # Should be faster than 1ms per inference
```

### Regression Tests

Regression tests ensure that existing functionality continues to work.

```python
def test_known_game_scenario():
    """Test a known game scenario produces expected results."""
    env = JudgementEnv(num_players=4, max_cards=7)

    # Set up a specific game state
    env.reset(seed=42)

    # Play a sequence of known actions
    actions = [3, 2, 1, 0, 4, 5, 6]
    for action in actions:
        if not env.game_state.is_terminal():
            obs, reward, done, truncated, info = env.step(action)

    # Verify expected final state
    assert env.game_state.is_terminal()
    assert env.game_state.scores[0] == expected_score
```

## Test Fixtures

### Using Fixtures

Fixtures provide reusable test setup and teardown.

```python
import pytest

@pytest.fixture
def sample_game_state():
    """Create a sample game state for testing."""
    env = JudgementEnv(num_players=4, max_cards=7)
    obs, info = env.reset(seed=42)
    return env.game_state

@pytest.fixture
def trained_agent():
    """Create a pre-trained agent for testing."""
    agent = PPOAgent(state_dim=100, action_dim=7)
    # Load pre-trained weights or train briefly
    return agent

def test_agent_with_sample_state(sample_game_state, trained_agent):
    """Test agent with sample game state."""
    # Test implementation
    pass
```

### Fixture Scope

```python
@pytest.fixture(scope="function")  # Default: created for each test
def function_scope_fixture():
    return "function_scope"

@pytest.fixture(scope="class")     # Created once per test class
def class_scope_fixture():
    return "class_scope"

@pytest.fixture(scope="module")    # Created once per test module
def module_scope_fixture():
    return "module_scope"

@pytest.fixture(scope="session")   # Created once per test session
def session_scope_fixture():
    return "session_scope"
```

### Fixture Dependencies

```python
@pytest.fixture
def base_config():
    """Base configuration for tests."""
    return EnvironmentConfig(num_players=4, max_cards=7)

@pytest.fixture
def env_with_config(base_config):
    """Environment with base configuration."""
    return JudgementEnv(config=base_config)

@pytest.fixture
def agent_for_env(env_with_config):
    """Agent configured for the environment."""
    state_dim = env_with_config.observation_space.shape[0]
    action_dim = env_with_config.action_space.n
    return PPOAgent(state_dim=state_dim, action_dim=action_dim)
```

## Best Practices

### Test Design

1. **Single Responsibility**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Tests should run quickly to enable rapid feedback

### Test Data

```python
# Use constants for test data
EXPECTED_WIN_RATE = 0.75
MAX_INFERENCE_TIME = 0.001
SAMPLE_GAME_SEED = 42

# Use factories for complex objects
class GameStateFactory:
    @staticmethod
    def create_bidding_phase():
        """Create a game state in bidding phase."""
        # Implementation

    @staticmethod
    def create_playing_phase():
        """Create a game state in playing phase."""
        # Implementation
```

### Error Handling

```python
def test_robust_error_handling():
    """Test that errors are handled gracefully."""
    # Test with invalid inputs
    with pytest.raises(ValueError):
        process_invalid_input()

    # Test with edge cases
    result = process_edge_case()
    assert result is not None

    # Test with missing data
    result = process_missing_data()
    assert result == default_value
```

### Performance Considerations

```python
@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of large datasets."""
    large_dataset = generate_large_dataset()

    start_time = time.time()
    result = process_dataset(large_dataset)
    end_time = time.time()

    assert end_time - start_time < 5.0  # Should complete within 5 seconds
    assert result is not None
```

## Debugging Tests

### Common Issues

1. **Flaky Tests**: Tests that sometimes pass and sometimes fail

   - Use fixed random seeds
   - Avoid timing-dependent assertions
   - Mock external dependencies

2. **Slow Tests**: Tests that take too long to run

   - Use smaller test datasets
   - Mock expensive operations
   - Mark as slow tests

3. **Test Dependencies**: Tests that depend on each other
   - Ensure tests are independent
   - Use proper setup and teardown
   - Avoid shared state

### Debugging Tools

```python
# Use pdb for debugging
import pdb

def test_with_debugging():
    result = complex_function()
    pdb.set_trace()  # Breakpoint
    assert result == expected_value

# Use pytest.set_trace() for pytest-aware debugging
def test_with_pytest_debugging():
    result = complex_function()
    pytest.set_trace()  # Pytest-aware breakpoint
    assert result == expected_value
```

### Test Output

```python
# Use capsys to capture output
def test_function_output(capsys):
    print_function()
    captured = capsys.readouterr()
    assert "expected output" in captured.out

# Use caplog to capture logging
def test_logging_output(caplog):
    with caplog.at_level(logging.INFO):
        logging_function()
    assert "expected log message" in caplog.text
```

## Continuous Integration

### GitHub Actions

The project includes GitHub Actions workflows for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=judgement_rl --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

Pre-commit hooks ensure code quality before commits:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

### Coverage Reporting

```bash
# Generate coverage report
pytest --cov=judgement_rl --cov-report=html --cov-report=term

# Coverage configuration in pyproject.toml
[tool.coverage.run]
source = ["src/judgement_rl"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

## Test Examples

### Environment Tests

```python
class TestJudgementEnv:
    """Test the Judgement environment."""

    def test_environment_initialization(self):
        """Test environment can be initialized."""
        env = JudgementEnv(num_players=4, max_cards=7)
        assert env.num_players == 4
        assert env.max_cards == 7

    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = JudgementEnv(num_players=4, max_cards=7)
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert info is not None
        assert env.game_state.current_player == 0

    def test_environment_step(self):
        """Test environment step functionality."""
        env = JudgementEnv(num_players=4, max_cards=7)
        obs, info = env.reset()

        legal_actions = env.get_legal_actions()
        action = legal_actions[0]

        obs, reward, done, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert info is not None

    def test_illegal_action_handling(self):
        """Test that illegal actions are handled properly."""
        env = JudgementEnv(num_players=4, max_cards=7)
        obs, info = env.reset()

        legal_actions = env.get_legal_actions()
        illegal_action = max(legal_actions) + 1

        with pytest.raises(ValueError):
            env.step(illegal_action)
```

### Agent Tests

```python
class TestPPOAgent:
    """Test the PPO agent."""

    @pytest.fixture
    def agent(self):
        """Create a PPO agent for testing."""
        return PPOAgent(
            state_dim=100,
            action_dim=7,
            hidden_dim=64,
            learning_rate=1e-4
        )

    def test_agent_initialization(self, agent):
        """Test agent can be initialized."""
        assert agent.state_dim == 100
        assert agent.action_dim == 7
        assert agent.hidden_dim == 64

    def test_action_selection(self, agent):
        """Test agent action selection."""
        state = torch.randn(1, 100)
        legal_actions = [0, 1, 2, 3]

        action, prob, value = agent.select_action(state, legal_actions)

        assert action in legal_actions
        assert 0 <= prob <= 1
        assert isinstance(value, float)

    def test_model_saving_loading(self, agent, tmp_path):
        """Test model saving and loading."""
        model_path = tmp_path / "test_model.pth"

        # Save model
        agent.save_model(str(model_path))
        assert model_path.exists()

        # Load model
        new_agent = PPOAgent(state_dim=100, action_dim=7)
        new_agent.load_model(str(model_path))

        # Verify models are equivalent
        state = torch.randn(1, 100)
        legal_actions = [0, 1, 2, 3]

        action1, prob1, value1 = agent.select_action(state, legal_actions)
        action2, prob2, value2 = new_agent.select_action(state, legal_actions)

        assert action1 == action2
        assert abs(prob1 - prob2) < 1e-6
        assert abs(value1 - value2) < 1e-6
```

### Integration Tests

```python
class TestAgentEnvironmentIntegration:
    """Test integration between agents and environment."""

    def test_agent_environment_compatibility(self):
        """Test that agent and environment work together."""
        env = JudgementEnv(num_players=4, max_cards=7)
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )

        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            legal_actions = env.get_legal_actions()
            action, _, _ = agent.select_action(obs, legal_actions)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        assert isinstance(total_reward, float)
        assert env.game_state.is_terminal()

    def test_training_loop_integration(self):
        """Test complete training loop integration."""
        trainer = SelfPlayTrainer()

        # Train for a few episodes
        stats = trainer.train(num_episodes=10)

        assert 'total_reward' in stats
        assert 'episodes_completed' in stats
        assert stats['episodes_completed'] == 10
```

This testing guide provides comprehensive coverage of testing practices and examples for the Judgement RL project. Follow these guidelines to ensure high-quality, maintainable tests.
