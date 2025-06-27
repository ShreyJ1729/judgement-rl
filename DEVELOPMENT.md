# Development Guide

This guide provides comprehensive information for developers working on the Judgement RL project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Contributing](#contributing)
- [Release Process](#release-process)

## Architecture Overview

### Package Structure

```
src/judgement_rl/
├── agents/              # RL agents and training logic
│   ├── __init__.py
│   ├── agent.py         # PPO agent and self-play trainer
│   └── heuristic_agent.py
├── environment/         # Game environment
│   ├── __init__.py
│   └── judgement_env.py
├── utils/              # Utilities and helpers
│   ├── __init__.py
│   ├── state_encoder.py
│   └── logging.py
├── config/             # Configuration management
│   ├── __init__.py
│   └── config.py
├── cli/                # Command-line interfaces
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gui.py
│   └── monitor.py
├── gui/                # GUI interface (future)
├── monitoring/         # Training monitoring (future)
└── __init__.py
```

### Key Design Principles

1. **Modularity**: Each component is self-contained with clear interfaces
2. **Configuration-driven**: All parameters are configurable via dataclasses
3. **Type safety**: Full type hints throughout the codebase
4. **Testability**: Comprehensive test coverage with pytest
5. **Logging**: Structured logging with multiple output formats
6. **CLI-first**: All functionality available via command-line tools

### Core Components

#### Environment (`judgement_env.py`)

- Implements the Judgement card game rules
- Provides Gym-like interface (reset, step, get_legal_actions)
- Handles game state management and reward calculation

#### State Encoder (`state_encoder.py`)

- Converts game state to numerical representation
- Handles both bidding and playing phases
- Provides consistent state dimensions

#### PPO Agent (`agent.py`)

- Implements Proximal Policy Optimization
- Includes experience collection and training loops
- Supports self-play training

#### Configuration (`config.py`)

- Dataclass-based configuration system
- Validation and default values
- Support for YAML configuration files

#### Logging (`logging.py`)

- Structured logging with multiple formatters
- Support for TensorBoard and Weights & Biases
- Training-specific logging utilities

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- git

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd judgement-rl
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Dependencies

The project uses several development tools:

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks for code quality

## Code Organization

### Module Responsibilities

#### `agents/`

- **`agent.py`**: Core PPO implementation, memory management, training loops
- **`heuristic_agent.py`**: Rule-based agent for comparison

#### `environment/`

- **`judgement_env.py`**: Game logic, state management, reward calculation

#### `utils/`

- **`state_encoder.py`**: State representation and encoding
- **`logging.py`**: Logging utilities and training loggers

#### `config/`

- **`config.py`**: Configuration dataclasses and utilities

#### `cli/`

- **`train.py`**: Training command-line interface
- **`evaluate.py`**: Evaluation command-line interface
- **`gui.py`**: GUI launcher
- **`monitor.py`**: Training monitoring

### Import Conventions

- Use absolute imports within the package
- Import from public API in `__init__.py` files
- Use relative imports for internal module dependencies

### Configuration Management

All configuration is handled through dataclasses in `config/config.py`:

```python
from judgement_rl.config import TrainingConfig

config = TrainingConfig(
    num_episodes=1000,
    learning_rate=1e-4,
    hidden_dim=256
)
```

## Testing

### Test Structure

```
tests/
├── conftest.py          # Pytest configuration and fixtures
├── test_environment.py  # Environment tests
├── test_agents.py       # Agent tests
├── test_utils.py        # Utility tests
└── test_integration.py  # Integration tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=judgement_rl

# Run specific test categories
pytest tests/test_environment.py
pytest tests/test_agents.py

# Run with markers
pytest -m "not slow"
pytest -m integration
```

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Test system performance characteristics
- **GUI tests**: Test user interface components

### Writing Tests

1. **Use fixtures**: Leverage pytest fixtures for common setup
2. **Test edge cases**: Include boundary conditions and error cases
3. **Mock external dependencies**: Use mocks for external services
4. **Use descriptive names**: Test names should clearly describe what's being tested

Example:

```python
def test_agent_action_selection(ppo_agent, sample_game_state):
    """Test that agent selects valid actions."""
    legal_actions = [0, 1, 2, 3]
    action, prob, value = ppo_agent.select_action(
        sample_game_state, legal_actions, epsilon=0.0
    )

    assert action in legal_actions
    assert 0 <= prob <= 1
    assert isinstance(value, float)
```

## Code Quality

### Code Style

The project uses several tools to maintain code quality:

#### Black (Code Formatting)

```bash
# Format all code
black src/ tests/

# Check formatting
black --check src/ tests/
```

#### Flake8 (Linting)

```bash
# Run linter
flake8 src/ tests/

# Configuration in setup.cfg or pyproject.toml
```

#### MyPy (Type Checking)

```bash
# Run type checker
mypy src/

# Configuration in mypy.ini or pyproject.toml
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically check code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Review Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass
- [ ] Code is formatted with Black
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation is updated
- [ ] New features have tests
- [ ] Breaking changes are documented

## Contributing

### Development Workflow

1. **Create feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**:

   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**:

   ```bash
   pytest
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit changes**:

   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: Explain what and why, not how
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update relevant documentation
5. **Breaking changes**: Clearly mark and explain

## Release Process

### Version Management

The project uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Steps

1. **Update version**:

   - Update `__version__` in `src/judgement_rl/__init__.py`
   - Update version in `setup.py`

2. **Update changelog**:

   - Document all changes since last release
   - Include breaking changes prominently

3. **Run full test suite**:

   ```bash
   pytest --cov=judgement_rl
   ```

4. **Build and test package**:

   ```bash
   python setup.py sdist bdist_wheel
   ```

5. **Create release tag**:

   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```

6. **Publish to PyPI** (if applicable):
   ```bash
   twine upload dist/*
   ```

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Changelog is updated
- [ ] Version numbers are updated
- [ ] Release notes are prepared
- [ ] Package builds successfully
- [ ] Tag is created and pushed

## Troubleshooting

### Common Issues

#### Import Errors

- Ensure you're in the correct virtual environment
- Check that the package is installed in development mode
- Verify import paths are correct

#### Test Failures

- Check that all dependencies are installed
- Ensure test data is available
- Verify environment variables are set

#### Performance Issues

- Profile the code to identify bottlenecks
- Check memory usage and optimize if needed
- Consider using smaller test datasets

### Getting Help

- Check the existing documentation
- Look at existing issues and pull requests
- Create a new issue with detailed information
- Include error messages and reproduction steps

## Future Development

### Planned Features

- **GUI Interface**: Complete GUI implementation
- **Monitoring Dashboard**: Real-time training visualization
- **Distributed Training**: Multi-GPU and multi-machine training
- **Model Serving**: REST API for model inference
- **Hyperparameter Optimization**: Automated hyperparameter tuning

### Architecture Evolution

- **Plugin System**: Allow custom agents and environments
- **Configuration Validation**: Enhanced configuration validation
- **Performance Optimization**: Improved training and inference speed
- **Documentation**: Enhanced API documentation and tutorials

### Contributing to Future Features

When contributing to new features:

1. **Design first**: Plan the architecture before implementation
2. **Follow patterns**: Use existing patterns and conventions
3. **Add tests**: Include comprehensive test coverage
4. **Document**: Update documentation and add examples
5. **Review**: Get feedback from maintainers early
