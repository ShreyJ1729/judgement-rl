# Judgement RL

A reinforcement learning implementation for the Judgement card game using PPO (Proximal Policy Optimization) with self-play training.

## Overview

Judgement is a trick-taking card game where players bid on the number of tricks they aim to win in each round. This project implements an AI agent that learns optimal strategies for bidding and gameplay using reinforcement learning.

### Key Features

- **PPO Agent**: Proximal Policy Optimization implementation with self-play training
- **Modular Architecture**: Clean separation of environment, agents, and utilities
- **Comprehensive Testing**: Full test suite with pytest
- **GUI Interface**: Play against the trained AI
- **Real-time Monitoring**: Live training progress visualization
- **Configuration Management**: Flexible configuration system

## Usage

All you need to know to run this (but pip install requirements first)

`python3 scripts/play_against_ai.py`

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd judgement-rl
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**:

   ```bash
   pip install -e .
   ```

4. **Install development dependencies** (optional):
   ```bash
   pip install -e ".[dev]"
   ```

## Quick Start

### Basic Usage

```python
from judgement_rl import JudgementEnv, PPOAgent, StateEncoder

# Initialize environment and agent
env = JudgementEnv(num_players=4, max_cards=7)
encoder = StateEncoder(num_players=4, max_cards=7)
agent = PPOAgent(encoder)

# Train the agent
for episode in range(100):
    agent.collect_experience(env, num_episodes=1, epsilon=0.1)
    if (episode + 1) % 10 == 0:
        agent.train(batch_size=64, num_epochs=4)
```

### Self-Play Training

```python
from judgement_rl import SelfPlayTrainer

# Initialize self-play trainer
trainer = SelfPlayTrainer(env, encoder, num_agents=4)

# Train with self-play
trainer.train(num_episodes=1000, episodes_per_update=10)
```

### Play Against AI

```bash
# Launch the GUI interface
python -m judgement_rl.gui
```

## Project Structure

```
judgement-rl/
├── src/judgement_rl/          # Main package
│   ├── agents/                # RL agents
│   ├── environment/           # Game environment
│   ├── utils/                 # Utilities and encoders
│   ├── config/                # Configuration management
│   ├── gui/                   # GUI interface
│   └── monitoring/            # Training monitoring
├── tests/                     # Test suite
├── docs/                      # Documentation
├── configs/                   # Configuration files
├── models/                    # Saved models
└── scripts/                   # Utility scripts
```

## Configuration

The project uses a flexible configuration system. Create custom configurations:

```python
from judgement_rl.config import TrainingConfig

config = TrainingConfig(
    num_episodes=1000,
    learning_rate=1e-4,
    hidden_dim=512
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=judgement_rl

# Run specific test categories
pytest tests/test_environment.py
pytest tests/test_agents.py
pytest tests/test_integration.py
```

## Training

### Basic Training

```bash
# Train a single agent
python -m judgement_rl.train --episodes 1000 --save-interval 100

# Train with self-play
python -m judgement_rl.train --self-play --episodes 2000
```

### Monitoring Training

```bash
# Start real-time monitoring
python -m judgement_rl.monitor
```

## Evaluation

Evaluate trained models:

```bash
# Evaluate against random agents
python -m judgement_rl.evaluate --model-path models/best_agent.pth

# Evaluate against heuristic agents
python -m judgement_rl.evaluate --model-path models/best_agent.pth --opponent heuristic
```

## API Reference

### Environment

```python
class JudgementEnv:
    def __init__(self, num_players: int = 4, max_cards: int = 7)
    def reset() -> Dict[str, Any]
    def step(player_idx: int, action: Any) -> Tuple[Dict[str, Any], float, bool]
    def get_legal_actions(player_idx: int) -> List[int]
```

### Agents

```python
class PPOAgent:
    def __init__(self, state_encoder: StateEncoder, **kwargs)
    def select_action(state: Dict, legal_actions: List[int]) -> Tuple[int, float, float]
    def collect_experience(env, num_episodes: int = 1)
    def train(batch_size: int = 64, num_epochs: int = 4)
    def save_model(filepath: str)
    def load_model(filepath: str)
```

### Self-Play Trainer

```python
class SelfPlayTrainer:
    def __init__(self, env, state_encoder: StateEncoder, num_agents: int = 4)
    def train(num_episodes: int, episodes_per_update: int = 10)
    def save_best_agent(filepath: str)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Development

### Code Style

The project uses:

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Run full test suite
5. Submit PR

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on the Judgement card game
- Uses PyTorch for deep learning
- Inspired by modern RL practices
