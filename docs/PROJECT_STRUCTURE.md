# Judgement RL Project Structure

This document describes the organized structure of the Judgement RL project.

## Directory Structure

```
judgement-rl/
├── src/                          # Source code
│   └── judgement_rl/             # Main package
│       ├── __init__.py           # Package initialization
│       ├── agents/               # Agent implementations
│       │   ├── __init__.py
│       │   ├── agent.py          # PPO agent and self-play trainer
│       │   └── heuristic_agent.py # Heuristic agent
│       ├── environment/          # Game environment
│       │   ├── __init__.py
│       │   └── judgement_env.py  # Judgement card game environment
│       └── utils/                # Utility modules
│           ├── __init__.py
│           └── state_encoder.py  # State encoding utilities
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # General evaluation script
│   ├── evaluate_best_model.py    # Best model evaluation script
│   ├── realtime_monitor.py       # Real-time monitoring
│   ├── gui_interface.py          # GUI interface
│   ├── demo_*.py                 # Demo scripts
│   └── play_against_ai.py        # Play against AI script
├── tests/                        # Test files
│   ├── test_*.py                 # Unit tests
│   └── test_implementation.py    # Implementation tests
├── models/                       # Trained models
│   ├── *.pth                     # PyTorch model files
│   ├── *.json                    # Training metrics
│   └── *.png                     # Training plots
├── results/                      # Evaluation results
│   ├── evaluation_results_*.json # Detailed evaluation results
│   └── evaluation_summary_*.txt  # Summary reports
├── configs/                      # Configuration files
│   └── config.py                 # Training and environment configs
├── docs/                         # Documentation
│   ├── README.md                 # Main README
│   ├── GUI_README.md             # GUI documentation
│   ├── PROJECT_STRUCTURE.md      # This file
│   └── STEP6_SUMMARY.md          # Implementation summary
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── .gitignore                    # Git ignore file
```

## Package Organization

### Core Package (`src/judgement_rl/`)

The main package is organized into logical modules:

- **`agents/`**: Contains all agent implementations

  - `agent.py`: PPO agent and self-play trainer
  - `heuristic_agent.py`: Simple heuristic-based agent

- **`environment/`**: Game environment implementation

  - `judgement_env.py`: Complete Judgement card game environment

- **`utils/`**: Utility modules
  - `state_encoder.py`: State encoding and action space utilities

### Scripts (`scripts/`)

Executable scripts for different tasks:

- **Training**: `train.py` - Main training script with self-play
- **Evaluation**:
  - `evaluate.py` - General evaluation script
  - `evaluate_best_model.py` - Evaluates best model against random players
- **Monitoring**: `realtime_monitor.py` - Real-time training monitoring
- **GUI**: `gui_interface.py` - Interactive GUI for playing against AI
- **Demos**: Various demo scripts for showcasing functionality

### Configuration (`configs/`)

Configuration files for training and environment parameters.

### Models and Results (`models/`, `results/`)

- **`models/`**: Stores trained model files and training metrics
- **`results/`**: Stores evaluation results and summaries

## Usage

### Installation

```bash
# Install the package in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### Running Scripts

```bash
# Training
python scripts/train.py

# Evaluation
python scripts/evaluate_best_model.py

# GUI Interface
python scripts/gui_interface.py
```

### Importing in Code

```python
from judgement_rl import PPOAgent, JudgementEnv, StateEncoder
from judgement_rl.agents import HeuristicAgent
```

## Benefits of This Structure

1. **Modularity**: Clear separation of concerns with logical grouping
2. **Maintainability**: Easy to find and modify specific components
3. **Reusability**: Core components can be easily imported and reused
4. **Testability**: Dedicated test directory with clear test organization
5. **Deployability**: Proper package structure for distribution
6. **Documentation**: Organized documentation in dedicated directory

## Development Workflow

1. **Core Development**: Work in `src/judgement_rl/`
2. **Scripts**: Add new scripts to `scripts/`
3. **Tests**: Add tests to `tests/`
4. **Configuration**: Modify `configs/config.py` for new parameters
5. **Documentation**: Update relevant files in `docs/`

## Best Practices

- Keep imports relative within the package
- Use absolute imports when importing from outside the package
- Maintain consistent naming conventions
- Add proper docstrings and type hints
- Update documentation when adding new features
