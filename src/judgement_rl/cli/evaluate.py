"""
Command-line interface for evaluating Judgement RL agents.

This module provides a comprehensive CLI for evaluating trained agents
against different opponents and generating detailed performance reports.
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent
from judgement_rl.agents.heuristic_agent import HeuristicAgent
from judgement_rl.config import (
    EvaluationConfig,
    DEFAULT_EVALUATION_CONFIG,
)
from judgement_rl.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Judgement RL agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model to evaluate
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model to evaluate",
    )

    # Evaluation settings
    parser.add_argument(
        "--num-games",
        type=int,
        default=DEFAULT_EVALUATION_CONFIG.num_games,
        help="Number of games to play for evaluation",
    )
    parser.add_argument(
        "--num-players", type=int, default=4, help="Number of players in the game"
    )
    parser.add_argument(
        "--max-cards", type=int, default=7, help="Maximum number of cards per round"
    )

    # Opponent settings
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic", "trained"],
        default="random",
        help="Type of opponent to evaluate against",
    )
    parser.add_argument(
        "--opponent-model-path",
        type=str,
        help="Path to opponent model (if opponent is 'trained')",
    )
    parser.add_argument(
        "--agent-position",
        type=int,
        default=0,
        help="Position of the agent being evaluated (0-3)",
    )

    # Metrics to track
    parser.add_argument(
        "--track-win-rate", action="store_true", default=True, help="Track win rate"
    )
    parser.add_argument(
        "--track-average-score",
        action="store_true",
        default=True,
        help="Track average score",
    )
    parser.add_argument(
        "--track-bid-accuracy",
        action="store_true",
        default=True,
        help="Track bid accuracy",
    )
    parser.add_argument(
        "--track-trick-win-rate",
        action="store_true",
        default=True,
        help="Track trick win rate",
    )

    # Output settings
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_EVALUATION_CONFIG.results_dir,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save evaluation results to file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def load_agent(model_path: str, encoder: StateEncoder) -> PPOAgent:
    """Load a trained agent from file."""
    agent = PPOAgent(state_encoder=encoder)
    agent.load_model(model_path)
    return agent


def create_opponent(
    opponent_type: str, encoder: StateEncoder, model_path: Optional[str] = None
):
    """Create an opponent agent."""
    if opponent_type == "random":
        return RandomAgent()
    elif opponent_type == "heuristic":
        return HeuristicAgent()
    elif opponent_type == "trained":
        if model_path is None:
            raise ValueError("Model path required for trained opponent")
        return load_agent(model_path, encoder)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


class RandomAgent:
    """Simple random agent for evaluation."""

    def select_action(self, state, legal_actions):
        """Select a random action."""
        action = np.random.choice(legal_actions)
        return action, 1.0 / len(legal_actions), 0.0


def evaluate_agent(args, logger):
    """Evaluate an agent against specified opponents."""
    logger.info(f"Loading agent from {args.model_path}")

    # Set up environment and encoder
    env = JudgementEnv(num_players=args.num_players, max_cards=args.max_cards)
    encoder = StateEncoder(num_players=args.num_players, max_cards=args.max_cards)

    # Load agent
    agent = load_agent(args.model_path, encoder)

    # Create opponent
    opponent = create_opponent(args.opponent, encoder, args.opponent_model_path)

    logger.info(f"Evaluating agent against {args.opponent} opponent")
    logger.info(f"Playing {args.num_games} games")
    logger.info(f"Agent position: {args.agent_position}")

    # Evaluation metrics
    results = {
        "games_played": 0,
        "agent_wins": 0,
        "agent_scores": [],
        "opponent_scores": [],
        "bid_accuracies": [],
        "trick_win_rates": [],
        "game_lengths": [],
        "agent_rewards": [],
    }

    # Play games
    for game in range(args.num_games):
        state = env.reset()
        game_rewards = [0] * args.num_players
        game_bids = [None] * args.num_players
        game_tricks_won = [0] * args.num_players
        game_length = 0
        done = False

        # Track bidding accuracy
        agent_bid = None
        agent_tricks_won = 0

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            if legal_actions:
                # Select action based on player
                if current_player == args.agent_position:
                    action, prob, value = agent.select_action(
                        state, legal_actions, epsilon=0.0
                    )

                    # Track bid if in bidding phase
                    if env.phase == "bidding":
                        agent_bid = action
                else:
                    action, prob, value = opponent.select_action(state, legal_actions)

                # Take step
                state, reward, done = env.step(current_player, action)
                game_rewards[current_player] += reward
                game_length += 1

                # Track tricks won
                if (
                    env.phase == "playing"
                    and len(env.current_trick) == 0
                    and game_length > 1
                ):
                    # Trick was just completed
                    if hasattr(env, "tricks_won"):
                        game_tricks_won = env.tricks_won.copy()
            else:
                break

        # Game completed
        results["games_played"] += 1
        results["game_lengths"].append(game_length)

        # Calculate final scores
        agent_score = game_rewards[args.agent_position]
        opponent_score = max(
            game_rewards[i] for i in range(args.num_players) if i != args.agent_position
        )

        results["agent_scores"].append(agent_score)
        results["opponent_scores"].append(opponent_score)
        results["agent_rewards"].append(agent_score)

        # Determine winner
        if agent_score > opponent_score:
            results["agent_wins"] += 1

        # Calculate bid accuracy
        if agent_bid is not None:
            agent_tricks_won = game_tricks_won[args.agent_position]
            bid_accuracy = 1.0 if agent_bid == agent_tricks_won else 0.0
            results["bid_accuracies"].append(bid_accuracy)

        # Calculate trick win rate
        if game_tricks_won[args.agent_position] > 0:
            trick_win_rate = game_tricks_won[args.agent_position] / env.round_cards
            results["trick_win_rates"].append(trick_win_rate)

        # Log progress
        if args.verbose and (game + 1) % 10 == 0:
            logger.info(f"Completed {game + 1}/{args.num_games} games")

    # Calculate final metrics
    final_results = calculate_metrics(results)

    # Log results
    logger.info("Evaluation completed!")
    logger.info(f"Games played: {final_results['games_played']}")
    logger.info(f"Win rate: {final_results['win_rate']:.3f}")
    logger.info(f"Average score: {final_results['average_score']:.2f}")
    logger.info(f"Bid accuracy: {final_results['bid_accuracy']:.3f}")
    logger.info(f"Trick win rate: {final_results['trick_win_rate']:.3f}")

    return final_results


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate final evaluation metrics."""
    final_results = {
        "games_played": results["games_played"],
        "win_rate": (
            results["agent_wins"] / results["games_played"]
            if results["games_played"] > 0
            else 0.0
        ),
        "average_score": (
            np.mean(results["agent_scores"]) if results["agent_scores"] else 0.0
        ),
        "score_std": (
            np.std(results["agent_scores"]) if results["agent_scores"] else 0.0
        ),
        "min_score": min(results["agent_scores"]) if results["agent_scores"] else 0.0,
        "max_score": max(results["agent_scores"]) if results["agent_scores"] else 0.0,
        "average_game_length": (
            np.mean(results["game_lengths"]) if results["game_lengths"] else 0.0
        ),
        "bid_accuracy": (
            np.mean(results["bid_accuracies"]) if results["bid_accuracies"] else 0.0
        ),
        "trick_win_rate": (
            np.mean(results["trick_win_rates"]) if results["trick_win_rates"] else 0.0
        ),
        "agent_scores": results["agent_scores"],
        "opponent_scores": results["opponent_scores"],
        "game_lengths": results["game_lengths"],
        "bid_accuracies": results["bid_accuracies"],
        "trick_win_rates": results["trick_win_rates"],
    }

    return final_results


def save_results(results: Dict[str, Any], args, logger):
    """Save evaluation results to file."""
    if not args.save_results:
        return

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    model_name = Path(args.model_path).stem
    opponent_name = args.opponent
    timestamp = f"{args.num_games}games"
    filename = f"evaluation_{model_name}_vs_{opponent_name}_{timestamp}.json"

    results_path = results_dir / filename

    # Prepare results for saving
    save_data = {
        "evaluation_config": {
            "model_path": args.model_path,
            "opponent": args.opponent,
            "opponent_model_path": args.opponent_model_path,
            "num_games": args.num_games,
            "num_players": args.num_players,
            "max_cards": args.max_cards,
            "agent_position": args.agent_position,
            "seed": args.seed,
        },
        "results": results,
    }

    # Save to file
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(
        name="evaluation", level="INFO" if args.verbose else "WARNING", use_colors=True
    )

    # Set random seed
    import torch
    import numpy as np

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        # Check if model file exists
        if not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            sys.exit(1)

        # Check opponent model if needed
        if args.opponent == "trained" and args.opponent_model_path:
            if not Path(args.opponent_model_path).exists():
                logger.error(
                    f"Opponent model file not found: {args.opponent_model_path}"
                )
                sys.exit(1)

        # Run evaluation
        results = evaluate_agent(args, logger)

        # Save results
        save_results(results, args, logger)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
