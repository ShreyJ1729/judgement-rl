"""
Command-line interface for launching the Judgement RL GUI.

This module provides a CLI for launching the GUI interface to play
against trained AI agents.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from judgement_rl.config import GUIConfig, DEFAULT_GUI_CONFIG
from judgement_rl.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Judgement RL GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # GUI settings
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_GUI_CONFIG.default_ai_model,
        help="Path to the AI model to play against",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=DEFAULT_GUI_CONFIG.ai_difficulty,
        help="AI difficulty level",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=DEFAULT_GUI_CONFIG.window_width,
        help="GUI window width",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=DEFAULT_GUI_CONFIG.window_height,
        help="GUI window height",
    )
    parser.add_argument(
        "--card-width",
        type=int,
        default=DEFAULT_GUI_CONFIG.card_width,
        help="Card display width",
    )
    parser.add_argument(
        "--card-height",
        type=int,
        default=DEFAULT_GUI_CONFIG.card_height,
        help="Card display height",
    )
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=DEFAULT_GUI_CONFIG.animation_speed,
        help="Animation speed (seconds)",
    )

    # Features
    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        default=DEFAULT_GUI_CONFIG.show_probabilities,
        help="Show AI action probabilities",
    )
    parser.add_argument(
        "--show-ai-thinking",
        action="store_true",
        default=DEFAULT_GUI_CONFIG.show_ai_thinking,
        help="Show AI thinking process",
    )
    parser.add_argument(
        "--auto-play",
        action="store_true",
        default=DEFAULT_GUI_CONFIG.auto_play,
        help="Enable auto-play mode",
    )

    # Game settings
    parser.add_argument(
        "--num-players", type=int, default=4, help="Number of players in the game"
    )
    parser.add_argument(
        "--max-cards", type=int, default=7, help="Maximum number of cards per round"
    )

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def launch_gui(args):
    """Launch the GUI interface."""
    try:
        # Import GUI module (this will be implemented separately)
        from judgement_rl.gui.interface import JudgementGUI

        # Create GUI configuration
        config = GUIConfig(
            window_width=args.window_width,
            window_height=args.window_height,
            default_ai_model=args.model_path,
            ai_difficulty=args.difficulty,
            card_width=args.card_width,
            card_height=args.card_height,
            animation_speed=args.animation_speed,
            show_probabilities=args.show_probabilities,
            show_ai_thinking=args.show_ai_thinking,
            auto_play=args.auto_play,
        )

        # Create and launch GUI
        gui = JudgementGUI(config)
        gui.run()

    except ImportError:
        print("GUI module not available. Please install GUI dependencies.")
        print("Run: pip install -e '.[gui]'")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to launch GUI: {e}")
        sys.exit(1)


def main():
    """Main GUI launcher function."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(
        name="gui",
        level="DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING"),
        use_colors=True,
    )

    # Check if model file exists
    if args.model_path and not Path(args.model_path).exists():
        logger.warning(f"Model file not found: {args.model_path}")
        logger.warning("GUI will launch without AI model")

    try:
        logger.info("Launching Judgement RL GUI...")
        logger.info(f"AI model: {args.model_path}")
        logger.info(f"Difficulty: {args.difficulty}")
        logger.info(f"Window size: {args.window_width}x{args.window_height}")

        # Launch GUI
        launch_gui(args)

    except KeyboardInterrupt:
        logger.info("GUI interrupted by user")
    except Exception as e:
        logger.error(f"GUI failed: {e}")
        raise


if __name__ == "__main__":
    main()
