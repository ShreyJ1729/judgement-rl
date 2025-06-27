"""
Command-line interface for monitoring Judgement RL training.

This module provides a CLI for monitoring training progress in real-time,
including live plots and metrics tracking.
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from judgement_rl.config import MonitoringConfig, DEFAULT_MONITORING_CONFIG
from judgement_rl.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor Judgement RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Monitoring settings
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_MONITORING_CONFIG.log_dir,
        help="Directory containing log files to monitor",
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Specific experiment to monitor"
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=DEFAULT_MONITORING_CONFIG.update_interval,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MONITORING_CONFIG.max_points,
        help="Maximum number of points to display",
    )

    # Metrics to track
    parser.add_argument(
        "--track-rewards",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.track_rewards,
        help="Track episode rewards",
    )
    parser.add_argument(
        "--track-losses",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.track_losses,
        help="Track training losses",
    )
    parser.add_argument(
        "--track-actions",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.track_actions,
        help="Track action distributions",
    )
    parser.add_argument(
        "--track-epsilon",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.track_epsilon,
        help="Track exploration epsilon",
    )

    # Visualization
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=DEFAULT_MONITORING_CONFIG.plot_interval,
        help="Plot update interval",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.save_plots,
        help="Save plots to file",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=DEFAULT_MONITORING_CONFIG.plots_dir,
        help="Directory to save plots",
    )

    # External tools
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.use_tensorboard,
        help="Enable TensorBoard monitoring",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.use_wandb,
        help="Enable Weights & Biases monitoring",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_MONITORING_CONFIG.wandb_project,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_MONITORING_CONFIG.wandb_entity,
        help="Weights & Biases entity",
    )

    # Output options
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        default=DEFAULT_MONITORING_CONFIG.log_to_file,
        help="Log monitoring data to file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def setup_monitoring(args):
    """Set up monitoring system."""
    try:
        from judgement_rl.monitoring.realtime_monitor import RealtimeMonitor

        config = MonitoringConfig(
            enabled=True,
            update_interval=args.update_interval,
            max_points=args.max_points,
            track_rewards=args.track_rewards,
            track_losses=args.track_losses,
            track_actions=args.track_actions,
            track_epsilon=args.track_epsilon,
            plot_interval=args.plot_interval,
            save_plots=args.save_plots,
            plots_dir=args.plots_dir,
            log_to_file=args.log_to_file,
            log_dir=args.log_dir,
            use_tensorboard=args.use_tensorboard,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )

        monitor = RealtimeMonitor(config)
        return monitor

    except ImportError:
        print("Monitoring module not available.")
        print("Please install monitoring dependencies:")
        print("pip install -e '.[monitoring]'")
        sys.exit(1)


def find_experiments(log_dir: str) -> list:
    """Find available experiments in log directory."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    experiments = []
    for item in log_path.iterdir():
        if item.is_file() and item.suffix == ".log":
            experiments.append(item.stem)

    return experiments


def select_experiment(
    experiments: list, experiment_name: Optional[str] = None
) -> Optional[str]:
    """Select experiment to monitor."""
    if not experiments:
        print("No experiments found in log directory.")
        return None

    if experiment_name:
        if experiment_name in experiments:
            return experiment_name
        else:
            print(f"Experiment '{experiment_name}' not found.")
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
            return None

    if len(experiments) == 1:
        return experiments[0]

    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp}")

    while True:
        try:
            choice = input(f"Select experiment (1-{len(experiments)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please try again.")


def monitor_experiment(monitor, experiment_name: str, args):
    """Monitor a specific experiment."""
    logger = setup_logger(
        name="monitor", level="INFO" if args.verbose else "WARNING", use_colors=True
    )

    logger.info(f"Starting monitoring for experiment: {experiment_name}")
    logger.info(f"Update interval: {args.update_interval}s")
    logger.info(f"Max points: {args.max_points}")

    try:
        # Start monitoring
        monitor.start_monitoring(experiment_name)

        # Keep monitoring until interrupted
        while True:
            time.sleep(args.update_interval)

            # Check if experiment is still active
            if not monitor.is_experiment_active(experiment_name):
                logger.info("Experiment appears to be finished.")
                break

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise
    finally:
        monitor.stop_monitoring()


def main():
    """Main monitoring function."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(
        name="monitor", level="INFO" if args.verbose else "WARNING", use_colors=True
    )

    try:
        # Find available experiments
        experiments = find_experiments(args.log_dir)

        if not experiments:
            logger.error(f"No experiments found in {args.log_dir}")
            logger.info("Make sure you have training logs in the specified directory.")
            sys.exit(1)

        # Select experiment
        experiment_name = select_experiment(experiments, args.experiment_name)
        if not experiment_name:
            sys.exit(1)

        # Set up monitoring
        monitor = setup_monitoring(args)

        # Start monitoring
        monitor_experiment(monitor, experiment_name, args)

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
