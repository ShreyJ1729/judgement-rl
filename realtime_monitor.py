#!/usr/bin/env python3
"""
Real-time training monitor for Judgement card game agents.
Provides live updating graphs of various training metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Callable
from collections import deque
import tkinter as tk
from tkinter import ttk
import queue


class RealtimeMonitor:
    """
    Real-time monitoring system for training metrics with live updating graphs.
    """

    def __init__(self, max_points: int = 1000, update_interval: float = 1.0):
        """
        Initialize the real-time monitor.

        Args:
            max_points: Maximum number of points to display on graphs
            update_interval: How often to update the display (seconds)
        """
        self.max_points = max_points
        self.update_interval = update_interval

        # Data storage for different metrics
        self.metrics = {
            "episode_rewards": deque(maxlen=max_points),
            "bidding_accuracy": deque(maxlen=max_points),
            "avg_score_per_round": deque(maxlen=max_points),
            "avg_score_per_game": deque(maxlen=max_points),
            "policy_loss": deque(maxlen=max_points),
            "value_loss": deque(maxlen=max_points),
            "entropy_loss": deque(maxlen=max_points),
            "avg_bid_error": deque(maxlen=max_points),
            "trick_win_rate": deque(maxlen=max_points),
            "declaration_success_rate": deque(maxlen=max_points),
            "exploration_rate": deque(maxlen=max_points),
            "learning_rate": deque(maxlen=max_points),
        }

        # Episode counters
        self.episode_count = 0
        self.training_step_count = 0

        # Agent-specific data
        self.agent_metrics = {}

        # Threading for real-time updates
        self.data_queue = queue.Queue()
        self.running = False
        self.update_thread = None

        # GUI elements
        self.fig = None
        self.axes = None
        self.ani = None
        self.paused = False
        self.status_text = None

        # Callback for external updates
        self.update_callback = None

    def add_agent(self, agent_name: str):
        """Add a new agent to track."""
        self.agent_metrics[agent_name] = {
            "episode_rewards": deque(maxlen=self.max_points),
            "bidding_accuracy": deque(maxlen=self.max_points),
            "avg_score_per_round": deque(maxlen=self.max_points),
            "avg_score_per_game": deque(maxlen=self.max_points),
            "policy_loss": deque(maxlen=self.max_points),
            "value_loss": deque(maxlen=self.max_points),
            "entropy_loss": deque(maxlen=self.max_points),
            "avg_bid_error": deque(maxlen=self.max_points),
            "trick_win_rate": deque(maxlen=self.max_points),
            "declaration_success_rate": deque(maxlen=self.max_points),
            "exploration_rate": deque(maxlen=self.max_points),
            "learning_rate": deque(maxlen=self.max_points),
        }

    def update_metrics(self, agent_name: str, metrics_dict: Dict[str, float]):
        """
        Update metrics for a specific agent.

        Args:
            agent_name: Name of the agent
            metrics_dict: Dictionary of metric values to update
        """
        if agent_name not in self.agent_metrics:
            self.add_agent(agent_name)

        for metric_name, value in metrics_dict.items():
            if metric_name in self.agent_metrics[agent_name]:
                self.agent_metrics[agent_name][metric_name].append(value)

        # Also update global metrics (average across all agents)
        for metric_name, value in metrics_dict.items():
            if metric_name in self.metrics:
                # Calculate average across all agents for this metric
                values = []
                for agent_data in self.agent_metrics.values():
                    if metric_name in agent_data and agent_data[metric_name]:
                        values.append(agent_data[metric_name][-1])

                if values:
                    avg_value = np.mean(values)
                    self.metrics[metric_name].append(avg_value)

        # Force update the plot if monitor is running
        if self.running and self.fig is not None:
            try:
                self.update_plot(0)  # Force immediate update
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception as e:
                # Ignore update errors during training
                pass

    def calculate_training_metrics(self, agent, env, episode_data):
        """
        Calculate comprehensive training metrics from episode data.

        Args:
            agent: The trained agent
            env: The game environment
            episode_data: Data from the completed episode

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}

        # Basic episode metrics
        if "episode_reward" in episode_data:
            metrics["episode_rewards"] = episode_data["episode_reward"]

        # Bidding accuracy
        if "declarations" in episode_data and "tricks_won" in episode_data:
            declarations = episode_data["declarations"]
            tricks_won = episode_data["tricks_won"]
            if declarations and tricks_won:
                # Calculate how often agents met their declarations
                success_count = sum(
                    1 for d, t in zip(declarations, tricks_won) if d == t
                )
                metrics["bidding_accuracy"] = success_count / len(declarations)

                # Average bid error
                bid_errors = [abs(d - t) for d, t in zip(declarations, tricks_won)]
                metrics["avg_bid_error"] = np.mean(bid_errors)

        # Trick win rate
        if "tricks_won" in episode_data and "total_tricks" in episode_data:
            total_tricks = episode_data["total_tricks"]
            if total_tricks > 0:
                metrics["trick_win_rate"] = (
                    sum(episode_data["tricks_won"]) / total_tricks
                )

        # Declaration success rate
        if "declarations" in episode_data and "tricks_won" in episode_data:
            declarations = episode_data["declarations"]
            tricks_won = episode_data["tricks_won"]
            if declarations and tricks_won:
                success_count = sum(
                    1 for d, t in zip(declarations, tricks_won) if d == t
                )
                metrics["declaration_success_rate"] = success_count / len(declarations)

        # Average scores
        if "episode_reward" in episode_data:
            metrics["avg_score_per_game"] = episode_data["episode_reward"]

            # Calculate per-round score if we have round information
            if "round_count" in episode_data and episode_data["round_count"] > 0:
                metrics["avg_score_per_round"] = (
                    episode_data["episode_reward"] / episode_data["round_count"]
                )

        # Learning metrics (if available from agent)
        if hasattr(agent, "training_stats"):
            stats = agent.training_stats
            if "policy_losses" in stats and stats["policy_losses"]:
                metrics["policy_loss"] = stats["policy_losses"][-1]
            if "value_losses" in stats and stats["value_losses"]:
                metrics["value_loss"] = stats["value_losses"][-1]
            if "entropy_losses" in stats and stats["entropy_losses"]:
                metrics["entropy_loss"] = stats["entropy_losses"][-1]

        # Exploration rate (if available)
        if hasattr(agent, "epsilon"):
            metrics["exploration_rate"] = agent.epsilon

        # Learning rate (if available)
        if hasattr(agent, "optimizer") and hasattr(agent.optimizer, "param_groups"):
            metrics["learning_rate"] = agent.optimizer.param_groups[0]["lr"]

        return metrics

    def setup_display(self):
        """Set up the matplotlib display with subplots."""
        plt.ion()  # Turn on interactive mode

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        self.axes = self.axes.flatten()

        # Add status text
        self.status_text = self.fig.text(
            0.02,
            0.98,
            "Status: Initializing...",
            transform=self.fig.transFigure,
            fontsize=10,
            verticalalignment="top",
        )

        # Add control buttons
        self.add_control_buttons()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for status text

    def add_control_buttons(self):
        """Add control buttons to the plot."""
        # Pause/Resume button
        ax_pause = plt.axes([0.02, 0.02, 0.1, 0.04])
        self.pause_button = Button(ax_pause, "Pause")
        self.pause_button.on_clicked(self.toggle_pause)

    def toggle_pause(self, event):
        """Toggle pause/resume of the display."""
        self.paused = not self.paused
        status = "Paused" if self.paused else "Running"
        if self.status_text:
            self.status_text.set_text(f"Status: {status}")
        if self.fig:
            self.fig.canvas.draw_idle()

    def update_plot(self, frame):
        """Update the plot with current data."""
        if self.paused:
            return self.axes

        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Plot configuration
        plot_configs = [
            ("Episode Rewards", "Reward", "episode_rewards"),
            ("Bidding Accuracy", "Accuracy", "bidding_accuracy"),
            ("Average Score per Round", "Score", "avg_score_per_round"),
            ("Policy Loss", "Loss", "policy_loss"),
            ("Value Loss", "Loss", "value_loss"),
            ("Average Bid Error", "Error", "avg_bid_error"),
            ("Trick Win Rate", "Rate", "trick_win_rate"),
            ("Declaration Success Rate", "Success Rate", "declaration_success_rate"),
            ("Exploration Rate", "Rate", "exploration_rate"),
        ]

        # Plot each metric
        for i, (title, ylabel, metric_key) in enumerate(plot_configs):
            if i < len(self.axes):
                ax = self.axes[i]
                ax.set_title(title)
                ax.set_xlabel("Episode")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

                # Plot data for each agent
                colors = plt.cm.Set3(np.linspace(0, 1, len(self.agent_metrics)))
                for j, (agent_name, agent_data) in enumerate(
                    self.agent_metrics.items()
                ):
                    if metric_key in agent_data and agent_data[metric_key]:
                        data = list(agent_data[metric_key])
                        episodes = list(range(len(data)))
                        ax.plot(
                            episodes,
                            data,
                            label=agent_name,
                            color=colors[j],
                            linewidth=2,
                        )

                # Also plot global average
                if metric_key in self.metrics and self.metrics[metric_key]:
                    data = list(self.metrics[metric_key])
                    episodes = list(range(len(data)))
                    ax.plot(
                        episodes,
                        data,
                        label="Global Avg",
                        color="black",
                        linewidth=3,
                        linestyle="--",
                        alpha=0.7,
                    )

                ax.legend(loc="upper right", fontsize=8)
                ax.set_ylim(bottom=0)

        # Update status
        total_episodes = max(
            [
                len(data.get("episode_rewards", []))
                for data in self.agent_metrics.values()
            ],
            default=0,
        )
        if self.status_text:
            self.status_text.set_text(
                f'Status: {"Paused" if self.paused else "Running"} | Episodes: {total_episodes}'
            )

        return self.axes

    def start_monitoring(self):
        """Start the real-time monitoring display."""
        self.setup_display()
        self.running = True

        # Start animation with faster updates
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=500,  # Update every 500ms for more responsive display
            blit=False,
            cache_frame_data=False,
        )

        plt.show(block=False)
        print("Real-time monitoring started. Close the plot window to stop monitoring.")

    def stop_monitoring(self):
        """Stop the real-time monitoring."""
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
        if self.fig:
            plt.close(self.fig)

    def save_metrics(self, filename: str):
        """Save current metrics to a file."""
        import json

        data = {
            "global_metrics": {k: list(v) for k, v in self.metrics.items()},
            "agent_metrics": {
                agent: {k: list(v) for k, v in metrics.items()}
                for agent, metrics in self.agent_metrics.items()
            },
            "episode_count": self.episode_count,
            "training_step_count": self.training_step_count,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to {filename}")


class TrainingCallback:
    """
    Callback class to integrate with training loops.
    """

    def __init__(self, monitor: RealtimeMonitor):
        self.monitor = monitor
        self.episode_data = {}

    def on_episode_end(self, agent_name: str, agent, env, episode_data: Dict):
        """
        Called at the end of each episode.

        Args:
            agent_name: Name of the agent
            agent: The trained agent
            env: The game environment
            episode_data: Data from the completed episode
        """
        # Calculate metrics
        metrics = self.monitor.calculate_training_metrics(agent, env, episode_data)

        # Update monitor
        self.monitor.update_metrics(agent_name, metrics)
        self.monitor.episode_count += 1

    def on_training_step(self, agent_name: str, agent, loss_data: Dict):
        """
        Called after each training step.

        Args:
            agent_name: Name of the agent
            agent: The trained agent
            loss_data: Training loss data
        """
        # Update training-specific metrics
        metrics = {}
        if "policy_loss" in loss_data:
            metrics["policy_loss"] = loss_data["policy_loss"]
        if "value_loss" in loss_data:
            metrics["value_loss"] = loss_data["value_loss"]
        if "entropy_loss" in loss_data:
            metrics["entropy_loss"] = loss_data["entropy_loss"]

        self.monitor.update_metrics(agent_name, metrics)
        self.monitor.training_step_count += 1


def create_monitor_and_callback(
    agent_names: List[str], max_points: int = 1000
) -> tuple:
    """
    Create a monitor and callback for training.

    Args:
        agent_names: List of agent names to monitor
        max_points: Maximum number of points to display

    Returns:
        Tuple of (monitor, callback)
    """
    monitor = RealtimeMonitor(max_points=max_points)

    # Add agents to monitor
    for agent_name in agent_names:
        monitor.add_agent(agent_name)

    callback = TrainingCallback(monitor)

    return monitor, callback


if __name__ == "__main__":
    # Example usage
    monitor, callback = create_monitor_and_callback(["Agent1", "Agent2"])
    monitor.start_monitoring()

    # Simulate some training data
    import time

    for i in range(100):
        # Simulate episode data
        episode_data = {
            "episode_reward": np.random.normal(10, 5),
            "declarations": [2, 3, 1, 2],
            "tricks_won": [2, 3, 1, 2],
            "total_tricks": 4,
            "round_count": 1,
        }

        callback.on_episode_end("Agent1", None, None, episode_data)

        # Simulate training step
        loss_data = {
            "policy_loss": np.random.exponential(0.1),
            "value_loss": np.random.exponential(0.05),
            "entropy_loss": np.random.exponential(0.01),
        }
        callback.on_training_step("Agent1", None, loss_data)

        time.sleep(0.1)

    monitor.save_metrics("training_metrics.json")
    monitor.stop_monitoring()
