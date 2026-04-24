"""
ARIA - Autonomous Research & Iteration Agent
Metrics Tracker
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt


class MetricsTracker:
    """
    Tracks and visualizes ARIA training metrics.
    Generates the 3 graphs shown to judges.
    """

    def __init__(self):
        self.reward_history: List[float] = []
        self.task_completion_history: List[float] = []
        self.adaptation_history: List[float] = []
        self.stage_history: List[int] = []
        self.episode_count: int = 0
        self.stage_summaries: Dict[int, Dict] = {}
        self.output_dir = "./results"
        os.makedirs(self.output_dir, exist_ok=True)

    def log_episode(
        self,
        reward: float,
        task_completion: float,
        adaptation_score: float,
        stage: int,
    ):
        """Log metrics for one episode"""
        self.reward_history.append(reward)
        self.task_completion_history.append(task_completion)
        self.adaptation_history.append(adaptation_score)
        self.stage_history.append(stage)
        self.episode_count += 1

    def log_stage(
        self,
        stage: int,
        avg_reward: float,
        episodes: int,
    ):
        """Log summary for completed stage"""
        self.stage_summaries[stage] = {
            "avg_reward": avg_reward,
            "episodes": episodes,
        }
        print(f"\n📊 Stage {stage} Summary:")
        print(f"   Average Reward : {avg_reward:.4f}")
        print(f"   Episodes       : {episodes}")

    def plot_reward_curve(self):
        """Graph 1 — Reward Curve"""
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.reward_history,
            color="#4A90D9",
            linewidth=2,
            label="Episode Reward"
        )
        if len(self.reward_history) >= 10:
            window = 10
            moving_avg = [
                sum(self.reward_history[max(0, i-window):i]) /
                min(i, window)
                for i in range(1, len(self.reward_history) + 1)
            ]
            plt.plot(
                moving_avg,
                color="#E74C3C",
                linewidth=2,
                linestyle="--",
                label="Moving Average"
            )
        plt.title("ARIA — Reward Curve", fontsize=14, fontweight="bold")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reward_curve.png", dpi=150)
        plt.close()
        print("✅ Saved reward_curve.png")

    def plot_task_completion(self):
        """Graph 2 — Task Completion Rate"""
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.task_completion_history,
            color="#2ECC71",
            linewidth=2,
            label="Task Completion Rate"
        )
        plt.axhline(
            y=0.78,
            color="#E74C3C",
            linestyle="--",
            label="Target (78%)"
        )
        plt.title(
            "ARIA — Task Completion Rate",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Episode")
        plt.ylabel("Completion Rate")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/task_completion.png",
            dpi=150
        )
        plt.close()
        print("✅ Saved task_completion.png")

    def plot_adaptation_score(self):
        """Graph 3 — Adaptation Score"""
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.adaptation_history,
            color="#9B59B6",
            linewidth=2,
            label="Adaptation Score"
        )
        plt.axhline(
            y=0.65,
            color="#E74C3C",
            linestyle="--",
            label="Target (65%)"
        )
        plt.title(
            "ARIA — Adaptation Score",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Episode")
        plt.ylabel("Adaptation Score")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/adaptation_score.png",
            dpi=150
        )
        plt.close()
        print("✅ Saved adaptation_score.png")

    def save_graphs(self):
        """Save all 3 graphs for judge demo"""
        print("\n📊 Generating judge demo graphs...")
        self.plot_reward_curve()
        self.plot_task_completion()
        self.plot_adaptation_score()
        print("✅ All graphs saved to ./results/")

    def save_metrics(self):
        """Save raw metrics to JSON"""
        metrics = {
            "total_episodes": self.episode_count,
            "reward_history": self.reward_history,
            "task_completion_history": self.task_completion_history,
            "adaptation_history": self.adaptation_history,
            "stage_summaries": self.stage_summaries,
        }
        path = f"{self.output_dir}/metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {path}")

    def print_summary(self):
        """Print final training summary"""
        print("\n" + "="*50)
        print("ARIA Training Summary")
        print("="*50)
        print(f"Total Episodes    : {self.episode_count}")
        if self.reward_history:
            print(f"Initial Reward    : {self.reward_history[0]:.4f}")
            print(f"Final Reward      : {self.reward_history[-1]:.4f}")
            print(f"Best Reward       : {max(self.reward_history):.4f}")
        if self.task_completion_history:
            print(f"Final Completion  : {self.task_completion_history[-1]:.2%}")
        if self.adaptation_history:
            print(f"Final Adaptation  : {self.adaptation_history[-1]:.2%}")
        print("="*50)