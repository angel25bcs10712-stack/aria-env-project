"""
ARIA - Autonomous Research & Iteration Agent
Curriculum Learning
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import List, Dict
from training.config import (
    ARIAConfig,
    stage1_config,
    stage2_config,
    stage3_config,
)


class CurriculumManager:
    """
    Manages 3-stage curriculum learning for ARIA.

    Stage 1 — Static World (Capped)
    Stage 2 — Dynamic World (Uncapped)
    Stage 3 — Full Enterprise (Uncapped)
    """

    def __init__(self):
        self.current_stage: int = 1
        self.total_stages: int = 3
        self.episode_count: int = 0
        self.stage_rewards: Dict[int, List[float]] = {1: [], 2: [], 3: []}
        self.stage_configs: Dict[int, ARIAConfig] = {
            1: stage1_config, 2: stage2_config, 3: stage3_config,
        }
        self.stage_names: Dict[int, str] = {
            1: "Stage 1 — Static World (Capped)",
            2: "Stage 2 — Dynamic World (Uncapped)",
            3: "Stage 3 — Full Enterprise (Uncapped)",
        }

    def get_current_config(self) -> ARIAConfig:
        return self.stage_configs[self.current_stage]

    def get_current_stage(self) -> int:
        return self.current_stage

    def log_reward(self, reward: float):
        self.stage_rewards[self.current_stage].append(reward)
        self.episode_count += 1

    def get_average_reward(self) -> float:
        rewards = self.stage_rewards[self.current_stage]
        if not rewards:
            return 0.0
        return round(sum(rewards) / len(rewards), 4)

    def get_recent_average_reward(self, window: int = 20) -> float:
        """Rolling window average — avoids early dilution."""
        rewards = self.stage_rewards[self.current_stage]
        if not rewards:
            return 0.0
        recent = rewards[-window:]
        return round(sum(recent) / len(recent), 4)

    def should_advance(self) -> bool:
        config = self.get_current_config()
        rewards = self.stage_rewards[self.current_stage]
        if len(rewards) < 10:
            return False
        window = getattr(config, 'advancement_window', 20)
        recent_avg = self.get_recent_average_reward(window=window)
        if recent_avg >= config.stage_advance_threshold:
            return True
        if len(rewards) >= config.max_episodes_per_stage:
            return True
        return False

    def advance_stage(self) -> bool:
        if self.current_stage >= self.total_stages:
            print("✅ Already at final stage!")
            return False
        print(f"\n{'='*50}")
        print(f"🚀 ADVANCING TO STAGE {self.current_stage + 1}")
        print(f"   From: {self.stage_names[self.current_stage]}")
        self.current_stage += 1
        print(f"   To:   {self.stage_names[self.current_stage]}")
        print(f"{'='*50}\n")
        return True

    def is_training_complete(self) -> bool:
        if self.current_stage < self.total_stages:
            return False
        rewards = self.stage_rewards[self.total_stages]
        config = self.stage_configs[self.total_stages]
        return len(rewards) >= config.episodes_per_stage

    def print_progress(self):
        print(f"\n{'='*50}")
        print(f"ARIA Curriculum Progress")
        print(f"{'='*50}")
        print(f"Current Stage : {self.stage_names[self.current_stage]}")
        print(f"Episode Count : {self.episode_count}")
        print(f"Avg Reward    : {self.get_average_reward():.4f}")
        print(f"Recent Avg    : {self.get_recent_average_reward():.4f}")
        for stage, rewards in self.stage_rewards.items():
            if rewards:
                avg = round(sum(rewards) / len(rewards), 4)
                print(f"Stage {stage} Avg  : {avg:.4f} ({len(rewards)} episodes)")
        print(f"{'='*50}\n")

    def get_full_summary(self) -> Dict:
        return {
            "current_stage": self.current_stage,
            "total_episodes": self.episode_count,
            "stage_rewards": {
                stage: {
                    "episodes": len(rewards),
                    "average": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
                    "best": max(rewards) if rewards else 0.0,
                }
                for stage, rewards in self.stage_rewards.items()
            },
            "training_complete": self.is_training_complete(),
        }