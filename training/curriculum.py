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
        3 tools, no policy changes
        Agent learns basic task completion

    Stage 2 — Dynamic World (Uncapped)
        5 tools, one policy change per episode
        Agent learns adaptation

    Stage 3 — Full Enterprise (Uncapped)
        Full 20-step workflows
        Multiple policy changes
        Agent learns autonomous enterprise behavior
    """

    def __init__(self):
        self.current_stage: int = 1
        self.total_stages: int = 3
        self.episode_count: int = 0
        self.stage_rewards: Dict[int, List[float]] = {
            1: [], 2: [], 3: []
        }

        # Stage configurations
        self.stage_configs: Dict[int, ARIAConfig] = {
            1: stage1_config,
            2: stage2_config,
            3: stage3_config,
        }

        # Stage names for logging
        self.stage_names: Dict[int, str] = {
            1: "Stage 1 — Static World (Capped)",
            2: "Stage 2 — Dynamic World (Uncapped)",
            3: "Stage 3 — Full Enterprise (Uncapped)",
        }

    def get_current_config(self) -> ARIAConfig:
        """Get config for current stage"""
        return self.stage_configs[self.current_stage]

    def get_current_stage(self) -> int:
        """Get current stage number"""
        return self.current_stage

    def log_reward(self, reward: float):
        """Log reward for current stage"""
        self.stage_rewards[self.current_stage].append(reward)
        self.episode_count += 1

    def get_average_reward(self) -> float:
        """Get average reward for current stage"""
        rewards = self.stage_rewards[self.current_stage]
        if not rewards:
            return 0.0
        return round(sum(rewards) / len(rewards), 4)

    def should_advance(self) -> bool:
        """
        Check if agent should advance to next stage.
        Advances when:
        - Average reward exceeds threshold, OR
        - Max episodes reached
        """
        config = self.get_current_config()
        rewards = self.stage_rewards[self.current_stage]

        # Not enough episodes yet
        if len(rewards) < 10:
            return False

        # Check reward threshold
        avg_reward = self.get_average_reward()
        if avg_reward >= config.stage_advance_threshold:
            return True

        # Force advance if max episodes reached
        if len(rewards) >= config.max_episodes_per_stage:
            return True

        return False

    def advance_stage(self) -> bool:
        """
        Advance to next stage.
        Returns False if already at final stage.
        """
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
        """Check if all stages are complete"""
        if self.current_stage < self.total_stages:
            return False
        rewards = self.stage_rewards[self.total_stages]
        config = self.stage_configs[self.total_stages]
        return len(rewards) >= config.episodes_per_stage

    def print_progress(self):
        """Print current training progress"""
        print(f"\n{'='*50}")
        print(f"ARIA Curriculum Progress")
        print(f"{'='*50}")
        print(f"Current Stage : {self.stage_names[self.current_stage]}")
        print(f"Episode Count : {self.episode_count}")
        print(f"Avg Reward    : {self.get_average_reward():.4f}")
        for stage, rewards in self.stage_rewards.items():
            if rewards:
                avg = round(sum(rewards) / len(rewards), 4)
                print(f"Stage {stage} Avg  : {avg:.4f} ({len(rewards)} episodes)")
        print(f"{'='*50}\n")

    def get_full_summary(self) -> Dict:
        """Return full curriculum summary"""
        return {
            "current_stage": self.current_stage,
            "total_episodes": self.episode_count,
            "stage_rewards": {
                stage: {
                    "episodes": len(rewards),
                    "average": round(sum(rewards) / len(rewards), 4)
                    if rewards else 0.0,
                    "best": max(rewards) if rewards else 0.0,
                }
                for stage, rewards in self.stage_rewards.items()
            },
            "training_complete": self.is_training_complete(),
        }