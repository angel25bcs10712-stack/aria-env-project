"""
ARIA - Autonomous Research & Iteration Agent
Reward Model — Multiple Independent Reward Functions
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import Dict, List


class RewardModel:
    """
    ARIA Reward Model with 4 Independent Reward Functions.

    R1 — Task Completion  (0.4 weight)
    R2 — Efficiency       (0.2 weight)
    R3 — Adaptation       (0.2 weight)
    R4 — Anti-Hacking     (0.2 weight)

    Capped Mode:    R ∈ [0, 1]
    Uncapped Mode:  R ∈ [0, ∞)
    """

    def __init__(self, capped: bool = True):
        self.capped = capped

        # Independent reward weights
        self.w1 = 0.4   # Task completion
        self.w2 = 0.2   # Efficiency
        self.w3 = 0.2   # Adaptation
        self.w4 = 0.2   # Anti-hacking

        # Timeout limit
        self.max_tool_calls = 30

        # History
        self.reward_history: List[Dict] = []

    # ─────────────────────────────────────────
    # MAIN COMPUTE
    # ─────────────────────────────────────────

    def compute(
        self,
        tasks_completed: int,
        total_tasks: int,
        tool_calls: int,
        min_tool_calls: int,
        adaptation_triggered: bool,
        policy_changed: bool,
        action_history: list = [],
    ) -> float:

        # R1 — Task Completion
        r1 = self.reward_task_completion(
            tasks_completed, total_tasks
        )

        # R2 — Efficiency
        r2 = self.reward_efficiency(
            tool_calls, min_tool_calls
        )

        # R3 — Adaptation
        r3 = self.reward_adaptation(
            adaptation_triggered, policy_changed
        )

        # R4 — Anti-Hacking
        r4 = self.reward_anti_hacking(
            tool_calls, action_history
        )

        # Combined reward
        reward = (
            self.w1 * r1 +
            self.w2 * r2 +
            self.w3 * r3 +
            self.w4 * r4
        )

        # Uncapped bonus
        if not self.capped:
            depth_bonus = r2 * (tool_calls / max(min_tool_calls, 1))
            reward += self.w2 * depth_bonus

        # Log
        self.reward_history.append({
            "reward": round(reward, 4),
            "r1_task": round(r1, 4),
            "r2_efficiency": round(r2, 4),
            "r3_adaptation": round(r3, 4),
            "r4_anti_hacking": round(r4, 4),
            "capped": self.capped,
        })

        return round(reward, 4)

    # ─────────────────────────────────────────
    # R1 — TASK COMPLETION
    # ─────────────────────────────────────────

    def reward_task_completion(
        self,
        tasks_completed: int,
        total_tasks: int,
    ) -> float:
        """
        Reward based on how many tasks completed.
        Most important signal — 40% weight.
        """
        if total_tasks == 0:
            return 0.0
        return tasks_completed / total_tasks

    # ─────────────────────────────────────────
    # R2 — EFFICIENCY
    # ─────────────────────────────────────────

    def reward_efficiency(
        self,
        tool_calls: int,
        min_tool_calls: int,
    ) -> float:
        """
        Reward based on efficiency.
        Quality gated — redundant calls penalized.
        Agent cannot game reward by making more calls.
        """
        if min_tool_calls == 0:
            return 1.0
        if tool_calls <= min_tool_calls:
            return 1.0
        penalty = (tool_calls - min_tool_calls) / min_tool_calls
        return max(0.0, 1.0 - penalty)

    # ─────────────────────────────────────────
    # R3 — ADAPTATION
    # ─────────────────────────────────────────

    def reward_adaptation(
        self,
        adaptation_triggered: bool,
        policy_changed: bool,
    ) -> float:
        """
        Reward based on policy adaptation.
        If policy changed but agent ignored it — penalized.
        If no policy change — full score.
        """
        if not policy_changed:
            return 1.0
        return 1.0 if adaptation_triggered else 0.0

    # ─────────────────────────────────────────
    # R4 — ANTI-HACKING
    # ─────────────────────────────────────────

    def reward_anti_hacking(
        self,
        tool_calls: int,
        action_history: list,
    ) -> float:
        """
        Penalize suspicious behavior:
        1. Too many tool calls (timeout)
        2. Repeated identical actions (looping)
        3. Only using one tool type (gaming)
        """

        # Check 1 — Timeout
        if tool_calls > self.max_tool_calls:
            return 0.0

        # Check 2 — Repeated identical actions
        if len(action_history) >= 4:
            last_4 = action_history[-4:]
            tools = [a["action"]["tool"] for a in last_4]
            ops = [a["action"]["operation"] for a in last_4]
            if len(set(tools)) == 1 and len(set(ops)) == 1:
                return 0.2  # Heavy penalty for looping

        # Check 3 — Tool diversity
        if len(action_history) >= 10:
            all_tools = [a["action"]["tool"] for a in action_history]
            unique_tools = len(set(all_tools))
            if unique_tools < 2:
                return 0.3  # Penalty for only using one tool

        return 1.0

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def get_last_reward_breakdown(self) -> Dict:
        """Get breakdown of last reward"""
        if not self.reward_history:
            return {}
        return self.reward_history[-1]

    def get_average_reward(self) -> float:
        """Return average reward across all episodes"""
        if not self.reward_history:
            return 0.0
        return round(
            sum(r["reward"] for r in self.reward_history) /
            len(self.reward_history), 4
        )

    def summary(self) -> Dict:
        """Return reward model summary"""
        return {
            "mode": "capped" if self.capped else "uncapped",
            "weights": {
                "R1 task_completion": self.w1,
                "R2 efficiency": self.w2,
                "R3 adaptation": self.w3,
                "R4 anti_hacking": self.w4,
            },
            "total_episodes": len(self.reward_history),
            "average_reward": self.get_average_reward(),
        }