"""
ARIA - Autonomous Research & Iteration Agent
State Management
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ARIAState:
    # Current step in episode
    step: int = 0
    
    # Maximum steps per episode
    max_steps: int = 20
    
    # Total tool calls made by agent
    tool_calls: int = 0
    
    # List of completed task IDs
    tasks_completed: List[str] = field(default_factory=list)
    
    # Whether policy has changed this episode
    policy_changed: bool = False
    
    # Whether agent detected and adapted to policy change
    adaptation_triggered: bool = False
    
    # Whether episode is finished
    done: bool = False
    
    # History of all actions taken
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # History of all observations
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current reward accumulated
    cumulative_reward: float = 0.0

    def reset(self):
        """Reset state to initial values"""
        self.step = 0
        self.tool_calls = 0
        self.tasks_completed = []
        self.policy_changed = False
        self.adaptation_triggered = False
        self.done = False
        self.action_history = []
        self.observation_history = []
        self.cumulative_reward = 0.0

    def increment_step(self):
        """Move to next step"""
        self.step += 1
        self.tool_calls += 1

    def add_completed_task(self, task_id: str):
        """Mark a task as completed"""
        if task_id not in self.tasks_completed:
            self.tasks_completed.append(task_id)

    def trigger_adaptation(self):
        """Agent detected and adapted to policy change"""
        self.adaptation_triggered = True

    def trigger_policy_change(self):
        """Policy has changed mid episode"""
        self.policy_changed = True

    def mark_done(self):
        """Mark episode as complete"""
        self.done = True

    def summary(self) -> Dict[str, Any]:
        """Return summary of current state"""
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "tool_calls": self.tool_calls,
            "tasks_completed": len(self.tasks_completed),
            "policy_changed": self.policy_changed,
            "adaptation_triggered": self.adaptation_triggered,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
        }