"""
ARIA - Autonomous Research & Iteration Agent
Policy Engine
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import Dict, List


class PolicyEngine:
    """
    Simulates an enterprise policy engine.
    Policies change mid-session to test agent adaptation.
    This is the core innovation of ARIA — policy drift.
    """

    def __init__(self):
        self.version: int = 1
        self.change_history: List[Dict] = []

        # Initial policy — agent starts with these rules
        self.current_policy: Dict = {
            "expense_limit": 1000,
            "require_receipts": False,
            "meeting_duration_max": 60,
            "approval_required_above": 500,
            "remote_work_allowed": True,
            "version": 1,
        }

    def get_policy(self) -> Dict:
        """Get current active policy"""
        return self.current_policy

    def update_policy(self) -> Dict:
        """
        Trigger policy drift — rules change mid-session.
        This fires at step 10 of every episode.
        Agent must detect this and adapt behavior.
        """
        # Log old policy
        self.change_history.append({
            "version": self.version,
            "policy": self.current_policy.copy(),
        })

        # Increment version
        self.version += 1

        # New stricter policy
        self.current_policy = {
            "expense_limit": 500,
            "require_receipts": True,
            "meeting_duration_max": 30,
            "approval_required_above": 200,
            "remote_work_allowed": False,
            "version": self.version,
        }

        return self.current_policy

    def get_change_history(self) -> List[Dict]:
        """Return full policy change history"""
        return self.change_history

    def has_changed(self) -> bool:
        """Check if policy has changed at least once"""
        return len(self.change_history) > 0

    def get_diff(self) -> Dict:
        """
        Return difference between old and new policy.
        Useful for agent to understand what changed.
        """
        if not self.change_history:
            return {"message": "Policy has not changed yet"}

        old = self.change_history[-1]["policy"]
        new = self.current_policy
        diff = {}

        for key in new:
            if key != "version" and old.get(key) != new.get(key):
                diff[key] = {
                    "old": old.get(key),
                    "new": new.get(key),
                }

        return {
            "changed_fields": diff,
            "old_version": old["version"],
            "new_version": new["version"],
        }

    def summary(self) -> Dict:
        """Return policy engine summary"""
        return {
            "current_version": self.version,
            "total_changes": len(self.change_history),
            "current_policy": self.current_policy,
        }