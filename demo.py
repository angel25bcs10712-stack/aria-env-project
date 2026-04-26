"""
ARIA - Autonomous Research & Iteration Agent
Full Demo Script
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import time
import random
import math
from environment.aria_env import ARIAEnvironment
from evaluation.metrics import MetricsTracker

# ─────────────────────────────────────────────
# SCRIPTED ACTIONS (simulates a trained agent)
# ─────────────────────────────────────────────

ACTIONS = [
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "revenue", "value": 150000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "expenses", "value": 80000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "net_profit", "value": 70000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "yoy_growth", "value": 12.5}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Thursday 2pm", "event": "Q3 Review"}},
    {"tool": "email", "operation": "send", "params": {"to": "manager@corp.com", "subject": "Q3 Ready", "body": "Done"}},
    {"tool": "calendar", "operation": "reschedule", "params": {"old_slot": "Thursday 3pm", "new_slot": "Friday 2pm"}},
    {"tool": "email", "operation": "send", "params": {"to": "client@external.com", "subject": "Rescheduled", "body": "Friday 2pm"}},
    {"tool": "policy", "operation": "get", "params": {}},
    {"tool": "email", "operation": "send", "params": {"to": "hr@corp.com", "subject": "Confirmed", "body": "Done"}},
    {"tool": "policy", "operation": "get", "params": {}},
    {"tool": "email", "operation": "send", "params": {"to": "team@corp.com", "subject": "Update", "body": "Done"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Monday 3pm", "event": "Sync"}},
    {"tool": "email", "operation": "send", "params": {"to": "finance@corp.com", "subject": "Report", "body": "Done"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Tuesday 2pm", "event": "Review"}},
    {"tool": "email", "operation": "send", "params": {"to": "ceo@corp.com", "subject": "Q3", "body": "Done"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Wednesday 2pm", "event": "Planning"}},
    {"tool": "email", "operation": "send", "params": {"to": "ops@corp.com", "subject": "Ops", "body": "Done"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Wednesday 4pm", "event": "Wrap"}},
    {"tool": "email", "operation": "send", "params": {"to": "all@corp.com", "subject": "Complete", "body": "Done"}},
]


def run_episode(capped, difficulty, verbose=False):
    """Run a single episode with scripted actions."""
    env = ARIAEnvironment(capped=capped, difficulty=difficulty)
    obs = env.reset()
    final_reward = 0.0
    for action in ACTIONS:
        obs, reward, done, info = env.step(action)
        if info.get("policy_changed") and verbose:
            print(f"   ⚠️  POLICY CHANGED at step {obs['step']}!")
        if info.get("adaptation_detected") and verbose:
            print(f"   ✅  Agent adapted at step {obs['step']}!")
        if done:
            final_reward = info.get("final_reward", 0.0)
            break
    return final_reward, obs


def simulate_improvement(base, episode, total, noise=0.04):
    """Simulate a sigmoid learning curve for demo purposes."""
    progress = episode / total
    improvement = 1 / (1 + math.exp(-10 * (progress - 0.5)))
    return max(0.0, base + (0.5 * improvement) + random.uniform(-noise, noise))


def main():
    print("\n" + "="*60)
    print("  ARIA - Autonomous Research & Iteration Agent")
    print("  Judge Demo Script")
    print("  Meta PyTorch OpenEnv Hackathon x Scaler 2026")
    print("  Author: Angel Singh")
    print("="*60)

    # Run a real episode first to show the environment works
    print("\n" + "-"*60)
    print("LIVE EPISODE — Scripted Agent (Stage 3, Uncapped)")
    print("-"*60)
    reward, obs = run_episode(capped=False, difficulty=1, verbose=True)
    print(f"   Episode Reward : {reward:.4f}")
    print(f"   Tasks Done     : {obs['tasks_completed']}/{obs['total_tasks']}")
    print(f"   Adapted        : {obs['adaptation_triggered']}")

    # Simulated training curves (for demo visualization)
    print("\n" + "-"*60)
    print("SIMULATED TRAINING CURVES (for visualization)")
    print("-"*60)

    metrics = MetricsTracker()
    episodes = 20

    # Stage 1
    print("\nSTAGE 1 - Static World (Capped Rewards)")
    for i in range(1, episodes + 1):
        reward = simulate_improvement(0.25, i, episodes)
        metrics.log_episode(
            reward=reward,
            task_completion=min(0.23 + (0.3 * i / episodes), 0.55),
            adaptation_score=0.0,
            stage=1,
        )
        if i % 5 == 0:
            print(f"   Episode {i:3d} | Reward: {reward:.4f}")
        time.sleep(0.02)

    # Stage 2
    print("\nSTAGE 2 - Dynamic World (Uncapped Rewards)")
    for i in range(1, episodes + 1):
        reward = simulate_improvement(0.45, i, episodes)
        metrics.log_episode(
            reward=reward,
            task_completion=min(0.45 + (0.25 * i / episodes), 0.70),
            adaptation_score=min(0.1 + (0.4 * i / episodes), 0.45),
            stage=2,
        )
        if i % 5 == 0:
            print(f"   Episode {i:3d} | Reward: {reward:.4f}")
        time.sleep(0.02)

    # Stage 3
    print("\nSTAGE 3 - Full Enterprise (Uncapped Rewards)")
    for i in range(1, episodes + 1):
        reward = simulate_improvement(0.60, i, episodes)
        metrics.log_episode(
            reward=reward,
            task_completion=min(0.60 + (0.20 * i / episodes), 0.78),
            adaptation_score=min(0.35 + (0.30 * i / episodes), 0.65),
            stage=3,
        )
        if i % 5 == 0:
            print(f"   Episode {i:3d} | Reward: {reward:.4f}")
        time.sleep(0.02)

    # Final Results
    print("\n" + "="*60)
    print("  ARIA - Final Results")
    print("="*60)
    print(f"\n  Reward Score")
    print(f"     Before : {metrics.reward_history[0]:.4f}")
    print(f"     After  : {metrics.reward_history[-1]:.4f}")
    print(f"\n  Task Completion")
    print(f"     Before : {metrics.task_completion_history[0]:.2%}")
    print(f"     After  : {metrics.task_completion_history[-1]:.2%}")
    print(f"\n  Adaptation Score")
    print(f"     Before : {metrics.adaptation_history[0]:.2%}")
    print(f"     After  : {metrics.adaptation_history[-1]:.2%}")

    # Save
    print("\n" + "-"*60)
    metrics.save_graphs()
    metrics.save_metrics()

    print("\n" + "="*60)
    print("  ARIA Demo Complete!")
    print("  Graphs saved to ./results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()