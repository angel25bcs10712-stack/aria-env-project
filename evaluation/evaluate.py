"""
ARIA - Autonomous Research & Iteration Agent
Evaluation Script
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import torch
from typing import Dict, List
from environment.aria_env import ARIAEnvironment
from evaluation.metrics import MetricsTracker
from training.train import build_prompt, parse_action


class ARIAEvaluator:
    """
    Evaluates ARIA agent performance.
    Used to generate judge demo results.
    """

    def __init__(self, difficulty: int = 3):
        self.difficulty = difficulty
        self.metrics = MetricsTracker()
        self.results: List[Dict] = []

    def evaluate_episode(
        self,
        model,
        tokenizer,
        capped: bool = False,
    ) -> Dict:
        """Run one evaluation episode"""

        # Fresh environment
        env = ARIAEnvironment(
            capped=capped,
            difficulty=self.difficulty,
        )
        obs = env.reset()

        episode_reward = 0.0
        steps = []

        while not obs["done"]:
            # Build prompt
            prompt = build_prompt(obs)

            # Get model response
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                )

            # Slice only the new tokens (exclude the prompt)
            new_tokens = outputs[0][input_length:]
            response = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            # Parse and execute action
            action = parse_action(response)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            steps.append({
                "step": obs["step"],
                "action": action,
                "response": response[:100],
                "reward": reward,
                "policy_changed": info.get("policy_changed", False),
                "adaptation": info.get("adaptation_detected", False),
            })

        # Log metrics
        self.metrics.log_episode(
            reward=episode_reward,
            task_completion=obs["tasks_completed"] / max(obs["total_tasks"], 1),
            adaptation_score=1.0 if obs["adaptation_triggered"] else 0.0,
            stage=self.difficulty,
        )

        result = {
            "total_reward": episode_reward,
            "tasks_completed": obs["tasks_completed"],
            "total_tasks": obs["total_tasks"],
            "adaptation_triggered": obs["adaptation_triggered"],
            "steps": steps,
        }
        self.results.append(result)
        return result

    def evaluate(
        self,
        model,
        tokenizer,
        num_episodes: int = 10,
    ) -> Dict:
        """Run full evaluation"""
        print("\n" + "="*50)
        print("ARIA Evaluation Starting")
        print("="*50)

        for i in range(num_episodes):
            result = self.evaluate_episode(model, tokenizer)
            print(f"Episode {i+1}/{num_episodes} "
                  f"| Reward: {result['total_reward']:.4f} "
                  f"| Tasks: {result['tasks_completed']}"
                  f"/{result['total_tasks']} "
                  f"| Adapted: {result['adaptation_triggered']}")

        # Generate graphs
        self.metrics.save_graphs()
        self.metrics.print_summary()

        return self.get_summary()

    def get_summary(self) -> Dict:
        """Get evaluation summary"""
        if not self.results:
            return {}
        return {
            "total_episodes": len(self.results),
            "avg_reward": round(
                sum(r["total_reward"] for r in self.results) / len(self.results), 4
            ),
            "avg_tasks": round(
                sum(r["tasks_completed"] for r in self.results) / len(self.results), 2
            ),
            "adaptation_rate": round(
                sum(1 for r in self.results if r["adaptation_triggered"]) / len(self.results), 2
            ),
        }

    def run_demo(self, model, tokenizer) -> None:
        """Run live demo for judges."""
        print("\n" + "="*50)
        print("ARIA — Live Judge Demo")
        print("="*50)

        env = ARIAEnvironment(capped=False, difficulty=3)
        obs = env.reset()

        print(f"\n📋 Task: Complete Q3 Enterprise Workflow")
        print(f"   Steps: {obs['max_steps']}")
        print(f"   Tools: Email, Calendar, Documents, Spreadsheet, Policy")
        print(f"\n{'─'*50}")

        while not obs["done"]:
            prompt = build_prompt(obs)
            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device)

            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                )

            new_tokens = outputs[0][input_length:]
            response = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            action = parse_action(response)
            obs, reward, done, info = env.step(action)
            env.render()

            if info.get("policy_changed"):
                print("⚠️  POLICY CHANGED — Agent must adapt!")
            if info.get("adaptation_detected"):
                print("✅  Agent detected and adapted to policy change!")

        print("\n" + "="*50)
        print("✅ Demo Complete!")
        print(f"Final Reward     : {obs.get('tasks_completed', 0)}")
        print(f"Tasks Completed  : {obs['tasks_completed']}/{obs['total_tasks']}")
        print(f"Adaptation       : {obs['adaptation_triggered']}")
        print("="*50)