"""
ARIA - Autonomous Research & Iteration Agent
Main Training Script
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from environment.aria_env import ARIAEnvironment
from training.curriculum import CurriculumManager
from training.config import ARIAConfig
from evaluation.metrics import MetricsTracker


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(observation: dict) -> str:
    """Convert environment observation to LLM prompt"""
    return f"""You are ARIA, an autonomous enterprise AI agent.
Your job is to complete workplace tasks across multiple tools.

Current Status:
- Step: {observation['step']}/{observation['max_steps']}
- Tasks Completed: {observation['tasks_completed']}/{observation['total_tasks']}
- Policy Changed: {observation['policy_changed']}
- Adaptation Triggered: {observation['adaptation_triggered']}

Available Tools:
- email: list, read, send
- calendar: check, schedule, reschedule
- document: list, read
- spreadsheet: read, write
- policy: get

Current Policy:
{observation['policy']}

Inbox Summary:
{observation['inbox']}

Available Documents:
{observation['available_docs']}

Spreadsheet Status:
{observation['spreadsheet']}

Instructions:
1. Choose the most important next action
2. If policy changed, query policy tool immediately
3. Complete tasks efficiently with minimum tool calls
4. Respond ONLY in this exact format:

TOOL: <tool_name>
OPERATION: <operation_name>
PARAMS: <key=value pairs>

Your action:"""


# ─────────────────────────────────────────────
# ACTION PARSER
# ─────────────────────────────────────────────

def parse_action(response: str) -> dict:
    """Parse LLM response into environment action"""
    try:
        lines = response.strip().split("\n")
        action = {
            "tool": None,
            "operation": None,
            "params": {}
        }

        for line in lines:
            if line.startswith("TOOL:"):
                action["tool"] = line.replace("TOOL:", "").strip().lower()
            elif line.startswith("OPERATION:"):
                action["operation"] = line.replace("OPERATION:", "").strip().lower()
            elif line.startswith("PARAMS:"):
                params_str = line.replace("PARAMS:", "").strip()
                if params_str and params_str != "none":
                    for pair in params_str.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            action["params"][k.strip()] = v.strip()

        # Fallback if parsing fails
        if not action["tool"]:
            action = {
                "tool": "policy",
                "operation": "get",
                "params": {}
            }

        return action

    except Exception:
        return {
            "tool": "policy",
            "operation": "get",
            "params": {}
        }


# ─────────────────────────────────────────────
# REWARD FUNCTION FOR GRPO
# ─────────────────────────────────────────────

def make_reward_function(curriculum: CurriculumManager):
    """Create reward function for GRPO trainer"""

    def reward_function(samples, prompts, outputs, **kwargs):
        """
        Called by GRPO after each generation.
        Runs the action in environment and returns reward.
        """
        rewards = []
        config = curriculum.get_current_config()

        for output in outputs:
            # Create fresh environment
            env = ARIAEnvironment(
                capped=config.capped,
                difficulty=config.difficulty,
            )
            obs = env.reset()
            total_reward = 0.0

            # Run full episode
            for _ in range(config.max_steps):
                prompt = build_prompt(obs)
                action = parse_action(output)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)
            curriculum.log_reward(total_reward)

        return rewards

    return reward_function


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    print("\n" + "="*50)
    print("ARIA Training Starting")
    print("="*50)

    # Initialize curriculum
    curriculum = CurriculumManager()
    metrics = MetricsTracker()

    # ── Load Model ──
    print("\n📦 Loading model...")
    config = curriculum.get_current_config()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=config.load_in_4bit,
    )
    print(f"✅ Model loaded: {config.model_name}")

    # ── Training Loop ──
    while not curriculum.is_training_complete():
        config = curriculum.get_current_config()
        stage = curriculum.get_current_stage()

        print(f"\n🚀 Training Stage {stage}")
        curriculum.print_progress()

        # ── GRPO Config ──
        grpo_config = GRPOConfig(
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_generations=config.num_generations,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            num_train_epochs=config.num_train_epochs,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            output_dir=config.output_dir,
            report_to="none",
        )

        # ── Build Training Prompts ──
        env = ARIAEnvironment(
            capped=config.capped,
            difficulty=config.difficulty,
        )
        obs = env.reset()
        prompts = [{"prompt": build_prompt(obs)}
                   for _ in range(config.episodes_per_stage)]

        # ── GRPO Trainer ──
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=make_reward_function(curriculum),
            args=grpo_config,
            train_dataset=prompts,
        )

        # ── Train ──
        trainer.train()

        # ── Log Metrics ──
        avg_reward = curriculum.get_average_reward()
        metrics.log_stage(
            stage=stage,
            avg_reward=avg_reward,
            episodes=config.episodes_per_stage,
        )
        metrics.plot_reward_curve()

        # ── Check Advancement ──
        if curriculum.should_advance():
            advanced = curriculum.advance_stage()
            if not advanced:
                break

    # ── Training Complete ──
    print("\n" + "="*50)
    print("✅ ARIA Training Complete!")
    print("="*50)
    curriculum.print_progress()
    metrics.print_summary()
    metrics.save_graphs()

    # ── Save Model ──
    print("\n💾 Saving model...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ Model saved to {config.output_dir}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()