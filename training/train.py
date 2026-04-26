"""
ARIA - Autonomous Research & Iteration Agent
Main Training Script
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from datasets import Dataset
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
    """
    Create reward function for GRPO trainer.

    Each output from the model is a single action string. We run a full
    episode using that action repeatedly (simulating a fixed-policy rollout)
    and return the episode's final reward.
    """
    episode_rewards = []  # Collect rewards outside GRPO's internal loop

    def reward_function(prompts, completions, **kwargs):
        """
        Called by GRPO after each generation batch.
        Runs the parsed action through a full episode and returns reward.
        """
        rewards = []
        config = curriculum.get_current_config()

        for completion in completions:
            # Extract the text content from the completion
            if isinstance(completion, list):
                # Chat format: list of dicts with 'content'
                text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            else:
                text = str(completion)

            # Create fresh environment for this rollout
            env = ARIAEnvironment(
                capped=config.capped,
                difficulty=config.difficulty,
            )
            obs = env.reset()
            total_reward = 0.0

            # Parse the model's output into an action
            action = parse_action(text)

            # Run full episode with this action
            # (simulates a fixed-policy agent)
            for step_idx in range(config.max_steps):
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break

            # The final reward from the episode
            final_reward = total_reward
            rewards.append(final_reward)
            episode_rewards.append(final_reward)

        return rewards

    return reward_function, episode_rewards


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

def build_training_dataset(config: ARIAConfig) -> Dataset:
    """
    Build a dataset of diverse prompts for training.
    Each prompt comes from a fresh environment reset.
    """
    prompts = []
    for _ in range(config.episodes_per_stage):
        env = ARIAEnvironment(
            capped=config.capped,
            difficulty=config.difficulty,
        )
        obs = env.reset()
        prompts.append({"prompt": build_prompt(obs)})

    return Dataset.from_list(prompts)


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Proper quantization config
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
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
            max_completion_length=config.max_completion_length,
            num_train_epochs=config.num_train_epochs,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            output_dir=config.output_dir,
            bf16=False,
            fp16=True,
            report_to="none",
        )

        # ── Build Training Dataset (diverse prompts) ──
        train_dataset = build_training_dataset(config)

        # ── Create reward function with external reward tracker ──
        reward_fn, episode_rewards = make_reward_function(curriculum)

        # ── GRPO Trainer ──
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=train_dataset,
            peft_config=peft_config,
        )

        # ── Train ──
        trainer.train()

        # ── Log rewards AFTER training (not inside reward fn) ──
        for r in episode_rewards:
            curriculum.log_reward(r)

        # ── Log Metrics ──
        avg_reward = curriculum.get_average_reward()
        metrics.log_stage(
            stage=stage,
            avg_reward=avg_reward,
            episodes=len(episode_rewards),
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