"""
ARIA - Autonomous Research & Iteration Agent
Training Configuration
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from dataclasses import dataclass


@dataclass
class ARIAConfig:
    """
    Complete configuration for ARIA training.
    Adjust these values onsite with compute credits.
    """

    # ─────────────────────────────────────────
    # MODEL CONFIG
    # ─────────────────────────────────────────

    # Base model to train (1.5B fits on T4/free Colab)
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Max tokens per response
    max_new_tokens: int = 256

    # Max sequence length
    max_seq_length: int = 1024

    # Load in 4bit for memory efficiency
    load_in_4bit: bool = True

    # Use bf16 if available, else fp16
    use_bf16: bool = False

    # ─────────────────────────────────────────
    # ENVIRONMENT CONFIG
    # ─────────────────────────────────────────

    # Training stages
    # Stage 1: capped=True,  difficulty=1
    # Stage 2: capped=False, difficulty=2
    # Stage 3: capped=False, difficulty=3
    capped: bool = True
    difficulty: int = 1

    # Episodes per training stage
    episodes_per_stage: int = 50

    # Max steps per episode
    max_steps: int = 20

    # ─────────────────────────────────────────
    # GRPO TRAINING CONFIG
    # ─────────────────────────────────────────

    # Learning rate
    learning_rate: float = 5e-6

    # Batch size
    per_device_train_batch_size: int = 1

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Number of generations per prompt
    num_generations: int = 4

    # Max prompt length
    max_prompt_length: int = 512

    # Max completion length
    max_completion_length: int = 256

    # Training epochs
    num_train_epochs: int = 1

    # Warmup steps
    warmup_steps: int = 10

    # ─────────────────────────────────────────
    # LOGGING CONFIG
    # ─────────────────────────────────────────

    # Log every N steps
    logging_steps: int = 5

    # Save every N steps
    save_steps: int = 50

    # Output directory
    output_dir: str = "./results"

    # HuggingFace repo to push model
    hub_model_id: str = "angel-singh/aria-env"

    # ─────────────────────────────────────────
    # CURRICULUM CONFIG
    # ─────────────────────────────────────────

    # Reward threshold to advance to next stage
    stage_advance_threshold: float = 0.5

    # Maximum episodes before forced stage advance
    max_episodes_per_stage: int = 150

    # Rolling window size for advancement check
    advancement_window: int = 20


# Default config instance
default_config = ARIAConfig()


# Stage specific configs
stage1_config = ARIAConfig(
    capped=True,
    difficulty=1,
    episodes_per_stage=50,
    learning_rate=5e-6,
)

stage2_config = ARIAConfig(
    capped=False,
    difficulty=2,
    episodes_per_stage=80,
    learning_rate=3e-6,
)

stage3_config = ARIAConfig(
    capped=False,
    difficulty=3,
    episodes_per_stage=100,
    learning_rate=1e-6,
)