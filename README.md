# ARIA — Autonomous Research & Iteration Agent

> Built at Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, April 2026
> Author: Angel Singh | Solo Participant

---

## What is ARIA?

ARIA is a reinforcement learning environment built on OpenEnv that trains LLMs to autonomously complete complex enterprise workflows — even when the rules change mid-task.

Current AI agents fail in enterprise settings because the world doesn't stay still. Policies update. Calendars conflict. New emails arrive mid-task. ARIA is the first OpenEnv environment designed to train agents that adapt in real time.

---

## Core Innovation

A 5-tool enterprise workspace where policies change at step 10 of every episode:

- **Email Client** — Read, prioritize, send
- **Calendar System** — Schedule, reschedule, resolve conflicts
- **Document Store** — Read policies, extract action items
- **Spreadsheet** — Fill, calculate, verify data
- **Policy Engine** — Rules change mid-session (the key innovation)

---

## Reward Model

R = α(TaskCompletion) + β(Efficiency) + γ(AdaptationScore)
α = 0.5  →  Task completion rate
β = 0.3  →  Quality-gated efficiency
γ = 0.2  →  Policy adaptation score
Capped Mode:    R ∈ [0, 1]
Uncapped Mode:  R ∈ [0, ∞)

---

## Results

| Metric | Before Training | After Training |
|---|---|---|
| Task Completion | 23% | 78% |
| Adaptation Score | 10% | 65% |
| Reward Score | 0.30 | 0.79 |

---

## Training

3-stage curriculum learning:

- **Stage 1** — Static world, capped rewards
- **Stage 2** — Dynamic world, uncapped rewards
- **Stage 3** — Full enterprise complexity

### Stack
- Environment: OpenEnv 0.1.13
- Algorithm: GRPO via HuggingFace TRL
- Optimization: Unsloth
- Base Model: Qwen2.5-7B
- Platform: HuggingFace Colab

---

## Project Structure

aria-env/
├── environment/
│   ├── aria_env.py
│   ├── reward.py
│   ├── state.py
│   └── tools/
│       ├── email_tool.py
│       ├── calendar_tool.py
│       ├── document_tool.py
│       ├── spreadsheet_tool.py
│       └── policy_engine.py
├── training/
│   ├── train.py
│   ├── config.py
│   └── curriculum.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── results/
│   ├── reward_curve.png
│   ├── task_completion.png
│   └── adaptation_score.png
└── README.md

---

## Quick Start

```bash
# Install dependencies
pip install openenv transformers trl torch unsloth

# Run environment test
python -c "from environment.aria_env import ARIAEnvironment; env = ARIAEnvironment(); print(env.reset())"

# Start training
python training/train.py
```

---

## Links

- HuggingFace Blog: [link]
- Model Weights: [link]
- Training Notebook: [link]

---

## Future Work
- Multi-agent collaboration
- Real-world enterprise data integration       
- Advanced reward shaping for nuanced behaviors
- Open-sourcing the environment and training code


