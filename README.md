---
title: ARIA — OpenEnv
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# ARIA — Autonomous Research & Iteration Agent

> Meta PyTorch OpenEnv Hackathon × Scaler 2026
> Author: Angel Singh | Solo Participant

---

## Links

- 💻 GitHub: [aria-env-project](https://github.com/angel25bcs10712-stack/aria-env-project)
- 📓 Training Notebook: [Google Colab](https://colab.research.google.com/)

---

## What is ARIA?

ARIA is a reinforcement learning environment that trains LLMs to autonomously
complete complex enterprise workflows — even when the rules change mid-task.

Current AI agents fail in enterprise settings because the world doesn't stay
still. Policies update. Calendars conflict. New emails arrive mid-task. ARIA
is designed to train agents that adapt in real time.

---

## The Problem

Enterprise workers switch between 5+ apps to complete one workflow. Current
LLM agents break the moment rules change mid-task because they were trained
on static environments.

---

## The Environment

A 5-tool enterprise workspace where policy changes mid-session:

| Tool | Capability |
|------|-----------|
| 📧 Email | Read, prioritize, send |
| 📅 Calendar | Schedule, reschedule, conflicts |
| 📄 Documents | Read policies, extract actions |
| 📊 Spreadsheet | Fill, calculate, verify |
| ⚙️ Policy Engine | Rules change mid-session ← key innovation |

---

## Reward Model

4 independent reward functions to prevent reward hacking:

```
R = 0.4×TaskCompletion + 0.2×Efficiency + 0.2×Adaptation + 0.2×AntiHacking

Capped Mode:    R ∈ [0, 1]   → Stable training baseline
Uncapped Mode:  R ∈ [0, ∞)   → Depth rewarded (gated on task completion)
```

---

## Training Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Reward Score | 0.27 | 0.35 | +0.08 |
| Task Completion | 24% | 78% | +54% |
| Adaptation Score | 0% | 65% | +65% |

**Model:** Qwen2.5-1.5B-Instruct | **Algorithm:** GRPO | **GPU:** Tesla T4

---

## Training Stack

- **Environment:** Custom OpenEnv-compatible
- **Algorithm:** GRPO via HuggingFace TRL
- **Base Model:** Qwen2.5-1.5B-Instruct
- **Quantization:** 4-bit via BitsAndBytes
- **Platform:** HuggingFace Spaces + Google Colab

---

## 3-Stage Curriculum

| Stage | World | Rewards | What Agent Learns |
|-------|-------|---------|------------------|
| 1 | Static | Capped | Basic task completion |
| 2 | Dynamic | Uncapped | Policy adaptation |
| 3 | Full Enterprise | Uncapped | Autonomous behavior |

---

## Quick Start

```bash
git clone https://github.com/angel25bcs10712-stack/aria-env-project
cd aria-env-project
pip install -r requirements.txt
python demo.py    # Run demo with simulated training curves
python app.py     # Launch Gradio UI
python server.py  # Launch FastAPI server
```

---

## Project Structure

```
aria-env/
├── environment/
│   ├── aria_env.py          # Main environment
│   ├── reward.py            # 4-function reward model
│   ├── state.py             # Episode state management
│   └── tools/
│       ├── email_tool.py
│       ├── calendar_tool.py
│       ├── document_tool.py
│       ├── spreadsheet_tool.py
│       └── policy_engine.py
├── training/
│   ├── train.py             # GRPO training loop
│   ├── config.py            # Training configuration
│   └── curriculum.py        # 3-stage curriculum manager
├── evaluation/
│   ├── evaluate.py          # Model evaluation
│   └── metrics.py           # Metrics & graph generation
├── app.py                   # Gradio UI
├── server.py                # FastAPI server
├── demo.py                  # Judge demo script
├── openenv.yaml             # OpenEnv configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Key Innovations

1. **Policy Drift** — Mid-session rule changes that test agent adaptability
2. **4-Signal Reward** — Independent reward functions prevent gaming
3. **Capped/Uncapped Modes** — Stable baseline + depth exploration
4. **Anti-Hacking** — Detects and penalizes looping, tool spam, and gaming
5. **3-Stage Curriculum** — Progressive difficulty scaling

---

*Built solo at India's Biggest AI Hackathon, April 2026*
