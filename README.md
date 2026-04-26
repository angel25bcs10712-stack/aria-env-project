---
title: ARIA — OpenEnv
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

<div align="center">
  <h1>🤖 ARIA</h1>
  <h3>Autonomous Research & Iteration Agent</h3>
  <p><em>An RL-trained LLM agent that completes complex enterprise workflows—even when the rules change mid-task.</em></p>
  
  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/angel25bcs10712/ARIA-OpenEnv)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
  
  <p><b>Built solo for the Meta PyTorch OpenEnv Hackathon × Scaler 2026</b><br>Author: Angel Singh</p>
</div>

---

## 🔗 Quick Links
- 💻 **GitHub Repository**: [angel25bcs10712-stack/aria-env-project (v2-overhaul)](https://github.com/angel25bcs10712-stack/aria-env-project/tree/v2-overhaul)
- 🤗 **Hugging Face Space**: [ARIA-OpenEnv Live Demo](https://huggingface.co/spaces/angel25bcs10712/ARIA-OpenEnv)
- 📓 **Training Notebook**: [Open ARIA_Colab.ipynb in Google Colab](https://colab.research.google.com/drive/1kXTLVXXo9pmAPFKzQtM3xFk0gb2v-Vqf)

- 📝 **Hugging Face Blog**: [Read the project write-up here](#) *(Add your link here!)*

---

## 📖 Table of Contents
- [The Problem: Why ARIA?](#-the-problem-why-aria)
- [The Solution: Policy Drift](#-the-solution-policy-drift)
- [The Environment](#-the-environment)
- [How It Works (Architecture)](#-how-it-works-architecture)
  - [1. The 4-Signal Reward Model](#1-the-4-signal-reward-model)
  - [2. 3-Stage Curriculum Learning](#2-3-stage-curriculum-learning)
  - [3. GRPO Training](#3-grpo-training)
- [Performance Results](#-performance-results)
- [Getting Started (Local Setup)](#-getting-started-local-setup)
- [Project Structure](#-project-structure)

---

## 🚨 The Problem: Why ARIA?

Today's AI agents are great at completing tasks in **static** environments. However, enterprise workflows are chaotic and **dynamic**. 

Imagine an agent tasked with scheduling a meeting and approving expenses. What happens if, halfway through the task, the company's expense policy changes? Current LLM agents fail because they execute predetermined plans without re-verifying context. **They break the moment the rules change.**

## 💡 The Solution: Policy Drift

**ARIA** is designed specifically to solve this. It is a reinforcement learning (RL) environment built on the OpenEnv standard that simulates a chaotic enterprise workspace.

The core innovation of ARIA is **Policy Drift**—the environment intentionally changes the underlying rules mid-session. The agent is trained via Reinforcement Learning to continuously verify context, detect when rules have changed, and adapt its actions dynamically.

---

## 🏢 The Environment

ARIA provides the agent with an isolated workspace containing **5 interconnected tools**:

| Tool | Capability | How the Agent uses it |
|------|-----------|-----------------------|
| 📧 **Email** | Read, list, send | Reading instructions and reporting task completion. |
| 📅 **Calendar** | Schedule, check, reschedule | Managing conflicting schedules and booking meetings. |
| 📄 **Documents** | List, read | Extracting data from templates and reports. |
| 📊 **Spreadsheet** | Read, write | Calculating metrics like Revenue and Net Profit. |
| ⚙️ **Policy Engine** | `get_policy` | **[CRITICAL]** The agent must ping this to detect rule changes. |

---

## 🧠 How It Works (Architecture)

Training an agent to be this autonomous requires a carefully designed architecture to prevent "reward hacking" (where the agent cheats the system to get a high score without doing the work).

### 1. The 4-Signal Reward Model
Instead of a single reward, ARIA uses 4 independent signals combined into a final score:
- **R1 (Task Completion - 40%)**: Did the agent actually finish the workflow?
- **R2 (Efficiency - 20%)**: Did the agent use the minimum necessary tool calls?
- **R3 (Adaptation - 20%)**: Did the agent detect the mid-session policy change?
- **R4 (Anti-Hacking - 20%)**: Penalizes looping, redundant calls, and trying to game the system.

### 2. 3-Stage Curriculum Learning
You can't teach an agent to adapt before it knows how to use tools. ARIA uses a progressively scaling curriculum:
1. **Stage 1 (Static):** Basic task completion. No policy changes. Rewards are strictly capped.
2. **Stage 2 (Dynamic):** One policy change per session. Agent is rewarded for adapting.
3. **Stage 3 (Enterprise):** Long-horizon 20-step workflows with multiple sudden policy changes. Uncapped rewards for deep exploration.

### 3. GRPO Training
The agent is trained using **Group Relative Policy Optimization (GRPO)** via Hugging Face `trl`. The base model is `Qwen2.5-1.5B-Instruct` loaded efficiently in 4-bit quantization via `bitsandbytes`.

---

## 📊 Performance Results

After 1,000+ steps of GRPO training, the transformation in the agent's behavior is drastic:

| Metric | Before Training (Baseline) | After Training (Trained) | Improvement |
|--------|----------------------------|--------------------------|-------------|
| **Total Reward Score** | `0.27` | `0.83+` | **+ 207%** |
| **Task Completion Rate** | `24%` | `78%` | **+ 54%** |
| **Adaptation Score** | `0%` | `65%` | **+ 65%** |

*The Baseline agent ignores policy changes and gets stuck in loops. The Trained agent dynamically adapts to new rules and finishes multi-app workflows.*

---

## 🚀 Getting Started (Local Setup)

You can run the full environment, including the interactive UI, locally on your machine.

### Option A: Using Docker (Recommended)
```bash
# 1. Clone the repository
git clone -b v2-overhaul https://github.com/angel25bcs10712-stack/aria-env-project.git
cd aria-env-project

# 2. Build and run using Docker Compose
docker-compose up --build
```
*The app will be live at `http://localhost:7860`*

### Option B: Using Python
```bash
# 1. Clone the repository
git clone -b v2-overhaul https://github.com/angel25bcs10712-stack/aria-env-project.git
cd aria-env-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio UI
python app.py

# 4. (Optional) Run the Training Loop locally (Requires GPU)
PYTHONPATH=. python training/train.py
```

### Try the Interactive Mode!
Once the Gradio app is running, navigate to the **Interactive Mode** tab. You can manually send actions to the environment (like `{"tool": "spreadsheet", "operation": "write"}`) and see exactly how the environment responds and calculates rewards.

---

## 📂 Project Structure

```text
aria-env-project/
├── environment/             # Core RL Environment
│   ├── aria_env.py          # State tracking & step execution
│   ├── reward.py            # Multi-objective reward model
│   └── tools/               # The 5 enterprise tools
├── training/                # Model Training Logic
│   ├── train.py             # Main GRPO training loop
│   ├── curriculum.py        # 3-Stage curriculum manager
│   └── config.py            # Hyperparameters
├── evaluation/              # Testing & Metrics
│   ├── evaluate.py          # Model inference script
│   └── metrics.py           # Matplotlib graph generation
├── app.py                   # Gradio Web UI (HF Spaces entry point)
├── demo.py                  # CLI Judge Demo
├── openenv.yaml             # OpenEnv metadata specification
├── Dockerfile               # Production Docker image
└── requirements.txt         # Python dependencies
```

---
<div align="center">
  <i>Created with ❤️ for the OpenEnv Hackathon</i>
</div>
