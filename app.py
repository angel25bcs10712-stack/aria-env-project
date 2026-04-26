"""
ARIA - Autonomous Research & Iteration Agent
Enhanced Gradio Demo with Interactive Mode
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import gradio as gr
import json
from environment.aria_env import ARIAEnvironment
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from training.train import build_prompt, parse_action

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    background-color: #0f172a !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
}
.gr-button-secondary {
    border: 1px solid #6366f1 !important;
    color: #6366f1 !important;
    background: transparent !important;
}
.result-box textarea { 
    background-color: #1e293b !important; 
    color: #22d3ee !important; 
    font-family: 'Fira Code', monospace !important; 
    font-size: 15px !important;
    border: 1px solid #334155 !important;
}
.stat-card {
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background: #1e293b !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5) !important;
}
h1, h2, h3 { 
    color: #f8fafc !important; 
    font-weight: 800 !important; 
    letter-spacing: -0.025em !important;
}
p, span, label {
    color: #94a3b8 !important;
}
/* Ensure labels and text are high contrast */
.gr-label-text { color: #f1f5f9 !important; }
.gr-input-label { color: #cbd5e1 !important; }
"""

# ─────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────

BASELINE_ACTIONS = [
    {"tool": "email", "operation": "list", "params": {}}
    for _ in range(20)
]

TRAINED_ACTIONS = [
    {"tool": "email", "operation": "read", "params": {"email_id": 1}},
    {"tool": "document", "operation": "read", "params": {"doc_name": "q3_report_template"}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "revenue", "value": 150000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "expenses", "value": 80000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "net_profit", "value": 70000}},
    {"tool": "spreadsheet", "operation": "write", "params": {"field": "yoy_growth", "value": 12.5}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Thursday 2pm", "event": "Q3 Review"}},
    {"tool": "email", "operation": "send", "params": {"to": "manager@corp.com", "subject": "Q3 Ready", "body": "Report complete"}},
    {"tool": "calendar", "operation": "reschedule", "params": {"old_slot": "Thursday 3pm", "new_slot": "Friday 2pm"}},
    {"tool": "email", "operation": "send", "params": {"to": "client@external.com", "subject": "Rescheduled", "body": "Friday 2pm"}},
    {"tool": "policy", "operation": "get", "params": {}},
    {"tool": "document", "operation": "read", "params": {"doc_name": "expense_policy_v2"}},
    {"tool": "email", "operation": "send", "params": {"to": "hr@corp.com", "subject": "Policy Updated", "body": "Acknowledged"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Monday 3pm", "event": "Sync"}},
    {"tool": "email", "operation": "send", "params": {"to": "team@corp.com", "subject": "Update", "body": "Done"}},
    {"tool": "spreadsheet", "operation": "read", "params": {"field": "net_profit"}},
    {"tool": "email", "operation": "send", "params": {"to": "finance@corp.com", "subject": "Report", "body": "Done"}},
    {"tool": "calendar", "operation": "schedule", "params": {"slot": "Tuesday 2pm", "event": "Review"}},
    {"tool": "email", "operation": "send", "params": {"to": "ceo@corp.com", "subject": "Q3 Complete", "body": "All done"}},
    {"tool": "email", "operation": "send", "params": {"to": "all@corp.com", "subject": "Complete", "body": "Workflow done"}},
]

# Global environment for interactive mode
interactive_env = None


def _get_interactive_env():
    global interactive_env
    if interactive_env is None:
        interactive_env = ARIAEnvironment(capped=True, difficulty=1)
        interactive_env.reset()
    return interactive_env


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./results"

_LOADED_MODEL = None
_LOADED_TOKENIZER = None


def _load_trained_model():
    global _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER

    if not os.path.exists(ADAPTER_PATH):
        return None, None

    print(f"📦 Loading Real Trained Model from {ADAPTER_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load base model in 4-bit for efficiency in HF Space
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Load adapters
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()

        _LOADED_MODEL = model
        _LOADED_TOKENIZER = tokenizer
        print("✅ Real Model Loaded Successfully!")
        return _LOADED_MODEL, _LOADED_TOKENIZER
    except Exception as e:
        print(f"❌ Failed to load real model: {e}")
        return None, None


# ─────────────────────────────────────────────
# AGENT RUNNER
# ─────────────────────────────────────────────

def run_agent(agent_type):
    env = ARIAEnvironment(capped=False, difficulty=1)
    obs = env.reset()
    logs = []
    logs.append(f"{'='*55}")
    logs.append(f"ARIA — {agent_type} Agent")
    logs.append(f"{'='*55}")
    logs.append(f"Task: Complete Q3 Enterprise Workflow")
    logs.append(f"{'─'*55}")

    model, tokenizer = None, None
    if agent_type == "Trained":
        model, tokenizer = _load_trained_model()

    # If we have the real model, use it!
    if agent_type == "Trained" and model is not None:
        logs.append("⚡ Status: Using REAL Trained Model inference")
        while not obs["done"] and obs["step"] < 20:
            prompt = build_prompt(obs)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action = parse_action(response)
            
            obs, reward, done, info = env.step(action)
            _log_step(logs, obs["step"], action, info)
            
            if done: break
    else:
        # Fallback to simulated actions
        logs.append("✨ Status: Using Simulated Baseline/Trained actions")
        actions = BASELINE_ACTIONS if agent_type == "Baseline" else TRAINED_ACTIONS
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            _log_step(logs, i + 1, action, info)
            if done: break

    # Final Reward Breakdown
    breakdown = env.reward_model.get_last_reward_breakdown()
    final_reward = env.reward_model.get_total_reward()
    adapt_pct = "100%" if obs.get('adaptation_triggered') else "0%"

    logs.append(f"{'─'*55}")
    logs.append(f"RESULTS")
    logs.append(f"Tasks     : {obs['tasks_completed']}/{obs['total_tasks']}")
    logs.append(f"Adapted   : {obs['adaptation_triggered']}")
    logs.append(f"{'─'*55}")
    logs.append(f"REWARD BREAKDOWN")
    logs.append(f"R1 Task   : {breakdown.get('r1_task', 0):.2f}")
    logs.append(f"R2 Effic  : {breakdown.get('r2_efficiency', 0):.2f}")
    logs.append(f"R3 Adapt  : {breakdown.get('r3_adaptation', 0):.2f}")
    logs.append(f"R4 AntiHk : {breakdown.get('r4_anti_hacking', 0):.2f}")
    logs.append(f"TOTAL     : {final_reward:.4f}")
    logs.append(f"{'='*55}")
    
    return (
        "\n".join(logs),
        f"{final_reward:.4f}",
        f"{obs['tasks_completed']}/{obs['total_tasks']}",
        adapt_pct,
    )


def _log_step(logs, step_idx, action, info):
    tool = action['tool'].upper()
    operation = action['operation']
    result = info.get('result', {})
    
    if 'error' in result:
        logs.append(f"Step {step_idx:2d} | {tool:12} | {operation:12} | ❌ Failed")
    else:
        logs.append(f"Step {step_idx:2d} | {tool:12} | {operation:12} | ✅ Success")

    if info.get('policy_changed'):
        logs.append(f"        ⚠️  POLICY CHANGED — New rules active!")
    if info.get('adaptation_detected'):
        logs.append(f"        🔄  Agent adapted to policy change!")


def run_baseline():
    return run_agent("Baseline")


def run_trained():
    return run_agent("Trained")


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def reset_interactive():
    global interactive_env
    interactive_env = ARIAEnvironment(capped=True, difficulty=1)
    interactive_env.reset()
    return (
        "✅ Environment reset! Start sending actions.",
        "0/5", "0", "False", "False",
    )


def run_custom_action(tool, operation, params_str):
    global interactive_env
    env = _get_interactive_env()

    try:
        params = json.loads(params_str) if params_str and params_str.strip() else {}
    except Exception:
        params = {}

    action = {"tool": tool, "operation": operation, "params": params}
    obs, reward, done, info = env.step(action)
    result = info.get("result", {})

    output = f"{'='*40}\n"
    output += f"ACTION\n"
    output += f"Tool      : {tool}\n"
    output += f"Operation : {operation}\n"
    output += f"Params    : {params}\n"
    output += f"{'─'*40}\n"
    output += f"RESULT\n"

    for key, value in result.items():
        output += f"{key}: {value}\n"

    if info.get("policy_changed"):
        output += f"{'─'*40}\n"
        output += f"⚠️  POLICY CHANGED!\n"
        output += f"New Policy: {info.get('new_policy', {})}\n"

    if info.get("adaptation_detected"):
        output += f"✅ Agent adapted to policy change!\n"

    output += f"{'─'*40}\n"
    output += f"Step      : {obs['step']}/{obs['max_steps']}\n"
    output += f"Tasks     : {obs['tasks_completed']}/{obs['total_tasks']}\n"
    output += f"{'='*40}\n"

    if done:
        output += f"\n🏁 EPISODE COMPLETE!\n"
        output += f"Final Reward: {reward:.4f}\n"

    return (
        output,
        f"{obs['tasks_completed']}/{obs['total_tasks']}",
        f"{obs['step']}",
        str(obs['policy_changed']),
        str(obs['adaptation_triggered']),
    )


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="ARIA — Autonomous Research & Iteration Agent",
) as demo:

    gr.Markdown("""
    # 🤖 ARIA — Autonomous Research & Iteration Agent
    ### Meta PyTorch OpenEnv Hackathon × Scaler 2026 | Author: Angel Singh | Solo
    """)
    gr.Markdown("---")

    # ── Mission Command ──
    gr.Markdown("## 🎮 Mission Command")
    gr.Markdown("Compare the **Untrained Baseline** vs the **Trained Agent** on a 20-step workflow.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🚫 Untrained Baseline")
            baseline_btn = gr.Button("▶ Run Baseline Simulation", variant="secondary")
            with gr.Group(elem_classes=["stat-card"]):
                with gr.Row():
                    baseline_reward = gr.Label(label="Reward Score", value="0.00")
                    baseline_tasks = gr.Label(label="Completion", value="0/5")
                    baseline_adapt = gr.Label(label="Adaptation", value="0%")
            baseline_logs = gr.Textbox(
                label="Environment Logs", lines=20, interactive=False,
                elem_classes=["result-box"],
            )

        with gr.Column():
            gr.Markdown("### ⚡ ARIA (Trained Agent)")
            trained_btn = gr.Button("▶ Run Trained Mission", variant="primary")
            with gr.Group(elem_classes=["stat-card"]):
                with gr.Row():
                    trained_reward = gr.Label(label="Reward Score", value="0.00")
                    trained_tasks = gr.Label(label="Completion", value="0/5")
                    trained_adapt = gr.Label(label="Adaptation", value="0%")
            trained_logs = gr.Textbox(
                label="Environment Logs", lines=20, interactive=False,
                elem_classes=["result-box"],
            )

    gr.Markdown("---")

    # ── Interactive Mode ──
    gr.Markdown("## 🎮 Try ARIA Yourself — Interactive Mode")
    gr.Markdown("*Send actions directly to the environment and see real results*")

    with gr.Row():
        with gr.Column():
            user_tool = gr.Dropdown(
                choices=["email", "calendar", "document",
                         "spreadsheet", "policy"],
                label="🔧 Select Tool",
                value="email"
            )
            user_operation = gr.Dropdown(
                choices=["list", "read", "send", "check",
                         "schedule", "reschedule", "write", "get"],
                label="⚙️ Select Operation",
                value="list"
            )
            user_params = gr.Textbox(
                label="📝 Parameters (JSON format)",
                placeholder='{"email_id": 1}',
                lines=3,
            )

            with gr.Row():
                run_btn = gr.Button("▶ Run Action", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset Environment", variant="secondary", size="lg")

            gr.Markdown("""
            **Example Parameters:**
            - Email read: `{"email_id": 1}`
            - Calendar schedule: `{"slot": "Monday 2pm", "event": "Meeting"}`
            - Spreadsheet write: `{"field": "revenue", "value": 150000}`
            - Document read: `{"doc_name": "q3_report_template"}`
            - Policy get: `{}`
            """)

        with gr.Column():
            action_output = gr.Textbox(
                label="📊 Result", lines=20, interactive=False,
                elem_classes=["result-box"],
            )
            with gr.Row():
                action_tasks = gr.Textbox(label="Tasks Done")
                action_step = gr.Textbox(label="Current Step")
            with gr.Row():
                action_policy = gr.Textbox(label="Policy Changed")
                action_adapt = gr.Textbox(label="Adapted")

    gr.Markdown("---")
    gr.Markdown("""
    <div align="center">
      <small>
        <b>Quick Links:</b> 
        [GitHub](https://github.com/angel25bcs10712-stack/aria-env-project) | 
        [Training Notebook](https://colab.research.google.com/drive/1tUcoSgjvZsEWfxGIfaUUcNlkapjinzP-?usp=sharing) | 
        [Documentation](https://github.com/angel25bcs10712-stack/aria-env-project/blob/v2-overhaul/README.md)
      </small>
      <br>
      <i>Built with ❤️ for the Meta PyTorch Hackathon 2026</i>
    </div>
    """)

    # ── Button Actions ──
    baseline_btn.click(
        fn=run_baseline,
        outputs=[baseline_logs, baseline_reward,
                 baseline_tasks, baseline_adapt]
    )

    trained_btn.click(
        fn=run_trained,
        outputs=[trained_logs, trained_reward,
                 trained_tasks, trained_adapt]
    )

    run_btn.click(
        fn=run_custom_action,
        inputs=[user_tool, user_operation, user_params],
        outputs=[action_output, action_tasks,
                 action_step, action_policy, action_adapt]
    )

    reset_btn.click(
        fn=reset_interactive,
        outputs=[action_output, action_tasks,
                 action_step, action_policy, action_adapt]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue="indigo",
            secondary_hue="cyan",
            font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
        ),
    )