"""
ARIA - Autonomous Research & Iteration Agent
FastAPI Server
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from environment.aria_env import ARIAEnvironment

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="ARIA Environment",
    description="Autonomous Research & Iteration Agent — OpenEnv Compatible",
    version="1.0.0",
)

# Global environment instance
env: Optional[ARIAEnvironment] = None

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ActionRequest(BaseModel):
    tool: str
    operation: str
    params: Dict[str, Any] = {}

class ResetRequest(BaseModel):
    capped: bool = True
    difficulty: int = 1

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "ARIA Environment",
        "version": "1.0.0",
        "author": "Angel Singh",
        "hackathon": "Meta PyTorch OpenEnv Hackathon x Scaler 2026",
        "status": "running",
    }

@app.post("/reset")
def reset(request: ResetRequest):
    """Reset environment to initial state"""
    global env
    env = ARIAEnvironment(
        capped=request.capped,
        difficulty=request.difficulty,
    )
    observation = env.reset()
    return {
        "status": "reset",
        "observation": observation,
    }

@app.post("/step")
def step(action: ActionRequest):
    """Execute one step in environment"""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    action_dict = {
        "tool": action.tool,
        "operation": action.operation,
        "params": action.params,
    }

    observation, reward, done, info = env.step(action_dict)

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    """Get current environment state"""
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return {
        "state": env.state.summary(),
        "policy": env.policy_engine.get_policy(),
        "spreadsheet": env.spreadsheet.summary(),
    }

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}

@app.post("/render")
def render():
    """Render current environment state"""
    global env
    if env is None:
        return {"error": "Environment not initialized"}
    env.render()
    return {"status": "rendered"}

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)