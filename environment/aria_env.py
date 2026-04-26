"""
ARIA - Autonomous Research & Iteration Agent
Main Environment
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

import random
from environment.state import ARIAState
from environment.tools.email_tool import EmailTool
from environment.tools.calendar_tool import CalendarTool
from environment.tools.document_tool import DocumentTool
from environment.tools.spreadsheet_tool import SpreadsheetTool
from environment.tools.policy_engine import PolicyEngine
from environment.reward import RewardModel


class ARIAEnvironment:
    """
    ARIA: Autonomous Research & Iteration Agent Environment

    A dynamic enterprise workspace where an agent must complete
    multi-app workflows across long-horizon sessions, even when
    rules change mid-task.

    Difficulty Levels:
        1 = Stage 1: 3 tools, static policy, capped rewards
        2 = Stage 2: 5 tools, one policy change, uncapped rewards
        3 = Stage 3: Full 20-step workflow, multiple policy changes
    """

    def __init__(self, capped: bool = True, difficulty: int = 1):
        self.capped = capped
        self.difficulty = difficulty

        # Initialize tools
        self.email = EmailTool()
        self.calendar = CalendarTool()
        self.documents = DocumentTool()
        self.spreadsheet = SpreadsheetTool()
        self.policy_engine = PolicyEngine()

        # Initialize state
        self.state = ARIAState()

        # Initialize reward model
        self.reward_model = RewardModel(capped=capped)

        # Total tasks based on difficulty
        self.total_tasks = {1: 5, 2: 10, 3: 20}[difficulty]

        # Minimum tool calls (ideal scenario)
        self.min_tool_calls = self.total_tasks

        # Randomize policy change step for higher difficulties
        if difficulty == 1:
            self.policy_change_step = 10  # Fixed for Stage 1
        elif difficulty == 2:
            self.policy_change_step = random.randint(8, 12)
        else:
            self.policy_change_step = random.randint(6, 14)

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────

    def reset(self) -> dict:
        """Reset environment to initial state"""
        self.state.reset()
        self.email = EmailTool()
        self.calendar = CalendarTool()
        self.documents = DocumentTool()
        self.spreadsheet = SpreadsheetTool()
        self.policy_engine = PolicyEngine()
        self.reward_model = RewardModel(capped=self.capped)

        # Re-randomize policy change step
        if self.difficulty == 1:
            self.policy_change_step = 10
        elif self.difficulty == 2:
            self.policy_change_step = random.randint(8, 12)
        else:
            self.policy_change_step = random.randint(6, 14)

        return self._get_observation()

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────

    def step(self, action: dict) -> tuple:
        """
        Execute one step in the environment.

        Args:
            action: {
                "tool": "email" | "calendar" | "document" |
                        "spreadsheet" | "policy",
                "operation": str,
                "params": dict
            }

        Returns:
            (observation, reward, done, info)
        """
        self.state.increment_step()
        reward = 0.0
        info = {}

        # ── Policy change at configured step ──
        if self.state.step == self.policy_change_step and not self.state.policy_changed:
            self.policy_engine.update_policy()
            self.state.trigger_policy_change()
            info["policy_changed"] = True
            info["new_policy"] = self.policy_engine.get_policy()

        # ── Execute action ──
        tool = action.get("tool")
        operation = action.get("operation")
        params = action.get("params", {})

        result = self._execute_tool(tool, operation, params)
        info["result"] = result

        # ── Log action history ──
        self.state.action_history.append({
            "step": self.state.step,
            "action": action,
            "result": result,
        })

        # ── Check adaptation ──
        if (self.state.policy_changed and
                tool == "policy" and
                operation == "get"):
            self.state.trigger_adaptation()
            info["adaptation_detected"] = True

        # ── Mark task complete for valid actions ──
        if (self._is_valid_result(result) and
                len(self.state.tasks_completed) < self.total_tasks):
            self.state.add_completed_task(f"step_{self.state.step}")

        # ── Per-step shaping reward ──
        step_reward = self.reward_model.compute_step_reward(
            action=action,
            result=result,
            tasks_completed=len(self.state.tasks_completed),
            total_tasks=self.total_tasks,
            policy_changed=self.state.policy_changed,
            adaptation_triggered=self.state.adaptation_triggered,
        )
        reward += step_reward

        # ── Check if episode done ──
        if self.state.step >= self.state.max_steps:
            self.state.mark_done()
            final_reward = self.reward_model.compute(
                tasks_completed=len(self.state.tasks_completed),
                total_tasks=self.total_tasks,
                tool_calls=self.state.tool_calls,
                min_tool_calls=self.min_tool_calls,
                adaptation_triggered=self.state.adaptation_triggered,
                policy_changed=self.state.policy_changed,
                action_history=self.state.action_history,
            )
            reward += final_reward
            self.state.cumulative_reward += reward
            info["final_reward"] = reward
            info["summary"] = self.state.summary()

        observation = self._get_observation()
        return observation, reward, self.state.done, info

    # ─────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────

    def _get_observation(self) -> dict:
        """Return current environment observation"""
        return {
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "policy": self.policy_engine.get_policy(),
            "inbox": self.email.list_inbox(),
            "available_docs": self.documents.list_docs(),
            "spreadsheet": self.spreadsheet.summary(),
            "tasks_completed": len(self.state.tasks_completed),
            "total_tasks": self.total_tasks,
            "policy_changed": self.state.policy_changed,
            "adaptation_triggered": self.state.adaptation_triggered,
            "done": self.state.done,
        }

    # ─────────────────────────────────────────
    # TOOL EXECUTION
    # ─────────────────────────────────────────

    def _execute_tool(self, tool: str, operation: str, params: dict) -> dict:
        """Route action to correct tool"""
        try:
            if tool == "email":
                return self._handle_email(operation, params)
            elif tool == "calendar":
                return self._handle_calendar(operation, params)
            elif tool == "document":
                return self._handle_document(operation, params)
            elif tool == "spreadsheet":
                return self._handle_spreadsheet(operation, params)
            elif tool == "policy":
                return self._handle_policy(operation, params)
            else:
                return {"error": f"Unknown tool: {tool}"}
        except Exception as e:
            return {"error": str(e)}

    def _handle_email(self, operation: str, params: dict) -> dict:
        if operation == "list":
            return {"emails": self.email.list_inbox()}
        elif operation == "read":
            return self.email.read(params.get("email_id"))
        elif operation == "send":
            return self.email.send(
                params.get("to"),
                params.get("subject"),
                params.get("body")
            )
        return {"error": "Unknown email operation"}

    def _handle_calendar(self, operation: str, params: dict) -> dict:
        if operation == "check":
            return self.calendar.check(params.get("slot"))
        elif operation == "schedule":
            return self.calendar.schedule(
                params.get("slot"),
                params.get("event")
            )
        elif operation == "reschedule":
            return self.calendar.reschedule(
                params.get("old_slot"),
                params.get("new_slot")
            )
        return {"error": "Unknown calendar operation"}

    def _handle_document(self, operation: str, params: dict) -> dict:
        if operation == "list":
            return {"docs": self.documents.list_docs()}
        elif operation == "read":
            return self.documents.read(params.get("doc_name"))
        return {"error": "Unknown document operation"}

    def _handle_spreadsheet(self, operation: str, params: dict) -> dict:
        if operation == "read":
            return self.spreadsheet.read(params.get("field"))
        elif operation == "write":
            return self.spreadsheet.write(
                params.get("field"),
                params.get("value")
            )
        return {"error": "Unknown spreadsheet operation"}

    def _handle_policy(self, operation: str, params: dict) -> dict:
        if operation == "get":
            return self.policy_engine.get_policy()
        return {"error": "Unknown policy operation"}

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _is_valid_result(self, result: dict) -> bool:
        """Check if tool action produced a valid result"""
        if "error" in result:
            return False
        valid_statuses = ["sent", "scheduled", "rescheduled", "written"]
        if result.get("status") in valid_statuses:
            return True
        return False

    def render(self):
        """Print current state to terminal"""
        print("\n" + "="*50)
        print(f"ARIA Environment — Step {self.state.step}/{self.state.max_steps}")
        print("="*50)
        print(f"Tasks Completed : {len(self.state.tasks_completed)}/{self.total_tasks}")
        print(f"Tool Calls      : {self.state.tool_calls}")
        print(f"Policy Changed  : {self.state.policy_changed}")
        print(f"Adaptation      : {self.state.adaptation_triggered}")
        print(f"Reward So Far   : {self.state.cumulative_reward:.4f}")
        print("="*50)