"""
ARIA - Autonomous Research & Iteration Agent
Spreadsheet Tool
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import Dict, Any, List, Optional


class SpreadsheetTool:
    """
    Simulates an enterprise spreadsheet.
    Agent can read and write data fields.
    """

    def __init__(self):
        self.sheets: Dict[str, Dict] = {
            "q3_report": {
                "revenue": None,
                "expenses": None,
                "net_profit": None,
                "yoy_growth": None,
            },
            "expense_tracker": {
                "total_expenses": None,
                "approved_expenses": None,
                "pending_expenses": None,
                "rejected_expenses": None,
            },
            "project_tracker": {
                "phase_1_status": None,
                "phase_2_status": None,
                "phase_3_status": None,
                "completion_percentage": None,
            },
        }
        self.active_sheet: str = "q3_report"

    def write(self, field: str, value: Any) -> Dict:
        """Write a value to a field in active sheet"""
        if not field:
            return {"error": "Field is required"}
        if field not in self.sheets[self.active_sheet]:
            return {"error": f"Field '{field}' not found in {self.active_sheet}"}
        self.sheets[self.active_sheet][field] = value
        return {
            "status": "written",
            "sheet": self.active_sheet,
            "field": field,
            "value": value,
        }

    def read(self, field: str) -> Dict:
        """Read a value from active sheet"""
        if not field:
            return {"error": "Field is required"}
        if field not in self.sheets[self.active_sheet]:
            return {"error": f"Field '{field}' not found in {self.active_sheet}"}
        return {
            "status": "success",
            "sheet": self.active_sheet,
            "field": field,
            "value": self.sheets[self.active_sheet][field],
        }

    def switch_sheet(self, sheet_name: str) -> Dict:
        """Switch to a different sheet"""
        if sheet_name not in self.sheets:
            return {"error": f"Sheet '{sheet_name}' not found"}
        self.active_sheet = sheet_name
        return {
            "status": "switched",
            "active_sheet": self.active_sheet,
        }

    def is_complete(self) -> bool:
        """Check if all fields in active sheet are filled"""
        return all(
            v is not None
            for v in self.sheets[self.active_sheet].values()
        )

    def list_sheets(self) -> List[str]:
        """List all available sheets"""
        return list(self.sheets.keys())

    def list_fields(self) -> List[str]:
        """List all fields in active sheet"""
        return list(self.sheets[self.active_sheet].keys())

    def summary(self) -> Dict:
        """Return spreadsheet summary"""
        active = self.sheets[self.active_sheet]
        filled = sum(1 for v in active.values() if v is not None)
        return {
            "active_sheet": self.active_sheet,
            "total_fields": len(active),
            "filled_fields": filled,
            "empty_fields": len(active) - filled,
            "is_complete": self.is_complete(),
            "data": active,
        }