"""
ARIA - Autonomous Research & Iteration Agent
Document Tool
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import Dict, List


class DocumentTool:
    """
    Simulates an enterprise document store.
    Agent can read and list documents.
    """

    def __init__(self):
        self.store: Dict[str, Dict] = {
            "q3_report_template": {
                "title": "Q3 Report Template",
                "content": "Q3 Report Template: Revenue | Expenses | Net Profit | YoY Growth",
                "category": "report",
                "last_updated": "2026-04-01",
            },
            "expense_policy_v1": {
                "title": "Expense Policy v1",
                "content": "Expense Policy v1: Max reimbursement $1000 per trip. No receipts required.",
                "category": "policy",
                "last_updated": "2026-01-01",
            },
            "expense_policy_v2": {
                "title": "Expense Policy v2",
                "content": "Expense Policy v2: Max reimbursement $500 per trip. Receipts mandatory.",
                "category": "policy",
                "last_updated": "2026-04-19",
            },
            "q3_revenue_data": {
                "title": "Q3 Revenue Data",
                "content": "Revenue: $150,000 | Expenses: $80,000 | Net Profit: $70,000 | YoY Growth: 12.5%",
                "category": "data",
                "last_updated": "2026-04-15",
            },
            "project_timeline": {
                "title": "Project Timeline",
                "content": "Phase 1: April | Phase 2: May | Phase 3: June | Deadline: July 1st",
                "category": "project",
                "last_updated": "2026-04-10",
            },
        }

    def read(self, doc_name: str) -> Dict:
        """Read a specific document by name"""
        if not doc_name:
            return {"error": "Document name is required"}
        if doc_name not in self.store:
            return {"error": f"Document '{doc_name}' not found"}
        return {
            "status": "success",
            "content": self.store[doc_name]["content"],
            "title": self.store[doc_name]["title"],
            "last_updated": self.store[doc_name]["last_updated"],
        }

    def list_docs(self) -> List[Dict]:
        """List all available documents"""
        return [
            {
                "name": name,
                "title": doc["title"],
                "category": doc["category"],
                "last_updated": doc["last_updated"],
            }
            for name, doc in self.store.items()
        ]

    def get_by_category(self, category: str) -> List[Dict]:
        """Get documents by category"""
        return [
            {"name": name, "title": doc["title"]}
            for name, doc in self.store.items()
            if doc["category"] == category
        ]

    def summary(self) -> Dict:
        """Return document store summary"""
        return {
            "total_docs": len(self.store),
            "categories": list(set(d["category"] for d in self.store.values())),
        }