"""
ARIA - Autonomous Research & Iteration Agent
Email Tool
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import List, Dict, Optional


class EmailTool:
    """
    Simulates an enterprise email client.
    Agent can read, send, and list emails.
    """

    def __init__(self):
        self.inbox: List[Dict] = [
            {
                "id": 1,
                "from": "manager@corp.com",
                "subject": "Q3 Report Due Friday",
                "body": "Please prepare Q3 report and schedule review meeting with the team.",
                "priority": "high",
                "read": False,
            },
            {
                "id": 2,
                "from": "hr@corp.com",
                "subject": "Policy Update — Expense Limit Changed",
                "body": "New expense policy effective immediately. Max limit is $500. Receipts mandatory.",
                "priority": "high",
                "read": False,
            },
            {
                "id": 3,
                "from": "client@external.com",
                "subject": "Meeting Request — Project Discussion",
                "body": "Can we meet Thursday 3pm to discuss the project timeline?",
                "priority": "medium",
                "read": False,
            },
            {
                "id": 4,
                "from": "finance@corp.com",
                "subject": "Q3 Revenue Numbers",
                "body": "Q3 Revenue: $150,000. Expenses: $80,000. Net Profit: $70,000. YoY Growth: 12.5%",
                "priority": "high",
                "read": False,
            },
            {
                "id": 5,
                "from": "team@corp.com",
                "subject": "Standup Notes",
                "body": "Team standup notes for this week. Action items assigned.",
                "priority": "low",
                "read": False,
            },
        ]
        self.sent: List[Dict] = []

    def list_inbox(self) -> List[Dict]:
        """List all emails in inbox"""
        return [
            {
                "id": e["id"],
                "from": e["from"],
                "subject": e["subject"],
                "priority": e["priority"],
                "read": e["read"],
            }
            for e in self.inbox
        ]

    def read(self, email_id: int) -> Dict:
        """Read a specific email by ID"""
        for email in self.inbox:
            if email["id"] == email_id:
                email["read"] = True
                return {
                    "status": "success",
                    "content": email,
                }
        return {"error": f"Email {email_id} not found"}

    def send(self, to: str, subject: str, body: str) -> Dict:
        """Send an email"""
        if not to or not subject or not body:
            return {"error": "Missing required fields: to, subject, body"}

        email = {
            "id": len(self.sent) + 100,
            "to": to,
            "subject": subject,
            "body": body,
        }
        self.sent.append(email)
        return {
            "status": "sent",
            "message": f"Email sent to {to}",
            "email": email,
        }

    def get_unread(self) -> List[Dict]:
        """Get all unread emails"""
        return [e for e in self.inbox if not e["read"]]

    def get_by_priority(self, priority: str) -> List[Dict]:
        """Get emails by priority level"""
        return [e for e in self.inbox if e["priority"] == priority]

    def summary(self) -> Dict:
        """Return inbox summary"""
        return {
            "total": len(self.inbox),
            "unread": len(self.get_unread()),
            "sent": len(self.sent),
            "high_priority": len(self.get_by_priority("high")),
        }