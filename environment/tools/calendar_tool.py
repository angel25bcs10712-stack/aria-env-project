"""
ARIA - Autonomous Research & Iteration Agent
Calendar Tool
Author: Angel Singh
Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler 2026
"""

from typing import Dict, List


class CalendarTool:
    """
    Simulates an enterprise calendar system.
    Agent can check, schedule, and reschedule meetings.
    """

    def __init__(self):
        self.events: Dict[str, str] = {
            "Monday 2pm": "Team standup",
            "Thursday 3pm": "Client call",
            "Friday 10am": "Q3 Review",
        }

    def check(self, slot: str) -> Dict:
        """Check if a time slot is free or busy"""
        if not slot:
            return {"error": "Slot is required"}
        if slot in self.events:
            return {
                "status": "busy",
                "slot": slot,
                "event": self.events[slot],
            }
        return {
            "status": "free",
            "slot": slot,
        }

    def schedule(self, slot: str, event: str) -> Dict:
        """Schedule a new event"""
        if not slot or not event:
            return {"error": "Slot and event are required"}
        if slot in self.events:
            return {
                "status": "conflict",
                "slot": slot,
                "existing": self.events[slot],
            }
        self.events[slot] = event
        return {
            "status": "scheduled",
            "slot": slot,
            "event": event,
        }

    def reschedule(self, old_slot: str, new_slot: str) -> Dict:
        """Reschedule an existing event"""
        if not old_slot or not new_slot:
            return {"error": "Old and new slots are required"}
        if old_slot not in self.events:
            return {"error": f"No event found at {old_slot}"}
        if new_slot in self.events:
            return {
                "status": "conflict",
                "slot": new_slot,
                "existing": self.events[new_slot],
            }
        event = self.events.pop(old_slot)
        self.events[new_slot] = event
        return {
            "status": "rescheduled",
            "event": event,
            "from": old_slot,
            "to": new_slot,
        }

    def list_events(self) -> List[Dict]:
        """List all scheduled events"""
        return [
            {"slot": slot, "event": event}
            for slot, event in self.events.items()
        ]

    def summary(self) -> Dict:
        """Return calendar summary"""
        return {
            "total_events": len(self.events),
            "events": self.list_events(),
        }