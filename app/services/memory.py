import os
import json
from datetime import datetime

class MemoryStore:
    def __init__(self):
        self.memory_dir = "memory"
        os.makedirs(self.memory_dir, exist_ok=True)

    def _path(self, session_id: str):
        return os.path.join(self.memory_dir, f"{session_id}.json")

    def get_memory(self, session_id: str):
        path = self._path(session_id)
        if not os.path.exists(path):
            return {"summary": "", "turns": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_memory(self, session_id: str, user_msg: str, agent_msg: str):
        memory = self.get_memory(session_id)
        memory["turns"].append({
            "time": datetime.utcnow().isoformat(),
            "user": user_msg,
            "agent": agent_msg,
        })
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def update_summary(self, session_id: str, summary: str):
        memory = self.get_memory(session_id)
        memory["summary"] = summary
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def recent_turns(self, session_id: str, limit: int = 6):
        memory = self.get_memory(session_id)
        return memory.get("turns", [])[-limit:]
