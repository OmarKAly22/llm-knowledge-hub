# short_term_memory.py
from typing import Dict, Any, List
from collections import OrderedDict
import time


class ShortTermMemory:
    """
    A lightweight in-memory buffer for recent conversation history.
    This simulates working memory — storing only the most recent and relevant exchanges.
    """

    def __init__(self, capacity: int = 20):
        """
        Initialize the short-term memory buffer.

        Args:
            capacity: Maximum number of memory entries to keep
        """
        self.capacity = capacity
        self.memories: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def store(self, key: str, value: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Store a memory entry in STM.

        Args:
            key: Unique identifier (e.g., 'interaction_1')
            value: Dictionary with content (role, text, importance, etc.)
            metadata: Optional extra info (timestamp, session_id, etc.)
        """
        entry = {
            "timestamp": time.time(),
            "content": value.get("content"),
            "role": value.get("role", "user"),
            "importance": value.get("importance", 0.5),
            "metadata": metadata or {},
        }

        # Add or update memory
        if key in self.memories:
            self.memories.move_to_end(key)
        self.memories[key] = entry

        # Enforce capacity (remove oldest if full)
        if len(self.memories) > self.capacity:
            self.memories.popitem(last=False)

    def retrieve(self, last_n: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent N memory entries.

        Args:
            last_n: Number of recent entries to return
        """
        return list(self.memories.values())[-last_n:]

    def clear(self):
        """Completely clear short-term memory."""
        self.memories.clear()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a brief summary of STM state (for debugging or stats).
        """
        return {
            "capacity": self.capacity,
            "current_size": len(self.memories),
            "oldest_entry_age": (
                time.time() - next(iter(self.memories.values()))["timestamp"]
                if self.memories else None
            ),
            "most_recent_entry": list(self.memories.values())[-1]
            if self.memories else None
        }
    
# -------------------------------
# Example usage and demonstration
# -------------------------------
if __name__ == "__main__":
    stm = ShortTermMemory(capacity=3)

    # Store some example memories
    stm.store("interaction_1", {"content": "Hello!", "role": "user"})
    time.sleep(0.5)
    stm.store("interaction_2", {"content": "Hi there, how can I help?", "role": "assistant"})
    time.sleep(0.5)
    stm.store("interaction_3", {"content": "What is the weather today?", "role": "user"})
    time.sleep(0.5)
    stm.store("interaction_4", {"content": "It's sunny and warm.", "role": "assistant"})

    # Retrieve recent memories
    print("\n Last 3 memories:")
    for m in stm.retrieve():
        print(m)

    # Show summary
    print("\nMemory summary:")
    print(stm.get_summary())

    # Clear memory
    stm.clear()
    print("\n After clearing memory:")
    print(stm.get_summary())
