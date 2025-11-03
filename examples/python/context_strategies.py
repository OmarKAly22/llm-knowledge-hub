from collections import deque
from enum import Enum
from typing import List, Dict
import time
from token_manager import TokenManager
class ContextStrategy(Enum):
    FIFO = "fifo"
    SLIDING = "sliding"
    IMPORTANCE = "importance"
    SUMMARY = "summary"
class ContextWindow:
    """
    Manages conversation context with different strategies.
    This is the core of any memory-enabled LLM application.
    """
    
    def __init__(self, 
                 max_tokens: int = 3000,  # Leave room for response
                 strategy: ContextStrategy = ContextStrategy.SLIDING):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.messages = deque()
        self.importance_scores = {}
        self.token_manager = TokenManager()
        
        # Always preserve system message
        self.system_message = None
    
    def add_message(self, role: str, content: str, importance: float = 0.5):
        """
        Add a message to the context window.
        
        Args:
            role: 'system', 'user', or 'assistant'
            content: The message content
            importance: Score from 0 to 1 (1 = most important)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        if role == "system":
            self.system_message = message
        else:
            self.messages.append(message)
            self.importance_scores[id(message)] = importance
        
        # Apply strategy to maintain window size
        self._maintain_window()
    
    def _maintain_window(self):
        """Apply the selected strategy to manage context size."""
        while self._get_total_tokens() > self.max_tokens:
            if self.strategy == ContextStrategy.FIFO:
                self._remove_oldest()
            elif self.strategy == ContextStrategy.SLIDING:
                self._apply_sliding_window()
            elif self.strategy == ContextStrategy.IMPORTANCE:
                self._remove_least_important()
            elif self.strategy == ContextStrategy.SUMMARY:
                self._compress_old_messages()
    
    def _remove_oldest(self):
        """FIFO: Remove the oldest message."""
        if self.messages:
            self.messages.popleft()
    
    def _apply_sliding_window(self):
        """Keep only the most recent messages that fit."""
        # Calculate how many recent messages fit
        total_tokens = 0
        messages_to_keep = []
        
        # Always include system message tokens
        if self.system_message:
            total_tokens += self.token_manager.count_tokens(
                self.system_message["content"]
            )
        
        # Add messages from newest to oldest
        for message in reversed(self.messages):
            msg_tokens = self.token_manager.count_tokens(message["content"])
            if total_tokens + msg_tokens <= self.max_tokens:
                messages_to_keep.insert(0, message)
                total_tokens += msg_tokens
            else:
                break
        
        self.messages = deque(messages_to_keep)
    
    def _remove_least_important(self):
        """Remove the message with lowest importance score."""
        if not self.messages:
            return
        
        min_importance = float('inf')
        min_index = -1
        
        for i, message in enumerate(self.messages):
            msg_importance = self.importance_scores.get(id(message), 0.5)
            if msg_importance < min_importance:
                min_importance = msg_importance
                min_index = i
        
        if min_index >= 0:
            del self.messages[min_index]
    
    def _compress_old_messages(self):
        """
        Summarize older messages (placeholder - requires LLM call).
        In production, this would call GPT-3.5 to create summaries.
        """
        if len(self.messages) < 10:
            return
        
        # Take first 5 messages to summarize
        to_summarize = []
        for _ in range(min(5, len(self.messages))):
            to_summarize.append(self.messages.popleft())
        
        # Create summary (in production, call LLM here)
        summary_content = f"[Summary of {len(to_summarize)} messages]"
        summary_message = {
            "role": "system",
            "content": summary_content,
            "timestamp": time.time()
        }
        
        # Add summary at the beginning
        self.messages.appendleft(summary_message)
    
    def _get_total_tokens(self) -> int:
        """Calculate total tokens in current context."""
        all_messages = []
        if self.system_message:
            all_messages.append(self.system_message)
        all_messages.extend(self.messages)
        
        return self.token_manager.count_messages(all_messages)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages for sending to LLM."""
        result = []
        if self.system_message:
            result.append(self.system_message)
        result.extend(self.messages)
        return result
    
    def get_stats(self) -> Dict:
        """Get statistics about the context window."""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self._get_total_tokens(),
            "max_tokens": self.max_tokens,
            "utilization": f"{(self._get_total_tokens() / self.max_tokens) * 100:.1f}%",
            "strategy": self.strategy.value
        }

# Example: Using different strategies
contexts = {
    "fifo": ContextWindow(max_tokens=1000, strategy=ContextStrategy.FIFO),
    "sliding": ContextWindow(max_tokens=1000, strategy=ContextStrategy.SLIDING),
    "importance": ContextWindow(max_tokens=1000, strategy=ContextStrategy.IMPORTANCE)
}
# Add system message to all
for ctx in contexts.values():
    ctx.add_message("system", "You are a helpful assistant.")
# Simulate conversation
messages = [
    ("user", "What is machine learning?", 0.8),  # Important question
    ("assistant", "Machine learning is...", 0.8),
    ("user", "Can you give an example?", 0.5),   # Follow-up
    ("assistant", "Sure! An example is...", 0.5),
    ("user", "How does it differ from AI?", 0.9), # Important comparison
    ("assistant", "The key differences are...", 0.9),
]
for role, content, importance in messages:
    for ctx in contexts.values():
        ctx.add_message(role, content, importance)

# Compare strategies
for name, ctx in contexts.items():
    stats = ctx.get_stats()
    print(f"\n{name.upper()} Strategy:")
    print(f"  Messages kept: {stats['total_messages']}")
    print(f"  Token usage: {stats['total_tokens']}/{stats['max_tokens']} ({stats['utilization']})")