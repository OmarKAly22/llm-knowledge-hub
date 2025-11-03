import uuid
from typing import Dict, List

# Import your local modules
from context_strategies import ContextWindow, ContextStrategy
from token_manager import TokenManager

# Import the memory modules (if they are separate files)
from long_term_memory_storage import LongTermMemory
from short_term_memory import ShortTermMemory

class WorkingMemory:
    """
    The integration layer that combines all memory systems.
    This is where the magic happens - orchestrating STM, LTM, and context.
    """
    
    def __init__(self, 
                 max_context_tokens: int = 3500,
                 stm_capacity: int = 20,
                 ltm_collection: str = "working_memory"):
        """
        Initialize all memory components.
        
        Args:
            max_context_tokens: Maximum tokens for context window
            stm_capacity: Short-term memory capacity
            ltm_collection: Long-term memory collection name
        """
        # Initialize components
        self.context = ContextWindow(
            max_tokens=max_context_tokens,
            strategy=ContextStrategy.IMPORTANCE
        )
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(collection_name=ltm_collection)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.interaction_count = 0
        
        # Importance calculation weights
        self.importance_weights = {
            "length": 0.2,
            "keywords": 0.3,
            "user_signal": 0.3,
            "recency": 0.2
        }
    
    def process(self, user_input: str) -> Dict:
        """
        Process user input through all memory systems.
        
        This is the main entry point that:
        1. Updates context window
        2. Stores in STM
        3. Determines LTM storage
        4. Retrieves relevant memories
        5. Builds optimized context
        
        Args:
            user_input: The user's message
        
        Returns:
            Dictionary with context and memory information
        """
        self.interaction_count += 1
        
        # Calculate importance of this input
        importance = self._calculate_importance(user_input)
        
        # 1. Update context window (immediate context)
        self.context.add_message("user", user_input, importance)
        
        # 2. Store in short-term memory (session)
        stm_key = f"interaction_{self.interaction_count}"
        self.stm.store(
            stm_key,
            {
                "role": "user",
                "content": user_input,
                "importance": importance
            },
            metadata={"session_id": self.session_id}
        )
        
        # 3. Determine if should persist to long-term memory
        if self._should_persist_to_ltm(user_input, importance):
            self.ltm.store(
                user_input,
                metadata={
                    "session_id": self.session_id,
                    "interaction": self.interaction_count,
                    "importance": importance,
                    "type": "user_input"
                }
            )
        
        # 4. Retrieve relevant memories from all sources
        relevant_memories = self._gather_relevant_memories(user_input)
        
        # 5. Build optimized context for LLM
        optimized_context = self._build_optimized_context(
            user_input,
            relevant_memories
        )
        
        return {
            "context": optimized_context,
            "memories_retrieved": len(relevant_memories),
            "importance_score": importance,
            "persisted_to_ltm": self._should_persist_to_ltm(user_input, importance),
            "session_stats": self.get_session_stats()
        }
    
    def _calculate_importance(self, text: str) -> float:
        """
        Calculate importance score for memory storage decisions.
        
        Factors considered:
        - Length of content
        - Presence of important keywords
        - User signals (e.g., "remember this")
        - Recency (newer = more important for STM)
        """
        scores = {}
        
        # Length factor (longer = more important, up to a point)
        scores["length"] = min(len(text) / 500, 1.0)
        
        # Keyword factor
        important_keywords = [
            "important", "remember", "critical", "key", 
            "decision", "password", "secret", "todo"
        ]
        keyword_count = sum(1 for kw in important_keywords if kw in text.lower())
        scores["keywords"] = min(keyword_count / 3, 1.0)
        
        # User signal factor
        user_signals = ["remember this", "important:", "note:", "todo:"]
        scores["user_signal"] = 1.0 if any(s in text.lower() for s in user_signals) else 0.0
        
        # Recency factor (always high for current interaction)
        scores["recency"] = 0.8
        
        # Calculate weighted average
        total_score = sum(
            scores[factor] * self.importance_weights[factor]
            for factor in scores
        )
        
        return min(total_score, 1.0)
    
    def _should_persist_to_ltm(self, content: str, importance: float) -> bool:
        """
        Determine if content should be saved to long-term memory.
        
        Criteria:
        - High importance score (>0.7)
        - Contains critical information
        - User explicitly asks to remember
        - Part of important conversation thread
        """
        # High importance
        if importance > 0.7:
            return True
        
        # Explicit remember request
        if any(phrase in content.lower() for phrase in ["remember", "don't forget", "save this"]):
            return True
        
        # Critical information patterns
        critical_patterns = ["password", "api key", "credential", "important"]
        if any(pattern in content.lower() for pattern in critical_patterns):
            return True
        
        # Default: don't persist everything to LTM
        return False
    
    def _gather_relevant_memories(self, query: str) -> List[Dict]:
        """
        Gather relevant memories from all memory systems.
        
        This performs parallel retrieval from:
        - Recent STM entries
        - Semantically similar LTM entries
        - Current context window
        """
        memories = []
        
        # Get from long-term memory (semantic search)
        ltm_memories = self.ltm.retrieve(query, n_results=3, min_similarity=0.6)
        for mem in ltm_memories:
            memories.append({
                "source": "ltm",
                "content": mem["content"],
                "similarity": mem["similarity"],
                "metadata": mem["metadata"]
            })
        
        # Get from short-term memory (recent interactions)
        stm_recent = self.stm.retrieve(last_n=3)
        for mem in stm_recent:
            if isinstance(mem, dict) and "content" in mem:
                memories.append({
                    "source": "stm",
                    "content": mem["content"],
                    "similarity": 0.8,  # Recent = high relevance
                    "metadata": {}
                })
        
        # Sort by relevance (similarity score)
        memories.sort(key=lambda x: x["similarity"], reverse=True)
        
        return memories[:5]  # Top 5 most relevant
    
    def _build_optimized_context(self, 
                                  current_input: str,
                                  memories: List[Dict]) -> List[Dict]:
        """
        Build optimized context for the LLM.
        
        This carefully constructs the context to:
        - Stay within token limits
        - Include most relevant information
        - Maintain conversation flow
        - Inject historical context appropriately
        """
        messages = []
        
        # System message with memory context
        system_content = "You are a helpful AI assistant with memory capabilities."
        
        if memories:
            system_content += "\n\nRelevant context from memory:"
            for i, mem in enumerate(memories[:3], 1):
                source = mem["source"].upper()
                content_preview = mem["content"][:150]
                system_content += f"\n{i}. [{source}] {content_preview}..."
        
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation context
        context_messages = self.context.get_messages()
        
        # Skip the system message from context (we already have one)
        for msg in context_messages:
            if msg["role"] != "system":
                messages.append(msg)
        
        # Ensure we're within token limits
        token_count = self.context.token_manager.count_messages(messages)
        max_tokens = self.context.max_tokens
        
        # If over limit, remove middle messages
        while token_count > max_tokens and len(messages) > 3:
            # Remove a message from the middle
            mid_index = len(messages) // 2
            del messages[mid_index]
            token_count = self.context.token_manager.count_messages(messages)
        
        return messages
    
    def store_response(self, response: str):
        """
        Store the assistant's response in memory systems.
        
        Args:
            response: The assistant's response to store
        """
        # Add to context window
        self.context.add_message("assistant", response, importance=0.6)
        
        # Store in STM
        stm_key = f"response_{self.interaction_count}"
        self.stm.store(
            stm_key,
            {
                "role": "assistant",
                "content": response,
                "importance": 0.6
            },
            metadata={"session_id": self.session_id}
        )
        
        # Consider storing important responses in LTM
        if self._should_persist_to_ltm(response, 0.6):
            self.ltm.store(
                response,
                metadata={
                    "session_id": self.session_id,
                    "interaction": self.interaction_count,
                    "type": "assistant_response"
                }
            )
    
    def get_session_stats(self) -> Dict:
        """Get statistics about the current session."""
        return {
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "context_tokens": self.context._get_total_tokens(),
            "stm_memories": len(self.stm.memories),
            "ltm_stats": self.ltm.get_statistics()
        }
    
    def end_session(self):
        """
        Clean up session and consolidate memories.
        Call this when a conversation ends.
        """
        # Optionally consolidate LTM to merge duplicates
        consolidated = self.ltm.consolidate(similarity_threshold=0.9)
        
        # Clear STM (session memories)
        self.stm = ShortTermMemory(capacity=self.stm.capacity)
        
        # Reset context but keep system message
        system_msg = self.context.system_message
        self.context = ContextWindow(
            max_tokens=self.context.max_tokens,
            strategy=self.context.strategy
        )
        if system_msg:
            self.context.system_message = system_msg
        
        # New session ID
        self.session_id = str(uuid.uuid4())
        self.interaction_count = 0
        
        return {
            "memories_consolidated": consolidated,
            "new_session_id": self.session_id
        }

# Example: Complete conversation with memory
def example_conversation():
    """Demonstrate a full conversation with memory management."""
    
    # Initialize working memory
    memory = WorkingMemory()
    
    print("=== AI Assistant with Memory ===\n")
    
    # Simulate a conversation
    conversations = [
        "Hi! My name is Alex and I'm learning Python.",
        "I'm particularly interested in machine learning and data science.",
        "Can you remember that I prefer practical examples over theory?",
        "What's a good library for data visualization?",
        "Actually, I changed my mind. I want to focus on web development instead.",
        "What Python framework would you recommend for web development?",
    ]
    
    for user_input in conversations:
        print(f"User: {user_input}")
        
        # Process through memory system
        result = memory.process(user_input)
        
        print(f"  [Importance: {result['importance_score']:.2f}]")
        print(f"  [Memories retrieved: {result['memories_retrieved']}]")
        print(f"  [Persisted to LTM: {result['persisted_to_ltm']}]")
        
        # Simulate assistant response
        response = f"I understand. Based on your interests..."
        memory.store_response(response)
        
        print(f"Assistant: {response}\n")
    
    # Check final statistics
    stats = memory.get_session_stats()
    print("\n=== Session Statistics ===")
    print(f"Total interactions: {stats['interaction_count']}")
    print(f"Context tokens used: {stats['context_tokens']}")
    print(f"STM memories: {stats['stm_memories']}")
    print(f"LTM total memories: {stats['ltm_stats']['total_memories']}")
    
    # End session
    end_stats = memory.end_session()
    print(f"\n=== Session Ended ===")
    print(f"Memories consolidated: {end_stats['memories_consolidated']}")
    print(f"New session ID: {end_stats['new_session_id']}")
# Run the example
if __name__ == "__main__":
    example_conversation()