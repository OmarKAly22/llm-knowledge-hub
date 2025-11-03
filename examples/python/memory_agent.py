import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory

class MemoryAgent:
    """
    Production-ready conversational agent with comprehensive memory management.
    This combines all memory systems into a cohesive assistant.
    """
    
    def __init__(self, 
                 model: str = "gpt-4",
                 memory_config: Optional[Dict] = None):
        """
        Initialize the memory-enabled agent.
        
        Args:
            model: LLM model to use
            memory_config: Configuration for memory systems
        """
        self.model = model
        
        # Default configuration
        default_config = {
            "max_context_tokens": 3500,
            "stm_capacity": 20,
            "ltm_persist_threshold": 0.7,
            "episode_size": 10,
            "consolidation_interval": 100  # Consolidate every N interactions
        }
        
        self.config = {**default_config, **(memory_config or {})}
        
        # Initialize memory systems
        self.working_memory = WorkingMemory(
            max_context_tokens=self.config["max_context_tokens"],
            stm_capacity=self.config["stm_capacity"]
        )
        
        self.episodic_memory = EpisodicMemory(
            episode_size=self.config["episode_size"]
        )
        
        # Metrics
        self.metrics = {
            "total_interactions": 0,
            "total_tokens_processed": 0,
            "memories_created": 0,
            "episodes_created": 0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    async def process_message(self, user_input: str) -> Dict:
        """
        Process a user message with full memory integration.
        
        This is the main entry point that:
        1. Processes through working memory
        2. Updates episodic memory
        3. Generates response
        4. Updates all memory systems
        5. Handles consolidation
        
        Args:
            user_input: The user's message
        
        Returns:
            Response and metadata
        """
        self.metrics["total_interactions"] += 1
        
        # 1. Process through working memory
        memory_result = self.working_memory.process(user_input)
        
        # 2. Get relevant episodic context
        episodic_context = await self._get_episodic_context_async(user_input)
        
        # 3. Build final context
        final_context = self._build_final_context(
            memory_result["context"],
            episodic_context
        )
        
        # 4. Generate response
        response = await self._generate_response(final_context)
        
        # 5. Update memory systems
        self.working_memory.store_response(response)
        self.episodic_memory.add_interaction(user_input, response)
        
        # 6. Update metrics
        tokens_used = self.working_memory.context.token_manager.count_tokens(
            user_input + response
        )
        self.metrics["total_tokens_processed"] += tokens_used
        
        # 7. Periodic consolidation
        if self.metrics["total_interactions"] % self.config["consolidation_interval"] == 0:
            await self._consolidate_memories_async()
        
        return {
            "response": response,
            "metadata": {
                "tokens_used": tokens_used,
                "memories_retrieved": memory_result["memories_retrieved"],
                "importance_score": memory_result["importance_score"],
                "persisted": memory_result["persisted_to_ltm"],
                "episode_count": len(self.episodic_memory.episodes),
                "session_stats": memory_result["session_stats"]
            }
        }
    
    async def _get_episodic_context_async(self, query: str) -> List[Dict]:
        """Asynchronously retrieve relevant episodic memories."""
        loop = asyncio.get_event_loop()
        
        # Run episodic search in thread pool
        episodes = await loop.run_in_executor(
            self.executor,
            self.episodic_memory.get_relevant_episodes,
            query,
            3  # max episodes
        )
        
        return episodes
    
    def _build_final_context(self, 
                             working_context: List[Dict],
                             episodic_context: List[Dict]) -> List[Dict]:
        """
        Combine working memory context with episodic memories.
        
        Args:
            working_context: Context from working memory
            episodic_context: Relevant episodes
        
        Returns:
            Optimized final context for LLM
        """
        # Start with working context
        final_context = working_context.copy()
        
        # Add episodic context to system message if relevant episodes found
        if episodic_context and len(final_context) > 0:
            system_msg = final_context[0]
            
            if system_msg["role"] == "system":
                episode_summary = "\n\nRelevant conversation episodes:"
                for ep in episodic_context[:2]:  # Top 2 episodes
                    episode_summary += f"\n- {ep['summary']}"
                
                system_msg["content"] += episode_summary
        
        return final_context
    
    async def _generate_response(self, context: List[Dict]) -> str:
        """
        Generate response using the LLM.
        
        Args:
            context: Optimized context for the LLM
        
        Returns:
            Generated response
        """
        try:
            # For demonstration - replace with actual LLM call
            # response = await openai.ChatCompletion.acreate(
            #     model=self.model,
            #     messages=context,
            #     temperature=0.7,
            #     max_tokens=500
            # )
            # return response.choices[0].message.content
            
            # Simulated response for demo
            return "I understand your request and have access to our conversation history."
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    async def _consolidate_memories_async(self):
        """Periodically consolidate memories to prevent bloat."""
        loop = asyncio.get_event_loop()
        
        # Run consolidation in background
        consolidation_task = loop.run_in_executor(
            self.executor,
            self.working_memory.ltm.consolidate,
            0.95  # similarity threshold
        )
        
        # Don't wait for completion (run in background)
        asyncio.create_task(self._log_consolidation(consolidation_task))
    
    async def _log_consolidation(self, task):
        """Log consolidation results."""
        try:
            result = await task
            self.logger.info(f"Consolidated {result} duplicate memories")
        except Exception as e:
            self.logger.error(f"Error during consolidation: {e}")
    
    def get_conversation_summary(self) -> str:
        """
        Generate a summary of the entire conversation.
        Useful for handoffs or session endings.
        """
        summary_parts = []
        
        # Add episode summaries
        if self.episodic_memory.episodes:
            summary_parts.append("Conversation episodes:")
            for ep in self.episodic_memory.episodes:
                summary_parts.append(f"- {ep['summary']}")
        
        # Add current episode if exists
        if self.episodic_memory.current_episode:
            summary_parts.append(
                f"Current episode: {len(self.episodic_memory.current_episode)} interactions"
            )
        
        # Add key metrics
        summary_parts.append(f"\nTotal interactions: {self.metrics['total_interactions']}")
        summary_parts.append(f"Total tokens: {self.metrics['total_tokens_processed']}")
        
        return "\n".join(summary_parts)
    
    def export_memory_state(self, filepath: str):
        """
        Export complete memory state for backup or analysis.
        
        Args:
            filepath: Where to save the memory export
        """
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "episodes": [
                {
                    "id": ep["id"],
                    "summary": ep["summary"],
                    "interaction_count": ep["interaction_count"]
                }
                for ep in self.episodic_memory.episodes
            ],
            "working_memory_stats": self.working_memory.get_session_stats(),
            "configuration": self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Memory state exported to {filepath}")
    
    def reset_session(self):
        """Reset the session while preserving long-term memories."""
        # End working memory session
        end_result = self.working_memory.end_session()
        
        # Close current episode
        self.episodic_memory._close_current_episode()
        
        # Update metrics
        self.metrics["episodes_created"] = len(self.episodic_memory.episodes)
        
        self.logger.info(f"Session reset. New session ID: {end_result['new_session_id']}")
        
        return end_result
# Example usage
async def main():
    """Demonstrate the complete memory-enabled agent."""
    
    # Initialize agent
    agent = MemoryAgent(
        model="gpt-4",
        memory_config={
            "max_context_tokens": 3000,
            "stm_capacity": 15,
            "episode_size": 5
        }
    )
    
    print("=== Memory-Enabled AI Assistant ===\n")
    
    # Simulate conversation
    messages = [
        "Hi! I'm working on a Python project for data analysis.",
        "I need to process CSV files and create visualizations.",
        "Can you remember that I prefer matplotlib over seaborn?",
        "What's the best way to handle missing data in pandas?",
        "Let's switch topics. I want to learn about web scraping now.",
        "What libraries would you recommend for web scraping?",
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        
        # Process message
        result = await agent.process_message(msg)
        
        print(f"Assistant: {result['response']}")
        print(f"  [Tokens: {result['metadata']['tokens_used']}]")
        print(f"  [Importance: {result['metadata']['importance_score']:.2f}]")
        print(f"  [Memories retrieved: {result['metadata']['memories_retrieved']}]")
        print()
    
    # Get conversation summary
    summary = agent.get_conversation_summary()
    print("\n=== Conversation Summary ===")
    print(summary)
    
    # Export memory state
    agent.export_memory_state("memory_export.json")
    
    # Reset for new session
    agent.reset_session()
    print("\n=== Session Reset - Ready for New Conversation ===")
# Run the example
if __name__ == "__main__":
    asyncio.run(main())