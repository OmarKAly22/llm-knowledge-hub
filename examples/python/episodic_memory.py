from typing import List, Dict
from datetime import datetime, timedelta
import uuid
import time

class EpisodicMemory:
    """
    Organizes memories into episodes for better narrative understanding.
    Think of this as creating "chapters" in a conversation.
    """
    
    def __init__(self, episode_size: int = 10, time_gap_minutes: int = 30):
        """
        Args:
            episode_size: Max interactions per episode
            time_gap_minutes: Time gap that triggers new episode
        """
        self.episode_size = episode_size
        self.time_gap = timedelta(minutes=time_gap_minutes)
        
        self.current_episode = []
        self.episodes = []
        self.episode_summaries = {}
    
    def add_interaction(self, user_input: str, assistant_response: str):
        """Add an interaction to the current episode."""
        interaction = {
            "timestamp": datetime.now(),
            "user": user_input,
            "assistant": assistant_response,
            "tokens": len((user_input + assistant_response).split())
        }
        
        # Check if should start new episode
        if self._should_start_new_episode(interaction):
            self._close_current_episode()
        
        self.current_episode.append(interaction)
    
    def _should_start_new_episode(self, interaction: Dict) -> bool:
        """
        Determine if a new episode should begin.
        
        Criteria:
        - Episode size reached
        - Large time gap
        - Topic change detected
        """
        if not self.current_episode:
            return False
        
        # Check size limit
        if len(self.current_episode) >= self.episode_size:
            return True
        
        # Check time gap
        last_time = self.current_episode[-1]["timestamp"]
        if interaction["timestamp"] - last_time > self.time_gap:
            return True
        
        # Check for explicit topic change signals
        topic_change_signals = [
            "let's talk about something else",
            "changing topic",
            "different question",
            "new subject",
            "moving on"
        ]
        
        user_input_lower = interaction["user"].lower()
        if any(signal in user_input_lower for signal in topic_change_signals):
            return True
        
        return False
    
    def _close_current_episode(self):
        """Close current episode and create summary."""
        if not self.current_episode:
            return
        
        episode_id = str(uuid.uuid4())
        
        # Create episode summary
        summary = self._generate_summary(self.current_episode)
        
        episode = {
            "id": episode_id,
            "start_time": self.current_episode[0]["timestamp"],
            "end_time": self.current_episode[-1]["timestamp"],
            "interaction_count": len(self.current_episode),
            "total_tokens": sum(i["tokens"] for i in self.current_episode),
            "summary": summary,
            "interactions": self.current_episode
        }
        
        self.episodes.append(episode)
        self.episode_summaries[episode_id] = summary
        self.current_episode = []
    
    def _generate_summary(self, interactions: List[Dict]) -> str:
        """
        Generate a summary of the episode.
        In production, use an LLM for better summaries.
        """
        # Simple extractive summary for demo
        topics = self._extract_topics(interactions)
        
        summary = f"Episode with {len(interactions)} interactions. "
        summary += f"Topics discussed: {', '.join(topics)}. "
        
        # Extract key points (simplified)
        key_points = []
        for interaction in interactions:
            # Look for questions
            if "?" in interaction["user"]:
                key_points.append("Question asked")
            # Look for decisions/conclusions
            if any(word in interaction["assistant"].lower() 
                   for word in ["recommend", "suggest", "should"]):
                key_points.append("Recommendation provided")
        
        if key_points:
            summary += f"Key points: {', '.join(set(key_points))}"
        
        return summary
    
    def _extract_topics(self, interactions: List[Dict]) -> List[str]:
        """Extract main topics from episode."""
        # Simplified topic extraction
        # In production, use NLP techniques like LDA or embeddings
        
        all_text = " ".join([
            i["user"] + " " + i["assistant"] 
            for i in interactions
        ])
        
        # Simple keyword-based topic detection
        topic_keywords = {
            "programming": ["python", "code", "function", "variable"],
            "machine learning": ["ml", "model", "training", "data"],
            "web development": ["website", "html", "css", "frontend"],
            "database": ["sql", "query", "table", "database"],
            "general": []
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text.lower() for kw in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ["general discussion"]
    
    def get_relevant_episodes(self, query: str, max_episodes: int = 3) -> List[Dict]:
        """
        Retrieve episodes relevant to a query.
        
        Args:
            query: Search query
            max_episodes: Maximum episodes to return
        
        Returns:
            List of relevant episodes with summaries
        """
        if not self.episodes:
            return []
        
        # Score episodes by relevance
        scored_episodes = []
        query_lower = query.lower()
        
        for episode in self.episodes:
            score = 0
            
            # Check summary relevance
            if query_lower in episode["summary"].lower():
                score += 2
            
            # Check interaction content
            for interaction in episode["interactions"]:
                if query_lower in interaction["user"].lower():
                    score += 1
                if query_lower in interaction["assistant"].lower():
                    score += 0.5
            
            if score > 0:
                scored_episodes.append((score, episode))
        
        # Sort by score and return top episodes
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        return [
            {
                "id": ep["id"],
                "summary": ep["summary"],
                "relevance_score": score,
                "interaction_count": ep["interaction_count"]
            }
            for score, ep in scored_episodes[:max_episodes]
        ]
    
    def get_episode_context(self, episode_id: str) -> List[Dict]:
        """Get full context of a specific episode."""
        for episode in self.episodes:
            if episode["id"] == episode_id:
                return episode["interactions"]
        return []
    
    def get_statistics(self) -> Dict:
        """Get episodic memory statistics."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "current_episode_size": len(self.current_episode)
            }
        
        total_interactions = sum(ep["interaction_count"] for ep in self.episodes)
        total_interactions += len(self.current_episode)
        
        avg_episode_size = total_interactions / (len(self.episodes) + (1 if self.current_episode else 0))
        
        return {
            "total_episodes": len(self.episodes),
            "current_episode_size": len(self.current_episode),
            "total_interactions": total_interactions,
            "average_episode_size": avg_episode_size,
            "total_tokens": sum(ep["total_tokens"] for ep in self.episodes)
        }

# Example usage of EpisodicMemory
if __name__ == "__main__":
    memory = EpisodicMemory(episode_size=3, time_gap_minutes=0.01)

    print("\n--- Adding interactions ---")
    memory.add_interaction("How do I write a Python function?", "You can define it using the def keyword.")
    time.sleep(1)
    memory.add_interaction("Can you show me an example of a for loop?", "Sure, here's a simple example using range().")
    time.sleep(1)
    memory.add_interaction("Thanks, now let's talk about databases.", "Okay, SQL databases use structured tables to store data.")
    memory.add_interaction("What is an SQL query?", "It is a command used to retrieve or modify data in a database.")
    memory.add_interaction("Different question: how do I train a model?", "You can train it using your dataset with scikit-learn or TensorFlow.")

    print("\n--- Episodes Created ---")
    for i, ep in enumerate(memory.episodes, 1):
        print(f"Episode {i}: {ep['summary']}")

    print("\n--- Relevant Episodes for 'database' ---")
    relevant = memory.get_relevant_episodes("database")
    for r in relevant:
        print(r)

    if relevant:
        episode_id = relevant[0]["id"]
        print(f"\n--- Context of Episode ID {episode_id} ---")
        for ctx in memory.get_episode_context(episode_id):
            print(f"User: {ctx['user']}")
            print(f"Assistant: {ctx['assistant']}\n")

    print("\n--- Statistics ---")
    print(memory.get_statistics())