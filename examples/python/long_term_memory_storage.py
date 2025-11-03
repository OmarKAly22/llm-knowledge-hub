import numpy as np
from typing import List, Dict, Optional
import chromadb
import openai
import uuid
import json
from datetime import datetime

class LongTermMemory:
    """
    Persistent memory using vector database for semantic search.
    This is the AI's "hard drive" - slower but permanent.
    """
    
    def __init__(self, 
                 collection_name: str = "long_term_memory",
                 persist_directory: str = "./memory_db"):
        """
        Initialize ChromaDB for vector storage.
        
        Args:
            collection_name: Name of the memory collection
            persist_directory: Where to store the database
        """
        # Initialize ChromaDB client with new API
        # Use PersistentClient for disk persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Long-term memory storage"}
        )
        
        # Track some metrics
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "last_consolidation": None
        }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        In production, cache embeddings to reduce API calls.
        """
        try:
            # Note: This requires OpenAI API key to be set
            # export OPENAI_API_KEY='your-key-here'
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback: simple hash-based pseudo-embedding
            # (Only for testing - not suitable for production)
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(1536).tolist()
    
    def store(self, 
              content: str,
              metadata: Dict = None,
              memory_id: str = None) -> str:
        """
        Store content in long-term memory with semantic indexing.
        
        Args:
            content: The content to remember
            metadata: Additional metadata (tags, source, etc.)
            memory_id: Optional ID (auto-generated if not provided)
        
        Returns:
            The ID of the stored memory
        """
        # Generate unique ID if not provided
        if memory_id is None:
            memory_id = str(uuid.uuid4())
        
        # Generate embedding for semantic search
        embedding = self._generate_embedding(content)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "length": len(content),
            "type": metadata.get("type", "general")
        })
        
        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        self.stats["total_stored"] += 1
        
        return memory_id
    
    def retrieve(self,
                 query: str,
                 n_results: int = 5,
                 min_similarity: float = 0.7,
                 filters: Dict = None) -> List[Dict]:
        """
        Retrieve relevant memories using semantic search.
        
        Args:
            query: The search query
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            filters: Metadata filters to apply
        
        Returns:
            List of relevant memories with similarity scores
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Perform semantic search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        # Format results
        memories = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Calculate similarity (1 - distance)
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - (distance / 2)  # Normalize to 0-1
                
                if similarity >= min_similarity:
                    memories.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "similarity": similarity
                    })
        
        self.stats["total_retrieved"] += len(memories)
        
        return memories
    
    def update(self, 
               memory_id: str,
               content: str = None,
               metadata_updates: Dict = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (if updating)
            metadata_updates: Metadata fields to update
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing memory
            existing = self.collection.get(ids=[memory_id])
            
            if not existing['ids']:
                return False
            
            # Update content if provided
            if content:
                embedding = self._generate_embedding(content)
                self.collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content]
                )
            
            # Update metadata if provided
            if metadata_updates:
                current_metadata = existing['metadatas'][0] if existing['metadatas'] else {}
                current_metadata.update(metadata_updates)
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[current_metadata]
                )
            
            return True
            
        except Exception as e:
            print(f"Error updating memory: {e}")
            return False
    
    def forget(self, memory_id: str = None, filters: Dict = None) -> int:
        """
        Delete memories by ID or metadata filters.
        
        Args:
            memory_id: Specific memory to delete
            filters: Delete all memories matching filters
        
        Returns:
            Number of memories deleted
        """
        if memory_id:
            self.collection.delete(ids=[memory_id])
            return 1
        elif filters:
            # Get memories matching filters
            results = self.collection.get(where=filters)
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
        
        return 0
    
    def consolidate(self, similarity_threshold: float = 0.95) -> int:
        """
        Merge highly similar memories to save space.
        This prevents memory bloat from duplicate information.
        
        Args:
            similarity_threshold: How similar memories must be to merge
        
        Returns:
            Number of memories consolidated
        """
        # Get all memories
        all_memories = self.collection.get()
        
        if not all_memories['ids']:
            return 0
        
        consolidated = 0
        processed = set()
        
        # Check each memory for duplicates
        for i, doc_id in enumerate(all_memories['ids']):
            if doc_id in processed:
                continue
            
            # Find similar memories
            similar = self.retrieve(
                all_memories['documents'][i],
                n_results=10,
                min_similarity=similarity_threshold
            )
            
            if len(similar) > 1:
                # Keep the first, merge metadata, delete others
                main_memory = similar[0]
                
                for duplicate in similar[1:]:
                    if duplicate['id'] != main_memory['id']:
                        # Merge metadata
                        self.update(
                            main_memory['id'],
                            metadata_updates={
                                "consolidated_from": duplicate['id'],
                                "consolidation_date": datetime.now().isoformat()
                            }
                        )
                        
                        # Delete duplicate
                        self.forget(duplicate['id'])
                        processed.add(duplicate['id'])
                        consolidated += 1
        
        self.stats["last_consolidation"] = datetime.now().isoformat()
        
        return consolidated
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics."""
        all_memories = self.collection.get()
        count = len(all_memories['ids']) if all_memories['ids'] else 0
        
        return {
            "total_memories": count,
            "total_stored": self.stats["total_stored"],
            "total_retrieved": self.stats["total_retrieved"],
            "last_consolidation": self.stats["last_consolidation"],
            "database_size_estimate": f"{count * 2}KB"  # Rough estimate
        }


# Example usage
if __name__ == "__main__":
    # Initialize Long-Term Memory
    ltm = LongTermMemory()
    
    print("=== Long-Term Memory System Demo ===\n")
    
    # Store various types of information
    print("Storing memories...")
    
    ltm.store(
        "The user prefers Python for data science projects.",
        metadata={"type": "user_preference", "topic": "programming"}
    )
    
    ltm.store(
        "The user is working on a machine learning project for customer churn prediction.",
        metadata={"type": "project_context", "topic": "ml"}
    )
    
    ltm.store(
        "Important: The production database password was changed last week.",
        metadata={"type": "critical_info", "topic": "security"}
    )
    
    print("✓ Stored 3 memories\n")
    
    # Retrieve relevant memories
    print("Retrieving memories for query: 'What programming language should I use?'")
    memories = ltm.retrieve(
        "What programming language should I use?",
        n_results=3
    )
    
    for i, memory in enumerate(memories, 1):
        print(f"\nMemory {i}:")
        print(f"  Similarity: {memory['similarity']:.2f}")
        print(f"  Content: {memory['content']}")
        print(f"  Type: {memory['metadata'].get('type', 'N/A')}")
    
    # Search with filters
    print("\n\nSearching for security-related memories...")
    security_memories = ltm.retrieve(
        "password",
        filters={"type": "critical_info"}
    )
    
    if security_memories:
        print(f"Found {len(security_memories)} security memory(ies)")
        for mem in security_memories:
            print(f"  - {mem['content'][:50]}...")
    
    # Consolidate duplicate memories
    print("\n\nConsolidating duplicate memories...")
    duplicates_removed = ltm.consolidate(similarity_threshold=0.95)
    print(f"✓ Consolidated {duplicates_removed} duplicate memories")
    
    # Check statistics
    print("\n=== Memory Statistics ===")
    stats = ltm.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")