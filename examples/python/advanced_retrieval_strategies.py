from typing import List
import numpy as np
import networkx as nx
import asyncio
from vector_dbs import VectorDatabase, Document, SearchResult
from working_embbeding_system import EmbeddingEngine
from datetime import datetime


class GraphRAG:
    """
    Graph-based RAG for complex reasoning.
    Combines knowledge graphs with vector retrieval.
    """
    
    def __init__(self,
                 vector_db: VectorDatabase,
                 embedding_system: EmbeddingEngine):
        """
        Initialize GraphRAG.
        
        Args:
            vector_db: Vector database for similarity search
            embedding_system: Embedding system
        """
        self.vector_db = vector_db
        self.embedding_system = embedding_system
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
    
    async def build_graph(self, documents: List[Document]):
        """
        Build knowledge graph from documents.
        Extracts entities and relationships.
        """
        for doc in documents:
            # Extract entities (simplified - use NER in production)
            entities = self._extract_entities(doc.content)
            
            # Add nodes
            for entity in entities:
                if entity not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        entity,
                        documents=[doc.doc_id],
                        mentions=1
                    )
                    # Generate embedding for entity
                    emb = await self.embedding_system.embed_batch([entity])
                    self.entity_embeddings[entity] = emb[0]
                else:
                    # Update existing node
                    self.knowledge_graph.nodes[entity]['documents'].append(doc.doc_id)
                    self.knowledge_graph.nodes[entity]['mentions'] += 1
            
            # Add edges between co-occurring entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if self.knowledge_graph.has_edge(entity1, entity2):
                        self.knowledge_graph[entity1][entity2]['weight'] += 1
                    else:
                        self.knowledge_graph.add_edge(entity1, entity2, weight=1)
    
    async def graph_enhanced_retrieval(self,
                                      query: str,
                                      k: int = 10) -> List[SearchResult]:
        """
        Retrieve documents using graph-enhanced search.
        Combines vector search with graph traversal.
        """
        # Get initial vector search results
        query_embedding = await self.embedding_system.embed_batch([query])
        initial_results = await self.vector_db.search(query_embedding[0], k=k//2)
        
        # Extract entities from query
        query_entities = self._extract_entities(query)
        
        # Find related entities in graph
        related_entities = set()
        for entity in query_entities:
            if entity in self.knowledge_graph:
                # Get neighbors up to 2 hops away
                neighbors = nx.single_source_shortest_path_length(
                    self.knowledge_graph, 
                    entity, 
                    cutoff=2
                )
                related_entities.update(neighbors.keys())
        
        # Get documents associated with related entities
        related_docs = set()
        for entity in related_entities:
            if entity in self.knowledge_graph:
                docs = self.knowledge_graph.nodes[entity].get('documents', [])
                related_docs.update(docs)
        
        # Retrieve additional documents
        if related_docs:
            # Get embeddings for related documents
            additional_results = await self._retrieve_documents(list(related_docs))
            
            # Merge results
            all_results = self._merge_results(initial_results, additional_results)
            
            # Re-rank based on graph importance
            ranked_results = self._rank_by_graph_importance(all_results, query_entities)
            
            return ranked_results[:k]
        
        return initial_results
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        In production, use NER models.
        """
        # Simplified entity extraction
        import re
        
        # Extract capitalized words (proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        
        # Extract technical terms (simplified)
        technical_terms = re.findall(r'\b(?:AI|ML|RAG|LLM|API|GPU|CPU)\b', text)
        entities.extend(technical_terms)
        
        return list(set(entities))
    
    async def _retrieve_documents(self, doc_ids: List[str]) -> List[SearchResult]:
        """Retrieve specific documents by ID."""
        # This would fetch documents from the database
        # Simplified for demonstration
        results = []
        for doc_id in doc_ids:
            results.append(SearchResult(
                doc_id=doc_id,
                content="",  # Would fetch actual content
                score=0.5,  # Base score for graph-based retrieval
                metadata={}
            ))
        return results
    
    def _merge_results(self,
                      results1: List[SearchResult],
                      results2: List[SearchResult]) -> List[SearchResult]:
        """Merge two result sets, removing duplicates."""
        seen = set()
        merged = []
        
        for result in results1 + results2:
            if result.doc_id not in seen:
                seen.add(result.doc_id)
                merged.append(result)
        
        return merged
    
    def _rank_by_graph_importance(self,
                                 results: List[SearchResult],
                                 query_entities: List[str]) -> List[SearchResult]:
        """
        Rank results based on graph importance.
        Uses PageRank and entity relevance.
        """
        # Calculate PageRank for entities
        if self.knowledge_graph.number_of_nodes() > 0:
            pagerank = nx.pagerank(self.knowledge_graph)
        else:
            pagerank = {}
        
        # Score each result
        for result in results:
            # Get entities in document
            doc_entities = self._extract_entities(result.content) if result.content else []
            
            # Calculate entity overlap
            overlap = len(set(doc_entities) & set(query_entities))
            
            # Calculate graph importance
            graph_score = sum(pagerank.get(entity, 0) for entity in doc_entities)
            
            # Combine scores
            result.score = result.score * 0.5 + overlap * 0.3 + graph_score * 0.2
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results

class AdaptiveRetriever:
    """
    Adaptive retrieval that chooses strategy based on query type.
    Uses ML to select the best retrieval approach.
    """
    
    def __init__(self,
                 vector_db: VectorDatabase,
                 embedding_system: EmbeddingEngine):
        self.vector_db = vector_db
        self.embedding_system = embedding_system
        self.graph_rag = GraphRAG(vector_db, embedding_system)
        
        # Query classifiers
        self.query_patterns = {
            'factual': ['what is', 'define', 'who is', 'when did'],
            'analytical': ['compare', 'analyze', 'evaluate', 'assess'],
            'procedural': ['how to', 'steps to', 'process for', 'guide to'],
            'exploratory': ['related to', 'similar to', 'examples of', 'types of']
        }
    
    async def retrieve(self,
                      query: str,
                      k: int = 10,
                      auto_select: bool = True) -> List[SearchResult]:
        """
        Adaptively retrieve documents based on query type.
        
        Args:
            query: Search query
            k: Number of results
            auto_select: Automatically select retrieval strategy
        
        Returns:
            Retrieved results
        """
        if auto_select:
            strategy = self._select_strategy(query)
        else:
            strategy = 'hybrid'
        
        if strategy == 'graph':
            return await self.graph_rag.graph_enhanced_retrieval(query, k)
        elif strategy == 'dense':
            query_emb = await self.embedding_system.embed_batch([query])
            return await self.vector_db.search(query_emb[0], k)
        elif strategy == 'hybrid':
            query_emb = await self.embedding_system.embed_batch([query])
            return await self.vector_db.hybrid_search(query_emb[0], query, k)
        else:
            # Multi-strategy ensemble
            results = await self._ensemble_retrieval(query, k)
            return results
    
    def _select_strategy(self, query: str) -> str:
        """
        Select retrieval strategy based on query analysis.
        """
        query_lower = query.lower()
        
        # Check query patterns
        query_type = 'general'
        for pattern_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                query_type = pattern_type
                break
        
        # Map query type to strategy
        strategy_map = {
            'factual': 'dense',
            'analytical': 'graph',
            'procedural': 'hybrid',
            'exploratory': 'graph',
            'general': 'hybrid'
        }
        
        return strategy_map.get(query_type, 'hybrid')
    
    async def _ensemble_retrieval(self,
                                 query: str,
                                 k: int) -> List[SearchResult]:
        """
        Ensemble retrieval using multiple strategies.
        """
        # Get results from multiple strategies
        query_emb = await self.embedding_system.embed_batch([query])
        
        # Parallel retrieval
        tasks = [
            self.vector_db.search(query_emb[0], k),
            self.vector_db.hybrid_search(query_emb[0], query, k),
            self.graph_rag.graph_enhanced_retrieval(query, k)
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # Merge and re-rank using reciprocal rank fusion
        merged = self._reciprocal_rank_fusion(results_list)
        
        return merged[:k]
    
    def _reciprocal_rank_fusion(self,
                               results_list: List[List[SearchResult]],
                               k: int = 60) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion to combine multiple result lists.
        """
        doc_scores = {}
        doc_map = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                
                # RRF score
                score = 1 / (k + rank + 1)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                else:
                    doc_scores[doc_id] = score
                    doc_map[doc_id] = result
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final results
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_map[doc_id]
            result.score = score
            final_results.append(result)
        
        return final_results

# --- Mock embedding system ---
class MockEmbeddingSystem:
    """Mocks embedding system."""
    async def embed_batch(self, texts):
        # Return random embeddings
        return [np.random.randn(128) for _ in texts]

# --- Mock vector DB ---
class MockVectorDB:
    """Mocks vector database."""
    def __init__(self):
        self.docs = {}

    async def insert(self, documents, embeddings):
        for doc, emb in zip(documents, embeddings):
            self.docs[doc.doc_id] = (doc, emb)
        return [doc.doc_id for doc in documents]

    async def search(self, query_embedding, k=10, filters=None):
        results = []
        for doc_id, (doc, emb) in self.docs.items():
            score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
            results.append(SearchResult(doc_id=doc_id, content=doc.content, score=score, metadata=doc.metadata))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def hybrid_search(self, query_embedding, query_text, k=10, alpha=0.5):
        return await self.search(query_embedding, k)

# --- Main test ---
async def main():
    # Sample documents
    docs = [
        Document("doc1", "AI embeddings are amazing. LLMs and RAG are powerful.", {"type": "ai"}, datetime.now()),
        Document("doc2", "Vector databases store embeddings and support hybrid search.", {"type": "db"}, datetime.now()),
        Document("doc3", "Weaviate and Qdrant are popular vector databases.", {"type": "weaviate"}, datetime.now()),
    ]

    # Create embeddings
    embeddings = np.random.randn(len(docs), 128)

    # Initialize mock systems
    embedding_system = MockEmbeddingSystem()
    vector_db = MockVectorDB()

    # Insert documents
    inserted_ids = await vector_db.insert(docs, embeddings)
    print("Inserted document IDs:", inserted_ids)

    # Initialize GraphRAG and build graph
    graph_rag = GraphRAG(vector_db, embedding_system)
    await graph_rag.build_graph(docs)
    print("\nKnowledge graph nodes:", graph_rag.knowledge_graph.nodes())

    # Test graph-enhanced retrieval
    query = "How does RAG work with AI?"
    graph_results = await graph_rag.graph_enhanced_retrieval(query, k=3)
    print("\nGraph-enhanced retrieval results:")
    for r in graph_results:
        print(f"{r.doc_id}: {r.content} (score: {r.score:.3f})")

    # Initialize AdaptiveRetriever
    retriever = AdaptiveRetriever(vector_db, embedding_system)

    # Test adaptive retrieval
    adaptive_results = await retriever.retrieve(query, k=3)
    print("\nAdaptive retrieval results:")
    for r in adaptive_results:
        print(f"{r.doc_id}: {r.content} (score: {r.score:.3f})")

# Run main
if __name__ == "__main__":
    asyncio.run(main())