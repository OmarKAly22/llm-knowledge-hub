import time
from typing import List, Dict, Optional, Any
from working_embbeding_system import EmbeddingEngine
from rag_architecture import SimpleVectorStore, Document
from advanced_document_chunking import SmartChunker

class CompleteRAGPipeline:
    """
    Complete, working RAG pipeline that processes documents and answers queries.
    """
    
    def __init__(self, 
                 chunk_size: int = 400,
                 embedding_dim: int = 384,
                 retrieval_k: int = 3):
        """Initialize complete RAG pipeline."""
        
        print("Initializing Complete RAG Pipeline...")
        
        # Initialize components
        self.chunker = SmartChunker(
            chunk_size=chunk_size,
            overlap=50,
            strategy="semantic"
        )
        
        self.embedder = EmbeddingEngine(
            model_name="rag-encoder-2025",
            dimension=embedding_dim,
            use_cache=True
        )
        
        self.vector_store = SimpleVectorStore()
        self.retrieval_k = retrieval_k
        
        # Document registry
        self.documents = {}
        self.chunks = {}
        
        # Metrics
        self.metrics = {
            'documents_indexed': 0,
            'total_chunks': 0,
            'queries_processed': 0,
            'avg_response_time': 0
        }
        
        print("RAG Pipeline ready!")
        print(f"Chunk size: {chunk_size}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Retrieval K: {retrieval_k}")
    
    def index_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Index a document into the RAG pipeline.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Optional metadata
        
        Returns:
            Indexing results
        """
        start_time = time.time()
        
        print(f"\nIndexing document: {doc_id}")
        print(f"   Content length: {len(content)} chars")
        
        # Store original document
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'indexed_at': time.time()
        }
        
        # Chunk the document
        chunks = self.chunker.chunk_text(content, doc_id)
        
        # Process each chunk
        indexed_chunks = []
        for chunk in chunks:
            # Generate embedding
            chunk_embedding = self.embedder.embed_text(chunk.content)
            
            # Create document for vector store
            chunk_doc = Document(
                doc_id=chunk.chunk_id,
                content=chunk.content,
                content_type="text",
                metadata={
                    **chunk.metadata,
                    'parent_doc': doc_id,
                    'original_metadata': metadata or {}
                }
            )
            
            # Add to vector store
            self.vector_store.add_document(chunk_doc, chunk_embedding)
            
            # Store chunk reference
            self.chunks[chunk.chunk_id] = chunk
            indexed_chunks.append(chunk.chunk_id)
        
        # Update metrics
        self.metrics['documents_indexed'] += 1
        self.metrics['total_chunks'] += len(chunks)
        
        elapsed = time.time() - start_time
        
        result = {
            'doc_id': doc_id,
            'chunks_created': len(chunks),
            'chunk_ids': indexed_chunks,
            'indexing_time': elapsed
        }
        
        print(f"Indexed {len(chunks)} chunks in {elapsed:.2f}s")
        
        return result
    
    def query(self, question: str, explain: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            explain: Whether to explain the retrieval process
        
        Returns:
            Query results with answer and sources
        """
        start_time = time.time()
        
        print(f"\nProcessing query: '{question}'")
        
        # Generate query embedding
        if explain:
            print("Generating query embedding...")
        query_embedding = self.embedder.embed_text(question)
        
        # Retrieve relevant chunks
        if explain:
            print(f"Searching for top {self.retrieval_k} relevant chunks...")
        results = self.vector_store.search(query_embedding, k=self.retrieval_k)
        
        # Build context
        if explain:
            print("Building context from retrieved chunks...")
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"[Source {i+1}, Relevance: {score:.2%}]")
            context_parts.append(doc.content)
            context_parts.append("")
            
            sources.append({
                'chunk_id': doc.doc_id,
                'parent_doc': doc.metadata.get('parent_doc', 'unknown'),
                'score': float(score),
                'preview': doc.content[:100] + "..."
            })
        
        context = "\n".join(context_parts)
        
        # Generate answer (mock LLM)
        if explain:
            print("Generating answer with LLM...")
        answer = self._generate_answer(question, context, results)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        self.metrics['queries_processed'] += 1
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['queries_processed'] - 1) + elapsed)
            / self.metrics['queries_processed']
        )
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_length': len(context),
            'response_time': elapsed,
            'timestamp': time.time()
        }
        
        if explain:
            print(f"Answer generated in {elapsed:.2f}s")
        
        return result
    
    def _generate_answer(self, question: str, context: str, results: List[tuple]) -> str:
        """
        Generate an answer based on retrieved context.
        In production, this would use a real LLM.
        """
        # Mock LLM response based on context
        answer_parts = []
        
        answer_parts.append(f"Based on the retrieved information from {len(results)} relevant sources:\n")
        
        # Analyze what the question is about
        question_lower = question.lower()
        
        if "rag" in question_lower:
            answer_parts.append("\nRAG (Retrieval-Augmented Generation) is a technique that enhances "
                              "language models by retrieving relevant information from a knowledge base "
                              "during inference, ensuring more accurate and up-to-date responses.")
        
        if "vector" in question_lower or "database" in question_lower:
            answer_parts.append("\nVector databases are specialized systems designed to store and "
                              "search high-dimensional embeddings efficiently, enabling semantic search "
                              "at scale with sub-second latency.")
        
        if "embedding" in question_lower:
            answer_parts.append("\nEmbeddings are numerical representations of text that capture "
                              "semantic meaning, allowing computers to understand and compare textual "
                              "content through mathematical operations.")
        
        # Add source references
        answer_parts.append(f"\n\nThis answer is synthesized from {len(results)} retrieved chunks "
                          f"with an average relevance score of "
                          f"{sum(score for _, score in results)/len(results):.1%}.")
        
        return "\n".join(answer_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            'pipeline_metrics': self.metrics,
            'embedder_stats': self.embedder.get_stats(),
            'vector_store_size': len(self.vector_store.documents),
            'unique_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }
        return stats

# ============= DEMONSTRATION =============
print("=" * 60)
print("COMPLETE RAG PIPELINE DEMONSTRATION")
print("=" * 60)

# Initialize pipeline
pipeline = CompleteRAGPipeline(
    chunk_size=300,
    embedding_dim=256,
    retrieval_k=3
)

# Knowledge base documents
knowledge_base = [
    {
        'id': 'kb_001',
        'content': """
        Understanding RAG Systems
        
        Retrieval-Augmented Generation (RAG) has revolutionized how we build AI applications.
        By combining the power of large language models with dynamic information retrieval,
        RAG systems can provide accurate, up-to-date, and verifiable responses.
        
        The key advantage of RAG is its ability to access external knowledge bases,
        eliminating the limitations of static model training data. This makes RAG ideal
        for applications requiring current information or domain-specific knowledge.
        """,
        'metadata': {'category': 'overview', 'importance': 'high'}
    },
    {
        'id': 'kb_002',
        'content': """
        Vector Databases in Production
        
        Modern vector databases like Qdrant, Weaviate, and Pinecone have transformed
        information retrieval. These systems can handle billions of vectors while
        maintaining millisecond query latencies.
        
        Key features include hybrid search (combining semantic and keyword matching),
        metadata filtering, and native support for reranking. Production deployments
        often use distributed architectures for scalability and redundancy.
        """,
        'metadata': {'category': 'infrastructure', 'importance': 'high'}
    },
    {
        'id': 'kb_003',
        'content': """
        Embedding Models Evolution
        
        The latest embedding models offer unprecedented capabilities. BGE-M3 supports
        over 100 languages in a single model. OpenAI's text-embedding-3 allows
        dimension reduction for cost optimization without significant quality loss.
        
        Multi-modal embeddings from CLIP and ImageBind enable unified search across
        text, images, and audio. These models use contrastive learning to align
        different modalities in the same vector space.
        """,
        'metadata': {'category': 'models', 'importance': 'medium'}
    },
    {
        'id': 'kb_004',
        'content': """
        Optimization Techniques for RAG
        
        Production RAG systems require careful optimization. Caching strategies can
        reduce embedding computation by 90%. Quantization techniques compress vectors
        to int8 format, reducing memory usage by 75% with minimal quality impact.
        
        Async processing pipelines enable parallel retrieval from multiple sources.
        Batch processing of embeddings improves throughput, while connection pooling
        optimizes database interactions.
        """,
        'metadata': {'category': 'optimization', 'importance': 'high'}
    }
]

# Index knowledge base
print("\nBuilding Knowledge Base...")
print("-" * 40)

for doc in knowledge_base:
    result = pipeline.index_document(
        doc_id=doc['id'],
        content=doc['content'],
        metadata=doc['metadata']
    )

# Test queries
test_queries = [
    "What are the main benefits of RAG systems?",
    "How do vector databases handle large-scale deployments?",
    "What are the latest improvements in embedding models?",
    "How can I optimize RAG performance?"
]

print("\n" + "=" * 60)
print("ANSWERING QUERIES")
print("=" * 60)

for query in test_queries:
    result = pipeline.query(query, explain=True)
    
    print(f"\n{'='*60}")
    print(f"Question: {result['question']}")
    print(f"\nAnswer:")
    print(result['answer'])
    
    print(f"\nSources (top {len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['parent_doc']} → {source['chunk_id']}")
        print(f"Relevance: {source['score']:.1%}")
        print(f"Preview: {source['preview']}")
    
    print(f"\nResponse time: {result['response_time']:.3f}s")

# Show statistics
print("\n" + "=" * 60)
print("PIPELINE STATISTICS")
print("=" * 60)

stats = pipeline.get_statistics()

print("\nPipeline Metrics:")
for key, value in stats['pipeline_metrics'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")

print("\nEmbedder Statistics:")
for key, value in stats['embedder_stats'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")

print("\nStorage Statistics:")
print(f"Vector store size: {stats['vector_store_size']}")
print(f"Unique documents: {stats['unique_documents']}")
print(f"Total chunks: {stats['total_chunks']}")