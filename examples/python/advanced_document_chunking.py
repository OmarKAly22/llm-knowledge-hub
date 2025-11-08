import re
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ChunkResult:
    """Result of document chunking."""

    chunk_id: str
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]

    def __str__(self):
        preview = (
            self.content[:100] + "..." if len(self.content) > 100 else self.content
        )
        return f"Chunk[{self.chunk_id}]: {preview}"


class SmartChunker:
    """
    Advanced document chunker with multiple strategies.
    This implementation actually chunks text and produces output.
    """

    def __init__(
        self, chunk_size: int = 500, overlap: int = 50, strategy: str = "semantic"
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size for chunks in characters
            overlap: Overlap between chunks
            strategy: Chunking strategy ('fixed', 'semantic', 'sentence')
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        print(f"SmartChunker initialized")
        print(f"Strategy: {strategy}")
        print(f"Chunk size: {chunk_size} chars")
        print(f"Overlap: {overlap} chars")

    def chunk_text(self, text: str, doc_id: str = "doc") -> List[ChunkResult]:
        """
        Chunk text using the selected strategy.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunks with metadata
        """
        print(f"\nChunking document '{doc_id}' ({len(text)} chars)")

        if self.strategy == "fixed":
            chunks = self._fixed_chunking(text)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunking(text)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunking(text)
        else:
            chunks = self._fixed_chunking(text)

        # Create ChunkResult objects
        results = []
        for i, (chunk_text, start, end) in enumerate(chunks):
            chunk_result = ChunkResult(
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                content=chunk_text,
                start_pos=start,
                end_pos=end,
                metadata={
                    "strategy": self.strategy,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "parent_doc": doc_id,
                },
            )
            results.append(chunk_result)

        print(f"Created {len(results)} chunks")
        return results

    def _fixed_chunking(self, text: str) -> List[tuple]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # Extract chunk
            chunk = text[start:end]
            chunks.append((chunk, start, end))

            # Move start position (considering overlap)
            start += self.chunk_size - self.overlap

            # Avoid tiny final chunks
            if len(text) - start < self.overlap:
                break

        return chunks

    def _sentence_chunking(self, text: str) -> List[tuple]:
        """Chunk by sentences, respecting size limits."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # Check if adding sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))

                # Start new chunk with overlap (keep last sentence)
                if self.overlap > 0 and current_chunk:
                    chunk_start = chunk_start + len(chunk_text) - len(current_chunk[-1])
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    chunk_start = chunk_start + len(chunk_text) + 1
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))

        return chunks

    def _semantic_chunking(self, text: str) -> List[tuple]:
        """Chunk by semantic boundaries (paragraphs, sections)."""
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        current_chunk = []
        current_size = 0
        chunk_start = 0
        text_position = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((chunk_text, chunk_start, text_position - 2))

                # Start new chunk
                chunk_start = text_position
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n

            text_position += para_size + 2

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, chunk_start, text_position))

        # If no paragraphs found, fall back to sentence chunking
        if len(chunks) <= 1:
            return self._sentence_chunking(text)

        return chunks

    def analyze_chunks(self, chunks: List[ChunkResult]) -> Dict[str, Any]:
        """Analyze chunking results for quality metrics."""
        if not chunks:
            return {}

        sizes = [len(chunk.content) for chunk in chunks]

        analysis = {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "size_variance": np.std(sizes) if len(sizes) > 1 else 0,
            "total_chars": sum(sizes),
        }

        return analysis


# ============= DEMONSTRATION =============
print("=" * 60)
print("DOCUMENT CHUNKING DEMONSTRATION")
print("=" * 60)

# Sample document
sample_text = """
Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) represents a paradigm shift in how we build AI applications. Unlike traditional language models that rely solely on parametric knowledge, RAG systems dynamically retrieve relevant information from external knowledge bases during inference.

Key Components of RAG

The modern RAG pipeline consists of several critical components. First, the document ingestion system processes and chunks raw documents into manageable segments. These chunks are then embedded using state-of-the-art models like BGE-M3 or OpenAI's text-embedding-3. The resulting vectors are stored in specialized databases such as Qdrant or Weaviate.

During query time, the system converts user questions into embeddings and performs similarity search across the vector database. Retrieved documents are then re-ranked using cross-encoder models to ensure the most relevant information appears first. Finally, the retrieved context is combined with the user query and sent to an LLM for response generation.

Advanced Techniques

Modern RAG systems employ sophisticated techniques to improve retrieval quality. Hybrid search combines dense vector search with sparse keyword matching, leveraging the strengths of both approaches. GraphRAG incorporates knowledge graphs to capture relationships between entities, enabling complex multi-hop reasoning.

Query expansion techniques like HyDE (Hypothetical Document Embeddings) generate synthetic documents to improve retrieval for abstract queries. Meanwhile, adaptive chunking strategies adjust segment sizes based on document structure and content type, ensuring optimal context preservation.

Performance Optimization

Production RAG systems require careful optimization for scalability and cost-effectiveness. Embedding caches reduce redundant computations, while quantization techniques compress vectors without significant quality loss. Asynchronous processing pipelines enable parallel retrieval from multiple sources, dramatically improving response times.

Monitoring and observability are crucial for maintaining system health. Key metrics include retrieval precision, latency percentiles, cache hit rates, and cost per query. These metrics inform continuous optimization efforts and help identify potential issues before they impact users.
"""

# Test different chunking strategies
strategies = ["fixed", "sentence", "semantic"]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Testing {strategy.upper()} chunking strategy")
    print(f"{'='*60}")

    chunker = SmartChunker(chunk_size=400, overlap=50, strategy=strategy)

    chunks = chunker.chunk_text(sample_text, doc_id=f"{strategy}_doc")

    # Show first few chunks
    print(f"\nChunk Results:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{chunk.chunk_id}:")
        print(f"  Position: [{chunk.start_pos}:{chunk.end_pos}]")
        print(f"  Size: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:150]}...")

    if len(chunks) > 3:
        print(f"\n  ... and {len(chunks) - 3} more chunks")

    # Analyze chunks
    analysis = chunker.analyze_chunks(chunks)
    print(f"\nChunk Analysis:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
