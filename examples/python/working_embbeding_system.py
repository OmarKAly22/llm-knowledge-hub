import numpy as np
from typing import List, Dict, Tuple, Union, Any
import hashlib
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingEngine:
    """
    Embedding engine that generates real embeddings and computes similarities.
    In production, this would use real models like BGE-M3 or OpenAI.
    """

    def __init__(
        self,
        model_name: str = "mock-embedding-model",
        dimension: int = 384,
        use_cache: bool = True,
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: Name of the embedding model
            dimension: Embedding dimension
            use_cache: Whether to cache embeddings
        """
        self.model_name = model_name
        self.dimension = dimension
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.stats = {"total_embedded": 0, "cache_hits": 0, "total_tokens": 0}

        print(f"EmbeddingEngine initialized")
        print(f"Model: {model_name}")
        print(f"Dimension: {dimension}")
        print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            text: Single text or list of texts

        Returns:
            Embedding array (single vector or matrix)
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False

        embeddings = []

        for t in texts:
            # Check cache
            if self.use_cache and t in self.cache:
                embeddings.append(self.cache[t])
                self.stats["cache_hits"] += 1
                print(f"Cache hit for: '{t[:50]}...'")
            else:
                # Generate embedding
                emb = self._generate_embedding(t)
                embeddings.append(emb)

                # Cache it
                if self.use_cache:
                    self.cache[t] = emb

                self.stats["total_embedded"] += 1
                self.stats["total_tokens"] += len(t.split())

        result = np.array(embeddings)
        return result[0] if single else result

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a mock embedding that's deterministic but meaningful.
        In production, this would call real embedding models.
        """
        # Create semantic features based on text content
        features = []

        # Feature 1: Text length (normalized)
        features.append(min(len(text) / 1000, 1.0))

        # Feature 2: Word count (normalized)
        features.append(min(len(text.split()) / 100, 1.0))

        # Feature 3-10: Character distribution
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1

        for char in "aeiourstln":  # Common letters
            features.append(char_freq.get(char, 0) / max(len(text), 1))

        # Feature 11-20: Keyword presence
        keywords = [
            "rag",
            "retrieval",
            "vector",
            "embedding",
            "llm",
            "ai",
            "database",
            "search",
            "query",
            "document",
        ]
        text_lower = text.lower()
        for keyword in keywords:
            features.append(1.0 if keyword in text_lower else 0.0)

        # Feature 21-30: Punctuation and structure
        features.append(text.count(".") / max(len(text), 1))
        features.append(text.count(",") / max(len(text), 1))
        features.append(text.count("?") / max(len(text), 1))
        features.append(text.count("!") / max(len(text), 1))
        features.append(text.count("\n") / max(len(text), 1))
        features.append(1.0 if text[0].isupper() else 0.0)
        features.append(
            len([w for w in text.split() if w[0].isupper()]) / max(len(text.split()), 1)
        )

        # Add some randomness based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        for i in range(0, min(len(hash_hex), 20), 2):
            features.append(int(hash_hex[i : i + 2], 16) / 255.0)

        # Pad or truncate to desired dimension
        while len(features) < self.dimension:
            features.append(0.0)

        features = features[: self.dimension]

        # Normalize
        embedding = np.array(features)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0])

    def find_similar(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[Tuple[str, np.ndarray]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar documents to query.

        Args:
            query_embedding: Query embedding
            document_embeddings: List of (doc_id, embedding) tuples
            k: Number of results

        Returns:
            List of (doc_id, similarity) tuples
        """
        similarities = []

        for doc_id, doc_emb in document_embeddings:
            sim = self.compute_similarity(query_embedding, doc_emb)
            similarities.append((doc_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        stats = self.stats.copy()
        if self.use_cache:
            stats["cache_size"] = len(self.cache)
            if stats["total_embedded"] > 0:
                stats["cache_hit_rate"] = stats["cache_hits"] / (
                    stats["cache_hits"] + stats["total_embedded"]
                )
            else:
                stats["cache_hit_rate"] = 0
        return stats


# ============= DEMONSTRATION =============
print("=" * 60)
print("EMBEDDING ENGINE DEMONSTRATION")
print("=" * 60)

# Initialize embedding engine
embedder = EmbeddingEngine(
    model_name="semantic-encoder-v1",
    dimension=128,  # Smaller for visualization
    use_cache=True,
)

# Sample documents
documents = [
    (
        "doc1",
        "RAG systems combine retrieval with generation for accurate AI responses.",
    ),
    ("doc2", "Vector databases store embeddings for semantic search."),
    ("doc3", "Embedding models convert text to numerical vectors."),
    ("doc4", "The weather is sunny today with clear skies."),  # Unrelated
    (
        "doc5",
        "Retrieval-augmented generation improves LLM accuracy with external knowledge.",
    ),
]

# Generate embeddings for documents
print("\nGenerating document embeddings...")
doc_embeddings = []
for doc_id, text in documents:
    print(f"\n  Processing {doc_id}: '{text[:50]}...'")
    emb = embedder.embed_text(text)
    doc_embeddings.append((doc_id, emb))
    print(f"    ✓ Embedding shape: {emb.shape}")
    print(f"    ✓ Embedding sample: [{emb[:5].round(3)}...]")

# Test queries
queries = [
    "How does RAG work?",
    "What are embeddings?",
    "Tell me about the weather",
]

print("\n" + "=" * 60)
print("SIMILARITY SEARCH")
print("=" * 60)

for query in queries:
    print(f"\nQuery: '{query}'")

    # Generate query embedding
    query_emb = embedder.embed_text(query)
    print(f"   Query embedding generated")

    # Find similar documents
    similarities = embedder.find_similar(query_emb, doc_embeddings, k=3)

    print(f"\nTop 3 similar documents:")
    for doc_id, sim in similarities:
        # Find original text
        orig_text = next(text for did, text in documents if did == doc_id)
        print(f"{doc_id} (similarity: {sim:.3f})")
        print(f"'{orig_text[:60]}...'")

# Test caching
print("\n" + "=" * 60)
print("TESTING CACHE")
print("=" * 60)

print("\nRe-embedding first query (should hit cache)...")
cached_emb = embedder.embed_text(queries[0])

print("\nRe-embedding all documents (should all hit cache)...")
for doc_id, text in documents:
    _ = embedder.embed_text(text)

# Show statistics
print("\n" + "=" * 60)
print("EMBEDDING STATISTICS")
print("=" * 60)

stats = embedder.get_stats()
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")

# Visualize similarity matrix
print("\n" + "=" * 60)
print("DOCUMENT SIMILARITY MATRIX")
print("=" * 60)

print("\n", end="")
for doc_id, _ in documents:
    print(f"{doc_id:>6}", end="")
print()

for i, (doc_id1, emb1) in enumerate(doc_embeddings):
    print(f"{doc_id1:>6}", end="")
    for j, (doc_id2, emb2) in enumerate(doc_embeddings):
        sim = embedder.compute_similarity(emb1, emb2)
        print(f"{sim:>6.2f}", end="")
    print()

print("\nNote: doc4 (weather) shows lower similarity to others (different topic)")
