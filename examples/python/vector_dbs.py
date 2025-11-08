from typing import List, Dict, Optional, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from datetime import datetime


@dataclass
class SearchResult:
    """Modern search result with metadata."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: Optional[List[str]] = None
    explanation: Optional[str] = None


# Mock Document class
@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime


# Mock DB to test methods without real Qdrant/Weaviate
class MockVectorDB:
    """Mock vector DB for testing."""

    def __init__(self):
        self.storage = []

    async def insert(self, documents, embeddings):
        for doc, emb in zip(documents, embeddings):
            self.storage.append((doc, emb))
        return [doc.doc_id for doc in documents]

    async def search(self, query_embedding, k=10, filters=None):
        results = []
        for doc, emb in self.storage:
            score = float(
                np.dot(query_embedding, emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            )
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def hybrid_search(self, query_embedding, query_text, k=10, alpha=0.5):
        # Simple: just call search for mock
        return await self.search(query_embedding, k)


class VectorDatabase(ABC):
    """Abstract base for vector databases."""

    @abstractmethod
    async def insert(
        self, documents: List[Document], embeddings: np.ndarray
    ) -> List[str]:
        """Insert documents with embeddings."""
        pass

    @abstractmethod
    async def search(
        self, query_embedding: np.ndarray, k: int = 10, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """Hybrid search combining vector and keyword search."""
        pass


class QdrantVectorDB(VectorDatabase):
    """
    Qdrant vector database - high-performance, cloud-native.
    One of the most popular choices in 2025.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        use_cloud: bool = False,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant client.

        Args:
            collection_name: Name of the collection
            host: Qdrant host
            port: Qdrant port
            use_cloud: Use Qdrant Cloud
            api_key: API key for cloud
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if use_cloud:
            self.client = QdrantClient(
                url=host,
                api_key=api_key,
            )
        else:
            self.client = QdrantClient(host=host, port=port)

        self.collection_name = collection_name

        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=768,  # Default size, adjust based on model
                    distance=Distance.COSINE,
                ),
            )

    async def insert(
        self, documents: List[Document], embeddings: np.ndarray
    ) -> List[str]:
        """Insert documents into Qdrant."""
        from qdrant_client.models import PointStruct

        points = []
        doc_ids = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=doc.doc_id,
                vector=embedding.tolist(),
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp.isoformat(),
                },
            )
            points.append(point)
            doc_ids.append(doc.doc_id)

        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        return doc_ids

    async def search(
        self, query_embedding: np.ndarray, k: int = 10, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Vector similarity search."""
        from qdrant_client.models import Filter, FieldCondition

        # Build filter if provided
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match={"value": value})
                )
            search_filter = Filter(must=conditions)

        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            query_filter=search_filter,
            with_payload=True,
        )

        # Convert to SearchResult
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    doc_id=result.id,
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata=result.payload.get("metadata", {}),
                )
            )

        return search_results

    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Hybrid search in Qdrant.
        Combines vector search with full-text search.
        """
        # Qdrant 1.7+ supports hybrid search natively
        from qdrant_client.models import SearchRequest, TextIndexParams

        # Perform hybrid search
        results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=query_embedding.tolist(),
                    limit=k,
                    with_payload=True,
                    params={"hnsw_ef": 128, "exact": False},
                )
            ],
        )

        # Note: Full hybrid search requires text index setup
        # This is simplified - in production, configure text indices

        search_results = []
        for batch_results in results:
            for result in batch_results:
                search_results.append(
                    SearchResult(
                        doc_id=result.id,
                        content=result.payload.get("content", ""),
                        score=result.score,
                        metadata=result.payload.get("metadata", {}),
                    )
                )

        return search_results


class WeaviateVectorDB(VectorDatabase):
    """
    Weaviate - GraphQL-based vector database with native hybrid search.
    Excellent for complex queries and knowledge graphs.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        auth_key: Optional[str] = None,
        class_name: str = "Document",
    ):
        """Initialize Weaviate client."""
        import weaviate
        from weaviate.auth import AuthApiKey

        if auth_key:
            self.client = weaviate.Client(
                url=url, auth_client_secret=AuthApiKey(api_key=auth_key)
            )
        else:
            self.client = weaviate.Client(url=url)

        self.class_name = class_name

        # Create schema if it doesn't exist
        self._create_schema()

    def _create_schema(self):
        """Create Weaviate schema."""
        schema = {
            "class": self.class_name,
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "tokenization": "word",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                },
                {"name": "metadata", "dataType": ["object"], "nestedProperties": []},
                {"name": "doc_id", "dataType": ["string"], "tokenization": "field"},
            ],
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "poolingStrategy": "masked_mean",
                    "vectorizeClassName": False,
                }
            },
        }

        try:
            self.client.schema.create_class(schema)
        except:
            pass  # Schema already exists

    async def insert(
        self, documents: List[Document], embeddings: np.ndarray
    ) -> List[str]:
        """Insert documents into Weaviate."""
        doc_ids = []

        with self.client.batch as batch:
            for doc, embedding in zip(documents, embeddings):
                properties = {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "doc_id": doc.doc_id,
                }

                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=embedding.tolist(),
                )
                doc_ids.append(doc.doc_id)

        return doc_ids

    async def search(
        self, query_embedding: np.ndarray, k: int = 10, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Vector search in Weaviate."""
        query = (
            self.client.query.get(self.class_name, ["content", "metadata", "doc_id"])
            .with_near_vector({"vector": query_embedding.tolist()})
            .with_limit(k)
            .with_additional(["distance"])
        )

        # Add filters if provided
        if filters:
            where_filter = self._build_where_filter(filters)
            query = query.with_where(where_filter)

        result = query.do()

        search_results = []
        if self.class_name in result.get("data", {}).get("Get", {}):
            for item in result["data"]["Get"][self.class_name]:
                search_results.append(
                    SearchResult(
                        doc_id=item.get("doc_id", ""),
                        content=item.get("content", ""),
                        score=1
                        - item["_additional"][
                            "distance"
                        ],  # Convert distance to similarity
                        metadata=item.get("metadata", {}),
                    )
                )

        return search_results

    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Native hybrid search in Weaviate.
        Combines vector and BM25 search.
        """
        query = (
            self.client.query.get(self.class_name, ["content", "metadata", "doc_id"])
            .with_hybrid(
                query=query_text,
                vector=query_embedding.tolist(),
                alpha=alpha,  # 0 = keyword only, 1 = vector only
            )
            .with_limit(k)
            .with_additional(["score", "explainScore"])
        )

        result = query.do()

        search_results = []
        if self.class_name in result.get("data", {}).get("Get", {}):
            for item in result["data"]["Get"][self.class_name]:
                search_results.append(
                    SearchResult(
                        doc_id=item.get("doc_id", ""),
                        content=item.get("content", ""),
                        score=item["_additional"]["score"],
                        metadata=item.get("metadata", {}),
                        explanation=item["_additional"].get("explainScore", ""),
                    )
                )

        return search_results

    def _build_where_filter(self, filters: Dict) -> Dict:
        """Build Weaviate where filter."""
        conditions = []

        for key, value in filters.items():
            conditions.append({"path": [key], "operator": "Equal", "value": value})

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"operator": "And", "operands": conditions}


async def main():
    # Create sample documents
    docs = [
        Document("doc1", "AI embeddings are amazing.", {"type": "ai"}, datetime.now()),
        Document(
            "doc2", "Vector databases store embeddings.", {"type": "db"}, datetime.now()
        ),
        Document(
            "doc3",
            "Weaviate supports hybrid search.",
            {"type": "weaviate"},
            datetime.now(),
        ),
    ]

    # Create mock embeddings
    embeddings = np.random.randn(len(docs), 128)

    # Initialize mock DB
    db = MockVectorDB()

    # Test insert
    inserted_ids = await db.insert(docs, embeddings)
    print("Inserted document IDs:", inserted_ids)

    # Test search
    query_emb = np.random.randn(128)
    search_results = await db.search(query_emb, k=2)
    print("\nSearch Results:")
    for r in search_results:
        print(f"{r.doc_id}: {r.content} (score: {r.score:.3f})")

    # Test hybrid search
    hybrid_results = await db.hybrid_search(query_emb, "embeddings")
    print("\nHybrid Search Results:")
    for r in hybrid_results:
        print(f"{r.doc_id}: {r.content} (score: {r.score:.3f})")


# Run main
if __name__ == "__main__":
    asyncio.run(main())
