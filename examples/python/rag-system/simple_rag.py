# examples/python/rag-system/simple_rag.py
"""
Simple RAG (Retrieval-Augmented Generation) System
Demonstrates embedding, vector search, and context injection
"""

import numpy as np
from typing import List, Dict


class SimpleEmbedding:
    """
    Simplified embedding model (uses basic word counting)
    In production, use OpenAI embeddings, sentence-transformers, etc.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}

    def embed(self, text: str) -> np.ndarray:
        """Create a simple embedding vector"""
        words = text.lower().split()

        # Build vocabulary
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        # Create simple bag-of-words vector
        vector = np.zeros(min(self.vocab_size, len(self.vocab) + 100))
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word] % len(vector)
                vector[idx] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector


class Document:
    """Represents a document with content and metadata"""

    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None

    def __repr__(self):
        return f"Document(content='{self.content[:50]}...', metadata={self.metadata})"


class VectorStore:
    """
    Simple in-memory vector store
    In production, use Pinecone, Chroma, Qdrant, etc.
    """

    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        for doc in documents:
            doc.embedding = self.embedding_model.embed(doc.content)
            self.documents.append(doc)
        print(f"✅ Added {len(documents)} documents to vector store")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Find k most similar documents to the query
        Uses cosine similarity
        """
        query_embedding = self.embedding_model.embed(query)

        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding)
            similarities.append((doc, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k documents
        return [doc for doc, _ in similarities[:k]]


class SimpleRAG:
    """
    Simple RAG system combining retrieval and generation
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def query(self, question: str, k: int = 3) -> str:
        """
        Query the RAG system
        1. Retrieve relevant documents
        2. Create context from retrieved docs
        3. Generate answer with context
        """
        print(f"\n🔍 Query: {question}\n")

        # Step 1: Retrieve relevant documents
        print("📚 Retrieving relevant documents...")
        relevant_docs = self.vector_store.similarity_search(question, k=k)

        print(f"Found {len(relevant_docs)} relevant documents:\n")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"  {i}. {doc.content[:80]}...")

        # Step 2: Create context
        context = self._create_context(relevant_docs)

        # Step 3: Generate answer (simulated)
        answer = self._generate_answer(question, context)

        return answer

    def _create_context(self, documents: List[Document]) -> str:
        """Create context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]\n{doc.content}\n")

        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer based on question and context
        In production, this would be an LLM call
        """
        print("\n💭 Generating answer with context...\n")

        # Simulated LLM response
        prompt = f"""
Context:
{context}

Question: {question}

Answer: Based on the provided context, here is the answer...
"""

        # In production, you would do:
        # answer = llm.generate(prompt)

        return f"""Based on the retrieved documents, I can answer your question about '{question}'.

The relevant information shows that the answer involves the key concepts found in the documents.

(In production, this would be a real LLM-generated answer based on the context)"""


def create_sample_knowledge_base() -> List[Document]:
    """Create sample documents for demonstration"""
    documents = [
        Document(
            "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum.",
            {"source": "python_intro", "topic": "programming"},
        ),
        Document(
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            {"source": "ml_basics", "topic": "ai"},
        ),
        Document(
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide more accurate and contextual responses.",
            {"source": "rag_explained", "topic": "ai"},
        ),
        Document(
            "Vector databases store data as high-dimensional vectors, enabling fast similarity search for AI applications.",
            {"source": "vector_db", "topic": "databases"},
        ),
        Document(
            "Embeddings are numerical representations of text that capture semantic meaning, allowing computers to understand language similarity.",
            {"source": "embeddings_intro", "topic": "nlp"},
        ),
        Document(
            "LangChain is a framework for developing applications powered by language models, providing tools for chains, agents, and memory.",
            {"source": "langchain_intro", "topic": "frameworks"},
        ),
        Document(
            "Prompt engineering involves crafting effective instructions for LLMs to get desired outputs, including techniques like few-shot learning.",
            {"source": "prompting", "topic": "llm"},
        ),
        Document(
            "Token limits define the maximum context window size for LLMs. GPT-4 supports up to 128K tokens while GPT-3.5 supports 16K.",
            {"source": "token_limits", "topic": "llm"},
        ),
    ]
    return documents


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Simple RAG System Demo")
    print("=" * 70)

    # Initialize components
    print("\n🔧 Initializing RAG system...")
    embedding_model = SimpleEmbedding()
    vector_store = VectorStore(embedding_model)

    # Create knowledge base
    print("\n📖 Creating knowledge base...")
    documents = create_sample_knowledge_base()
    vector_store.add_documents(documents)

    # Create RAG system
    rag = SimpleRAG(vector_store)

    # Example queries
    queries = [
        "What is RAG?",
        "Tell me about Python programming",
        "How do token limits work in LLMs?",
        "What are embeddings?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        answer = rag.query(query, k=2)
        print(f"\n✅ Answer:\n{answer}")

    print("\n" + "=" * 70)
    print("🎉 RAG Demo Complete!")
    print("=" * 70)
