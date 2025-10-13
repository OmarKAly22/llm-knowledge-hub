# Tools and Frameworks

## LLM Providers

### Commercial APIs
- **OpenAI**: GPT-4, GPT-3.5 - Industry leading models
  - Website: https://platform.openai.com
  - Best for: General purpose, strong reasoning
- **Anthropic**: Claude 3 family - Long context, safety-focused
  - Website: https://www.anthropic.com
  - Best for: Long documents, ethical AI
- **Google**: Gemini - Multimodal capabilities
  - Website: https://ai.google.dev
  - Best for: Multimodal tasks
- **Cohere**: Command models - Enterprise-focused
  - Website: https://cohere.com
  - Best for: Enterprise deployments

### Open Source Models
- **Meta Llama**: 7B to 70B parameters
  - Website: https://llama.meta.com
  - Best for: Local deployment, customization
- **Mistral**: 7B, Mixtral 8x7B
  - Website: https://mistral.ai
  - Best for: Efficient performance
- **Falcon**: 7B to 180B
  - Website: https://falconllm.tii.ae
  - Best for: Research, experimentation
- **MPT**: MosaicML models
  - Website: https://www.mosaicml.com
  - Best for: Commercial use cases

## Development Frameworks

### Agent Frameworks
- **LangChain**: Comprehensive agent/chain framework
  - GitHub: https://github.com/langchain-ai/langchain
  - Best for: Quick prototyping, extensive integrations
- **LangGraph**: State-machine based workflows
  - GitHub: https://github.com/langchain-ai/langgraph
  - Best for: Complex stateful workflows
- **AutoGen**: Multi-agent conversations (Microsoft)
  - GitHub: https://github.com/microsoft/autogen
  - Best for: Multi-agent systems
- **CrewAI**: Role-based agent teams
  - GitHub: https://github.com/joaomdmoura/crewAI
  - Best for: Business workflows
- **Semantic Kernel**: Microsoft's orchestration framework
  - GitHub: https://github.com/microsoft/semantic-kernel
  - Best for: Enterprise .NET applications

### RAG Frameworks
- **LlamaIndex**: Data framework for LLMs
  - GitHub: https://github.com/run-llama/llama_index
  - Best for: Complex data ingestion
- **Haystack**: NLP framework with RAG support
  - GitHub: https://github.com/deepset-ai/haystack
  - Best for: Production RAG pipelines
- **LangChain**: RAG chains and retrievers
  - Best for: Quick RAG implementation

## Vector Databases

### Cloud-based
- **Pinecone**: Managed vector database
  - Website: https://www.pinecone.io
  - Best for: Scalable cloud deployment
- **Weaviate**: Open-source, cloud available
  - Website: https://weaviate.io
  - Best for: Hybrid search, flexibility
- **Qdrant**: Fast similarity search
  - Website: https://qdrant.tech
  - Best for: High performance

### Self-hosted
- **Chroma**: Embedded vector database
  - GitHub: https://github.com/chroma-core/chroma
  - Best for: Local development, simplicity
- **Milvus**: Scalable vector database
  - Website: https://milvus.io
  - Best for: Large-scale deployments
- **FAISS**: Facebook's similarity search
  - GitHub: https://github.com/facebookresearch/faiss
  - Best for: Research, CPU optimization

## Tools

### Embeddings
- **OpenAI Embeddings**: text-embedding-3-small/large
  - Best for: High quality, ease of use
- **Sentence Transformers**: Open-source models
  - GitHub: https://github.com/UKPLab/sentence-transformers
  - Best for: Local deployment, free
- **Cohere Embed**: Multilingual embeddings
  - Best for: Multiple languages

### Observability
- **LangSmith**: LangChain monitoring
  - Website: https://www.langchain.com/langsmith
  - Best for: LangChain applications
- **Helicone**: LLM observability platform
  - Website: https://www.helicone.ai
  - Best for: Multi-provider monitoring
- **Weights & Biases**: ML experiment tracking
  - Website: https://wandb.ai
  - Best for: ML experiments
- **PromptLayer**: Prompt management and tracking
  - Website: https://promptlayer.com
  - Best for: Prompt versioning

### Testing
- **PromptFoo**: LLM testing framework
  - GitHub: https://github.com/promptfoo/promptfoo
  - Best for: Automated testing
- **Giskard**: AI testing and monitoring
  - Website: https://www.giskard.ai
  - Best for: Quality assurance
- **DeepEval**: LLM evaluation framework
  - GitHub: https://github.com/confident-ai/deepeval
  - Best for: Evaluation metrics

## Local Development

### Running Models Locally
- **Ollama**: Easy local model running
  - Website: https://ollama.com
  - Best for: Simplest local setup
- **LM Studio**: GUI for local models
  - Website: https://lmstudio.ai
  - Best for: Non-technical users
- **llama.cpp**: CPU-optimized inference
  - GitHub: https://github.com/ggerganov/llama.cpp
  - Best for: CPU-only systems
- **vLLM**: Fast inference engine
  - GitHub: https://github.com/vllm-project/vllm
  - Best for: Production inference

### IDEs with AI
- **Cursor**: AI-first code editor
  - Website: https://cursor.sh
  - Best for: AI-native development
- **GitHub Copilot**: AI pair programmer
  - Website: https://github.com/features/copilot
  - Best for: Code completion
- **Continue**: Open-source Copilot alternative
  - Website: https://continue.dev
  - Best for: Open-source option

## Deployment & Production

### Model Serving
- **TGI (Text Generation Inference)**: Hugging Face
  - GitHub: https://github.com/huggingface/text-generation-inference
  - Best for: Hugging Face models
- **vLLM**: High-throughput inference
  - Best for: Production serving
- **Ray Serve**: Scalable ML serving
  - Website: https://docs.ray.io/en/latest/serve/
  - Best for: Complex deployments

### Cloud Platforms
- **AWS SageMaker**: Full ML platform
  - Best for: AWS ecosystem
- **Google Cloud AI**: Vertex AI
  - Best for: Google Cloud users
- **Azure AI**: Azure ML
  - Best for: Microsoft ecosystem
- **Hugging Face Inference**: Managed endpoints
  - Best for: Quick deployment

## Prompt Engineering Tools

- **PromptBase**: Prompt marketplace
  - Website: https://promptbase.com
- **PromptPerfect**: Prompt optimization
  - Website: https://promptperfect.jina.ai
- **LangChain Hub**: Prompt templates
  - Website: https://smith.langchain.com/hub

## Data Processing

- **Unstructured**: Document processing
  - GitHub: https://github.com/Unstructured-IO/unstructured
  - Best for: PDF, DOCX, etc.
- **LangChain Document Loaders**: Various formats
  - Best for: Quick integration
- **PyPDF**: PDF processing
  - Best for: Simple PDF tasks

## Cost Management

- **OpenMeter**: Usage tracking
  - Website: https://openmeter.io
- **Helicone**: Cost analytics
  - Best for: Multi-provider costs
- **LangSmith**: Token tracking
  - Best for: LangChain apps

---

## Recommended Stacks

### Beginner Stack
- Model: OpenAI GPT-3.5
- Framework: LangChain
- Vector DB: Chroma
- Observability: LangSmith

### Production Stack
- Model: OpenAI GPT-4 + local fallback
- Framework: Custom or LangGraph
- Vector DB: Pinecone or Qdrant
- Observability: Helicone + custom logging

### Local/Open Source Stack
- Model: Llama 3 via Ollama
- Framework: LangChain or custom
- Vector DB: Chroma or FAISS
- Observability: Weights & Biases

### Enterprise Stack
- Model: Azure OpenAI
- Framework: Semantic Kernel
- Vector DB: Azure AI Search
- Observability: Azure Monitor

---

*Last updated: October 2025*