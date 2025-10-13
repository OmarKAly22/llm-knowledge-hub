# LLM Era Q&A: Essential Knowledge for Software Engineers

## Prompt Engineering

### Q: What's the difference between system prompts and user prompts?
**A:** System prompts set the AI's behavior, role, and constraints (like "You are a helpful coding assistant"). User prompts are the actual queries or instructions. System prompts persist across the conversation while user prompts change each time.

### Q: When should I use few-shot vs zero-shot prompting?
**A:** Use zero-shot when the task is simple and well-understood (like "Summarize this text"). Use few-shot (providing examples) when you need specific output formatting, consistent style, or the task is ambiguous. Generally: 0 examples for simple tasks, 1-3 examples for format guidance, 5+ examples for complex pattern recognition.

### Q: What's temperature and when should I adjust it?
**A:** Temperature controls randomness (0.0 to 2.0). Use low temperature (0.0-0.3) for factual, deterministic tasks like code generation or data extraction. Use medium (0.5-0.7) for balanced creativity. Use high (0.8-1.0+) for creative writing or brainstorming. Default is usually 0.7-1.0.

### Q: How do I prevent the LLM from making things up?
**A:** You can't eliminate hallucinations completely, but you can reduce them by: (1) Being specific in prompts, (2) Asking for sources/citations, (3) Using RAG to ground responses in real data, (4) Adding "If you don't know, say so" instructions, (5) Lowering temperature, (6) Validating outputs programmatically.

### Q: What are the most common prompt engineering mistakes?
**A:** (1) Being too vague or ambiguous, (2) Not specifying output format, (3) Mixing multiple unrelated tasks in one prompt, (4) Not providing necessary context, (5) Over-complicating simple tasks, (6) Not testing with edge cases, (7) Assuming the model remembers previous context without including it.

### Q: Should I write prompts in natural language or be more structured?
**A:** Both work, but structured prompts are more reliable. Use clear sections with XML/markdown tags, numbered steps, and explicit constraints. For production systems, structured prompts are easier to version control and debug.

---

## AI Integration Architecture

### Q: When should I use streaming vs waiting for complete responses?
**A:** Use streaming for user-facing applications where perceived latency matters (chat interfaces, writing assistants). Wait for complete responses when you need to process the full output programmatically, validate it, or when the response will be short. Streaming improves UX but adds complexity to error handling.

### Q: How do I handle rate limits from LLM APIs?
**A:** Implement exponential backoff with retries, use token bucket algorithms for request management, implement queuing systems for high-traffic apps, cache responses aggressively, and monitor your usage to stay within quotas. Always have fallback behavior for when limits are hit.

### Q: Should I build with multiple LLM providers or stick to one?
**A:** For production apps, having fallback providers is smart but adds complexity. Start with one provider, design your abstraction layer well (so you can swap providers), then add alternatives if: (1) You need higher reliability, (2) Different models excel at different tasks, (3) You want cost optimization, or (4) You face geopolitical/business risks with a single provider.

### Q: What's the best way to validate LLM outputs before using them?
**A:** Layer your validation: (1) Schema validation (is it the right format?), (2) Constraint checking (within expected ranges?), (3) Business logic validation (makes sense for your domain?), (4) Sanity checks (no obvious errors?), (5) For critical operations, use a second LLM call to verify, (6) Always have a human review option for high-stakes decisions.

### Q: How do I know if I should use an LLM or traditional code?
**A:** Use traditional code when: logic is deterministic, performance is critical, you need 100% reliability, or the task is simple (like sorting). Use LLMs when: you need natural language understanding, the problem requires reasoning, rules are fuzzy or change frequently, or you're dealing with unstructured data. Often the answer is both—LLM for understanding/generation, traditional code for validation/execution.

---

## RAG (Retrieval-Augmented Generation)

### Q: When do I need RAG vs just using the LLM's knowledge?
**A:** Use RAG when: (1) You need information beyond the model's training cutoff, (2) You have proprietary/private data, (3) You need factual accuracy with citations, (4) Information changes frequently, (5) The knowledge base is too large to fit in context, or (6) You want to reduce hallucinations by grounding responses.

### Q: What's the optimal chunk size for documents?
**A:** There's no universal answer, but 500-1000 tokens (roughly 300-700 words) works for most cases. Smaller chunks (200-500 tokens) for precise retrieval, larger chunks (1000-2000) when you need more context. Test with your specific data. Also consider semantic chunking (by paragraphs/sections) rather than fixed sizes.

### Q: Vector database vs traditional database—which should I use?
**A:** Use vector databases for semantic search over unstructured text. Use traditional databases for structured data queries. In practice, use both: vector DB for finding relevant documents, traditional DB for filtering by metadata (dates, categories, IDs). Hybrid search (combining semantic + keyword + metadata) often works best.

### Q: How many documents should I retrieve and pass to the LLM?
**A:** Start with 3-5 chunks (top-k=3-5). More isn't always better—too many chunks add noise and cost. If answers are incomplete, increase to 10. If you're getting irrelevant information, reduce to 3 or improve your retrieval. Consider reranking: retrieve 20, rerank to get best 5.

### Q: My RAG system returns irrelevant results. How do I fix it?
**A:** Common fixes: (1) Improve chunking strategy (better semantic boundaries), (2) Add metadata filtering before semantic search, (3) Use better embedding models, (4) Implement reranking, (5) Improve query transformation (rephrase user questions), (6) Use hybrid search (semantic + keyword), (7) Fine-tune your embedding model on your domain.

### Q: Should I rebuild my vector database every time my data changes?
**A:** No—use incremental updates. Most vector DBs support adding/updating/deleting individual vectors. Batch updates when possible for efficiency. Full rebuilds only when you change the embedding model or chunking strategy.

---

## Context Management & Memory

### Q: How do I handle conversations that exceed the token limit?
**A:** Strategies: (1) Sliding window (keep last N messages), (2) Summarize old messages periodically, (3) Extract key facts and store separately, (4) Use external memory for long-term facts, (5) For very long contexts, use models with larger context windows, (6) Prioritize recent and important context over everything.

### Q: What's the difference between token limits and actual usable context?
**A:** Token limits are maximums, but usable context is less due to: (1) System prompts take tokens, (2) Your instructions take tokens, (3) Response generation needs tokens, (4) Attention degrades with very long contexts (models forget the middle). Rule of thumb: use 60-80% of stated limits for reliable performance.

### Q: Should I store conversation history in the database?
**A:** Yes for production apps. Store messages with timestamps, track conversation threads, enable users to resume conversations. This is separate from passing history to the LLM—you store everything but only send relevant recent context to the LLM.

### Q: How do I implement "memory" for a chatbot across sessions?
**A:** Multi-layered approach: (1) Conversation history (short-term): Store in DB, load recent messages, (2) User facts (long-term): Extract and store key information about the user, (3) Preferences: Track user preferences separately, (4) Semantic memory: Use vector DB to search past relevant conversations. On each new message, retrieve relevant memories and include in context.

### Q: What's the best way to summarize long conversations?
**A:** Periodic summarization: Every N messages, ask the LLM to summarize what's been discussed, then replace old messages with the summary. Keep the most recent messages unsummarized for detail. Alternative: Maintain a running summary that gets updated with each interaction.

---

## Evaluation & Testing

### Q: How do I write unit tests for LLM features?
**A:** You can't test LLMs like deterministic code, but you can: (1) Test tool/function calling with known inputs, (2) Use assertions on output structure (JSON schema, required fields), (3) Check for presence of key information, (4) Test negative cases (refuses harmful requests), (5) Use temperature=0 for more deterministic outputs in tests, (6) Track regression with golden datasets.

### Q: What's the difference between automated metrics (BLEU, ROUGE) and LLM-as-judge?
**A:** Automated metrics compare text similarity (n-gram overlap) and are fast but shallow—they miss semantic equivalence. LLM-as-judge uses another LLM to evaluate quality, relevance, correctness. It's slower and costs money but better captures semantic quality. Use automated metrics for quick feedback, LLM-as-judge for quality, human eval for final validation.

### Q: How do I evaluate creative outputs like marketing copy?
**A:** This is hard to automate. Strategies: (1) LLM-as-judge with criteria (persuasiveness, clarity, tone), (2) Human evaluation panels, (3) A/B testing with real users, (4) Proxy metrics (engagement, click-through rates), (5) Brand compliance checks (automated), (6) Combine multiple evaluation methods.

### Q: What metrics should I track in production for LLM features?
**A:** Technical: Latency, token usage, cost per request, error rates, timeout rates. Quality: User thumbs up/down, explicit feedback, task completion rates, retry rates. Safety: Content filter triggers, policy violations, user reports. Track these over time and set up alerts for anomalies.

### Q: How many evaluation examples do I need?
**A:** Start with 20-50 diverse examples covering common cases and edge cases. Expand to 100-200 for production confidence. For critical applications, aim for 500+. Quality matters more than quantity—ensure examples cover different scenarios, edge cases, and failure modes.

---

## Security & Safety

### Q: What is prompt injection and how do I prevent it?
**A:** Prompt injection is when users manipulate the AI by putting instructions in their input (like "Ignore previous instructions and..."). Prevention: (1) Clearly separate system instructions from user input using delimiters, (2) Validate and sanitize user inputs, (3) Use separate system/user message roles, (4) Add explicit warnings in system prompt, (5) Monitor outputs for policy violations, (6) For critical operations, always validate with traditional code.

### Q: Should I filter LLM outputs or inputs?
**A:** Both. Input filtering prevents malicious prompts and reduces costs by blocking obviously bad requests. Output filtering catches when the LLM generates inappropriate content despite your safeguards. Use content moderation APIs (OpenAI Moderation, Perspective API) or build custom filters.

### Q: Can I safely put API keys or secrets in prompts?
**A:** Never. LLMs can leak information from their context through clever prompting. Instead: (1) Use secure credential storage, (2) Make API calls from your backend, not through the LLM, (3) Give LLMs access to tools that call APIs, rather than the keys themselves, (4) Audit all LLM outputs for accidental secret disclosure.

### Q: How do I prevent my chatbot from being used for harmful purposes?
**A:** Layered approach: (1) Clear usage policies in system prompt, (2) Content filtering on inputs/outputs, (3) Rate limiting per user, (4) Monitoring for abuse patterns, (5) Human review for reported content, (6) Don't give LLMs access to dangerous tools without approval gates, (7) Implement usage analytics to detect misuse patterns.

### Q: Should I worry about jailbreaking?
**A:** Yes, but don't over-worry. Jailbreaking is constantly evolving. Best practices: (1) Use models with strong safety training, (2) Layer your defenses (system prompts + content filters + output validation), (3) Monitor for attempts, (4) Have incident response plans, (5) Accept that determined attackers might succeed—focus on detecting and responding. For critical applications, add human oversight.

---

## Cost & Performance Optimization

### Q: What's more expensive—input tokens or output tokens?
**A:** Output tokens typically cost 2-3x more than input tokens. This matters for cost optimization: it's cheaper to send long prompts with context than to have the LLM generate lengthy responses. Cache context-heavy prompts, be concise in output requirements.

### Q: When should I use GPT-4 vs GPT-3.5 vs local models?
**A:** GPT-4/Claude Opus: Complex reasoning, code generation, important tasks where quality matters. GPT-3.5/Claude Sonnet: Everyday tasks, summarization, simple classification—10x cheaper. Local models (Llama, Mistral): High-volume tasks, privacy-sensitive data, when you control infrastructure. Start with strong models, optimize to weaker ones where quality is acceptable.

### Q: What's the ROI of caching?
**A:** Huge for repeated queries. Caching identical inputs can give 90%+ cost reduction for common queries. Even semantic caching (similar queries) can save 30-50%. Implement caching early—it's low-hanging fruit for cost optimization. Balance cache size vs hit rate.

### Q: How do I reduce latency for LLM calls?
**A:** (1) Use streaming for perceived performance, (2) Parallel calls when tasks are independent, (3) Edge deployment closer to users, (4) Smaller models for simple tasks, (5) Precompute when possible (embeddings, common responses), (6) Use faster inference providers (Groq, Together), (7) Batch requests when appropriate.

### Q: What's a realistic cost per user per month for an LLM-powered app?
**A:** Highly variable. Simple chatbot: $0.10-$1 per active user. Heavy usage (coding assistant, research tool): $5-$20 per user. Enterprise features with large context: $20-$100+. Monitor your specific usage patterns. Set per-user limits to prevent runaway costs.

---

## MCP (Model Context Protocol)

### Q: What problem does MCP solve?
**A:** MCP standardizes how LLMs access external data and tools. Before MCP, every AI app needed custom integrations for each data source. MCP provides a universal protocol—write once, use everywhere. It's like USB for AI integrations.

### Q: Should I build an MCP server or just use API calls directly?
**A:** Use MCP when: (1) You want reusability across different AI apps, (2) You're building tools others will use, (3) You want standardization, or (4) You're integrating with MCP-compatible clients. Use direct API calls for quick prototypes or app-specific integrations. MCP adds some overhead but pays off in maintainability.

### Q: What's the difference between MCP resources and tools?
**A:** Resources are read-only data sources (files, database records, API responses)—things the LLM can read. Tools are actions the LLM can execute (running queries, making API calls, sending emails)—things the LLM can do. Resources provide context, tools enable actions.

### Q: Can MCP servers be chained or composed?
**A:** Yes! An LLM can use multiple MCP servers simultaneously. For example, one server for database access, another for email, another for calendar. The LLM orchestrates which servers to use for each task. This composability is a key MCP advantage.

### Q: Is MCP production-ready?
**A:** MCP is relatively new (late 2024). It's backed by Anthropic and gaining adoption, but the ecosystem is still developing. Use it for new projects where standardization matters. For existing production systems, evaluate if migration benefits outweigh costs. Monitor the ecosystem's growth.

---

## Local Development Setup

### Q: Can I actually run useful models on my laptop?
**A:** Yes! Modern laptops can run 7B-13B parameter models quite well. 7B models (like Mistral 7B, Llama 3 8B) work on 16GB RAM. They're good for many tasks—summarization, simple coding, chat, classification. They're not as capable as GPT-4, but usable for development and many production use cases.

### Q: What's the difference between running models with Ollama vs LM Studio?
**A:** Ollama is CLI-first and developer-focused—great for scripting and automation. LM Studio has a GUI and is more user-friendly for non-technical users. Both run the same models (GGUF format). Use Ollama for development workflows, LM Studio for experimentation and team members who prefer GUIs.

### Q: Should I use GPU or CPU for local inference?
**A:** GPU is 5-10x faster but not required. For development, CPU inference works fine—you'll wait 2-5 seconds instead of subsecond responses. For production with high traffic, GPU is worth it. If you have an NVIDIA GPU (8GB+ VRAM), definitely use it. Apple Silicon Macs use unified memory well.

### Q: How do I choose between local models and API calls?
**A:** Local models for: Development/testing, privacy-sensitive data, high-volume with low complexity, offline scenarios. API calls for: Best quality needed, complex reasoning, don't want infrastructure burden, want latest models. Hybrid: Local for most tasks, API for complex ones.

### Q: What's quantization and should I use it?
**A:** Quantization reduces model precision (32-bit → 8-bit → 4-bit) to save memory and speed inference. You should use it for local models—4-bit or 8-bit quantized models run on consumer hardware with minimal quality loss. Q4 models are 4x smaller than full precision. Always start with quantized models locally.

### Q: How much does running local models actually save?
**A:** Potentially significant for high volume. If you make 100K API calls per month at $0.01 each = $1000. A local server might cost $200/month to run. But factor in: your time managing infrastructure, lower quality outputs, limited model selection. For small projects or privacy needs, local is great. For startups scaling fast, APIs are usually better initially.

---

## General LLM Questions

### Q: What's the difference between GPT-4, Claude, and Llama?
**A:** GPT-4 (OpenAI): Strong all-rounder, best for complex reasoning, large ecosystem. Claude (Anthropic): Long context windows (200K+), strong at analysis and writing, good safety. Llama (Meta): Open source, run locally or on your infrastructure, models from 7B to 70B+, customizable. Choose based on: cost, privacy needs, capability requirements, context length needs.

### Q: Will LLMs replace software engineers?
**A:** No, but they'll change the job. LLMs are tools that increase productivity—like IDEs and Stack Overflow before them. They handle boilerplate, speed up coding, help with debugging. But you still need engineers to: design systems, understand requirements, make architectural decisions, debug complex issues, ensure quality and security. LLMs make you more productive, not obsolete.

### Q: Should I fine-tune models or use RAG?
**A:** Start with RAG—it's faster, cheaper, and easier to update. Fine-tuning when: (1) You need specific style/tone consistently, (2) Your task is very domain-specific, (3) You have 1000+ high-quality examples, (4) You need behavior changes, not just knowledge. Often RAG + prompting solves 90% of use cases. Fine-tuning is for the last 10% of optimization.

### Q: What's the best way to learn about new LLM developments?
**A:** (1) Follow key researchers on Twitter/X, (2) Read papers on arxiv.org (especially the LLM category), (3) Join AI engineering communities (Discord, Reddit), (4) Build projects—hands-on is best, (5) Take courses (fast.ai, deeplearning.ai), (6) Read official docs for major providers, (7) Experiment with new models as they release. The field moves fast—stay curious.

### Q: How do I explain LLM limitations to non-technical stakeholders?
**A:** Use analogies: LLMs are like smart interns—knowledgeable but need supervision. They can hallucinate (make up facts confidently). They're probabilistic (same input ≠ same output). They don't "know" things, they predict patterns. They need clear instructions. They can't access real-time data unless connected to tools. Set realistic expectations upfront to avoid disappointment.

---

## Quick Decision Framework

**"Should I use an LLM for this?"**
- Can traditional code do it reliably? → Use traditional code
- Is it deterministic and rule-based? → Traditional code
- Do I need 100% accuracy? → Traditional code + LLM verification
- Does it involve natural language, reasoning, or creativity? → Probably LLM
- Is the data unstructured? → Probably LLM
- Do the rules change frequently? → LLM
- Still unsure? → Prototype both approaches

**"Which model should I use?"**
- Need best quality? → GPT-4 or Claude Opus
- Everyday tasks? → GPT-3.5 or Claude Sonnet
- Very long documents? → Claude (200K context)
- Privacy critical? → Local Llama model
- Need open source? → Llama or Mistral
- Cost very sensitive? → Smaller models or local
- Start with best model, optimize down as needed

**"How much will this cost?"**
1. Estimate tokens per request (input + output)
2. Multiply by requests per day
3. Check provider pricing (typically $0.01-0.10 per 1K tokens)
4. Factor in: caching (30-50% savings), model choice (10x variation)
5. Add buffer for development/testing
6. Monitor actual usage—estimates are often wrong

---

This Q&A covers the essential knowledge you need as a software engineer working with LLMs. Bookmark it and refer back when you encounter these questions in real projects!