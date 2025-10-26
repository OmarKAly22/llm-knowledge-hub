# 🐍 Python LLM Tutorial Series

> A comprehensive 12-week journey from Python basics to production-ready LLM applications

Part of the [LLM Knowledge Hub](https://github.com/OmarKAly22/llm-knowledge-hub) | [Live Site](https://llm-knowledge-hub.onrender.com/)

## 📚 Overview

This tutorial series takes you from Python fundamentals to building production-ready LLM applications. Each week focuses on practical, hands-on learning with real code examples, exercises, and mini-projects.

## 🎯 Who Is This For?

- **Software Engineers** transitioning to AI/LLM development
- **Python Developers** wanting to integrate LLMs into applications
- **Data Scientists** expanding into generative AI
- **Students** learning practical AI engineering

**Prerequisites:**
- Basic Python knowledge (functions, classes, loops)
- Familiarity with APIs and JSON
- Command line basics
- Curiosity and willingness to experiment!

## 📅 12-Week Curriculum

### **Phase 1: Foundations (Weeks 1-3)**
Build a solid foundation in LLM development basics.

| Week | Tutorial | Topics Covered | Project |
|------|----------|----------------|---------|
| 1 | [Python Basics for LLM Development](tutorials/01_python_basics_llm.py) | Environment setup, API keys, First LLM calls, Error handling | Smart Q&A System |
| 2 | Understanding LLM APIs | OpenAI, Anthropic, Streaming, Token management | API Comparison Tool |
| 3 | Prompt Engineering with Python | Templates, Few-shot, Chain-of-thought, Optimization | Prompt Library Manager |

### **Phase 2: Core Patterns (Weeks 4-6)**
Master the essential patterns for building AI applications.

| Week | Tutorial | Topics Covered | Project |
|------|----------|----------------|---------|
| 4 | Building Your First AI Agent | ReAct pattern, Tool calling, Agent loops, Decision making | Task Automation Agent |
| 5 | RAG Systems in Python | Vector DBs, Embeddings, Chunking, Retrieval | Document Q&A System |
| 6 | Memory Management | Conversation history, Context windows, State persistence | Contextual Chatbot |

### **Phase 3: Advanced Architectures (Weeks 7-9)**
Explore sophisticated patterns and production concerns.

| Week | Tutorial | Topics Covered | Project |
|------|----------|----------------|---------|
| 7 | Multi-Agent Systems | Sequential/Parallel execution, Communication, Consensus | Research Team Simulator |
| 8 | Tool Integration | Custom tools, API wrappers, File/DB operations | Swiss Army Agent |
| 9 | Production Patterns | Error handling, Rate limiting, Monitoring, Cost optimization | Production-Ready API |

### **Phase 4: Real-World Applications (Weeks 10-12)**
Build complete, practical applications.

| Week | Tutorial | Topics Covered | Project |
|------|----------|----------------|---------|
| 10 | Code Generation Agent | Code parsing, Test generation, Review automation | AI Pair Programmer |
| 11 | Customer Service Bot | Intent classification, Escalation, Multi-turn dialogue | Support Desk Assistant |
| 12 | Personal AI Assistant | Task scheduling, Email, Knowledge base, Voice | Personal Productivity AI |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/OmarKAly22/llm-knowledge-hub.git
cd llm-knowledge-hub/tutorials
```

### 2. Set Up Your Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 4. Start Learning!
```bash
# Run the first tutorial
python tutorials/01_python_basics_llm.py
```

## 📂 Repository Structure

```
tutorials/
├── README.md                          # This file
├── TUTORIAL_TEMPLATE.py              # Reusable template for all tutorials
├── requirements.txt                  # Python dependencies
├── .env.example                     # Example environment variables
├── 01_foundations/                  # Weeks 1-3
│   ├── 01_python_basics_llm.py
│   ├── 02_llm_apis.py
│   └── 03_prompt_engineering.py
├── 02_core_patterns/                # Weeks 4-6
│   ├── 04_first_agent.py
│   ├── 05_rag_system.py
│   └── 06_memory_management.py
├── 03_advanced/                     # Weeks 7-9
│   ├── 07_multi_agent.py
│   ├── 08_tool_integration.py
│   └── 09_production_patterns.py
├── 04_applications/                 # Weeks 10-12
│   ├── 10_code_agent.py
│   ├── 11_customer_service.py
│   └── 12_personal_assistant.py
├── projects/                        # Complete project code
│   ├── smart_qa_system/
│   ├── rag_document_chat/
│   └── ai_assistant/
├── data/                           # Sample data and outputs
├── prompts/                        # Prompt templates
└── utils/                          # Shared utilities

```

## 📖 Tutorial Format

Each tutorial follows a consistent structure:

1. **Conceptual Overview** - Theory and background
2. **Basic Implementation** - Simple, working examples
3. **Advanced Features** - Production-ready enhancements
4. **Hands-on Exercises** - Practice problems (with solutions)
5. **Mini-Project** - Combine concepts in a practical application
6. **Key Takeaways** - Summary and next steps

## 💻 Code Examples

### Quick Example: Your First LLM Call
```python
from dotenv import load_dotenv
import openai
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Make your first call
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, AI!"}]
)

print(response.choices[0].message.content)
```

### Building an Agent (Week 4 Preview)
```python
class SimpleAgent:
    def __init__(self, tools):
        self.tools = tools
        self.memory = []
    
    def think(self, objective):
        # Decide what to do next
        return self.llm.complete(f"Given objective: {objective}, what should I do?")
    
    def act(self, action):
        # Execute the chosen action
        tool = self.select_tool(action)
        return tool.execute()
    
    def observe(self, result):
        # Process the result
        self.memory.append(result)
        return self.llm.complete(f"Given result: {result}, what next?")
```

## 🎓 Learning Resources

### From the LLM Knowledge Hub
- [LLM Fundamentals Guide](https://llm-knowledge-hub.onrender.com/llm-guide)
- [Agentic AI Guide](https://llm-knowledge-hub.onrender.com/agentic-guide)
- [100+ Q&A](https://llm-knowledge-hub.onrender.com/llm-qa)
- [Interactive Diagrams](https://llm-knowledge-hub.onrender.com/diagrams)

### External Resources
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Documentation](https://docs.anthropic.com)
- [LangChain Python](https://python.langchain.com)
- [Vector Database Guides](https://www.pinecone.io/learn/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### How to Contribute a Tutorial
1. Use the `TUTORIAL_TEMPLATE.py` as your starting point
2. Follow the established format and style
3. Include working code examples
4. Add exercises with solutions
5. Test everything thoroughly
6. Submit a pull request

## 📊 Progress Tracking

Track your progress through the series:

- [ ] Week 1: Python Basics for LLM Development
- [ ] Week 2: Understanding LLM APIs
- [ ] Week 3: Prompt Engineering with Python
- [ ] Week 4: Building Your First AI Agent
- [ ] Week 5: RAG Systems in Python
- [ ] Week 6: Memory Management
- [ ] Week 7: Multi-Agent Systems
- [ ] Week 8: Tool Integration
- [ ] Week 9: Production Patterns
- [ ] Week 10: Code Generation Agent
- [ ] Week 11: Customer Service Bot
- [ ] Week 12: Personal AI Assistant

## ❓ FAQ

**Q: Do I need prior AI/ML experience?**
A: No! This series assumes Python knowledge but teaches AI/LLM concepts from scratch.

**Q: How much time should I dedicate per week?**
A: Plan for 3-5 hours per week: 1 hour for the tutorial, 2-3 hours for exercises and projects.

**Q: Can I skip around the tutorials?**
A: While possible, we recommend following the sequence as concepts build on each other.

**Q: Are the API keys free?**
A: Both OpenAI and Anthropic offer free tiers or credits for new users. Budget ~$10-20 for the entire series if heavily using the APIs.

**Q: What if I get stuck?**
A: Check the troubleshooting section in each tutorial, visit our [Q&A section](https://llm-knowledge-hub.onrender.com/llm-qa), or open an issue on GitHub.

## 📈 What You'll Achieve

By completing this 12-week series, you'll be able to:

✅ Build production-ready LLM applications  
✅ Create sophisticated AI agents  
✅ Implement RAG systems for knowledge retrieval  
✅ Design multi-agent architectures  
✅ Handle real-world challenges like rate limiting and cost optimization  
✅ Deploy AI solutions at scale  

## 🌟 Success Stories

*This section will feature projects and testimonials from learners who complete the series.*

## 📝 License

This tutorial series is part of the LLM Knowledge Hub and is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

Special thanks to:
- The OpenAI and Anthropic teams for their excellent APIs
- The open-source community for inspiration and tools
- All contributors to the LLM Knowledge Hub project

---

**Ready to start your journey?** Begin with [Tutorial 1: Python Basics for LLM Development](tutorials/01_python_basics_llm.py)! 🚀

*Happy Learning!* 🐍🤖
