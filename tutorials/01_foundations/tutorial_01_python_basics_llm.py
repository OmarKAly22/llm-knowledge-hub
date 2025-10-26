"""
================================================================================
Tutorial 01: Python Basics for LLM Development
================================================================================
Part of the LLM Knowledge Hub Python Tutorial Series
GitHub: https://github.com/OmarKAly22/llm-knowledge-hub
Website: https://llm-knowledge-hub.onrender.com/

Author: LLM Knowledge Hub Contributors
Date: January 2025
Duration: 45-60 minutes
================================================================================

📚 LEARNING OBJECTIVES
----------------------
By the end of this tutorial, you will:
1. Set up a professional Python environment for LLM development
2. Understand essential Python patterns for AI/ML work
3. Master API key management and environment configuration
4. Build your first LLM-powered Python application

📋 PREREQUISITES
---------------
- Python 3.8+ installed (check with: python --version)
- Basic Python knowledge (variables, functions, classes)
- A code editor (VS Code recommended)
- OpenAI or Anthropic API key (we'll show you how to get one)

📦 REQUIRED PACKAGES
-------------------
pip install python-dotenv requests openai anthropic rich

================================================================================
SECTION 1: CONCEPTUAL OVERVIEW
================================================================================
"""

# Working with LLMs in Python requires understanding several key concepts:
# 1. Asynchronous programming for API calls
# 2. Environment management for sensitive data
# 3. Error handling for network operations
# 4. Token management and cost optimization

"""
Key Concepts:
-------------
• API Clients: Python libraries that handle LLM communication
• Environment Variables: Secure storage for API keys and configuration
• Token Management: Understanding and optimizing LLM usage costs
• Streaming Responses: Handling real-time LLM output
• Error Resilience: Dealing with rate limits and network issues

Real-World Applications:
-----------------------
• Chatbots and conversational AI
• Content generation and summarization
• Code analysis and generation
• Data extraction and processing
• Automated customer support
"""

# ================================================================================
# SECTION 2: BASIC IMPLEMENTATION
# ================================================================================

print("=" * 80)
print("SECTION 2: SETTING UP YOUR LLM DEVELOPMENT ENVIRONMENT")
print("=" * 80)

import os
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

# Step 1: Environment Setup
# --------------------------
# Create a .env file in your project root (never commit this to git!)

def setup_environment():
    """
    Sets up the development environment for LLM work.
    Creates necessary files and directories.
    """
    # Create project structure
    project_dirs = [
        "config",
        "data",
        "prompts",
        "outputs",
        "logs"
    ]
    
    for dir_name in project_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_template = """# LLM API Keys (Keep Secret!)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Configuration
MODEL_NAME=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.7

# Logging
LOG_LEVEL=INFO
"""
        env_file.write_text(env_template)
        print("✅ Created .env template file")
        
        # Also create .gitignore to protect secrets
        gitignore = Path(".gitignore")
        if not gitignore.exists():
            gitignore.write_text(".env\n*.log\ndata/\noutputs/\n__pycache__/\n")
            print("✅ Created .gitignore file")
    
    print("\n📝 Next steps:")
    print("1. Add your API keys to .env file")
    print("2. Never commit .env to version control")
    print("3. Install required packages: pip install python-dotenv openai anthropic")

# Run setup
setup_environment()

# Step 2: Loading Environment Variables Safely
# --------------------------------------------
from dotenv import load_dotenv

class ConfigManager:
    """
    Manages configuration and environment variables safely.
    
    This class demonstrates:
    - Secure API key management
    - Configuration validation
    - Default value handling
    """
    
    def __init__(self, env_file: str = ".env"):
        """Initialize configuration manager."""
        load_dotenv(env_file)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model_name": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            "max_tokens": int(os.getenv("MAX_TOKENS", "1000")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        if not (self.config["openai_api_key"] or self.config["anthropic_api_key"]):
            print("⚠️  Warning: No API keys found in environment!")
            print("Please add your API keys to the .env file")
            print("\nHow to get API keys:")
            print("- OpenAI: https://platform.openai.com/api-keys")
            print("- Anthropic: https://console.anthropic.com/account/keys")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

# Example usage
config = ConfigManager()
print(f"\nConfiguration loaded:")
print(f"- Model: {config.get('model_name')}")
print(f"- Max Tokens: {config.get('max_tokens')}")
print(f"- Temperature: {config.get('temperature')}")

# Step 3: Your First LLM Call
# ---------------------------

class SimpleLLMClient:
    """
    A simple, clean interface for LLM interactions.
    Supports both OpenAI and Anthropic.
    """
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize the LLM client.
        
        Args:
            provider: Either "openai" or "anthropic"
        """
        self.config = ConfigManager()
        self.provider = provider
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate LLM client."""
        if self.provider == "openai":
            import openai
            openai.api_key = self.config.get("openai_api_key")
            self.client = openai
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.config.get("anthropic_api_key")
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Get a completion from the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            The LLM's response as a string
        """
        try:
            if self.provider == "openai":
                response = self.client.ChatCompletion.create(
                    model=kwargs.get("model", self.config.get("model_name")),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.config.get("temperature")),
                    max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens"))
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.completions.create(
                    model=kwargs.get("model", "claude-2"),
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7)
                )
                return response.completion
                
        except Exception as e:
            print(f"❌ Error calling {self.provider}: {e}")
            return None

# Example: Your first LLM interaction
print("\n" + "=" * 80)
print("YOUR FIRST LLM INTERACTION")
print("=" * 80)

# Uncomment when you have API keys:
# client = SimpleLLMClient(provider="openai")
# response = client.complete("Explain Python decorators in 2 sentences")
# print(f"LLM Response: {response}")

# ================================================================================
# SECTION 3: ADVANCED FEATURES
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 3: PRODUCTION-READY PATTERNS")
print("=" * 80)

import time
import logging
from functools import wraps
from typing import Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionLLMClient(SimpleLLMClient):
    """
    Production-ready LLM client with advanced features.
    
    Additions:
    - Automatic retries with exponential backoff
    - Response caching
    - Token counting and cost tracking
    - Streaming support
    - Request/response logging
    """
    
    def __init__(self, provider: str = "openai", cache_enabled: bool = True):
        """Initialize with production features."""
        super().__init__(provider)
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
    def retry_with_backoff(max_retries: int = 3):
        """Decorator for automatic retries with exponential backoff."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        wait_time = 2 ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {max_retries} attempts failed")
                            raise
                return None
            return wrapper
        return decorator
    
    @retry_with_backoff(max_retries=3)
    def complete_with_retry(self, prompt: str, **kwargs) -> str:
        """Complete with automatic retry on failure."""
        # Check cache first
        cache_key = f"{prompt}_{kwargs}"
        if self.cache_enabled and cache_key in self.cache:
            logger.info("Cache hit! Returning cached response")
            return self.cache[cache_key]
        
        # Log the request
        logger.info(f"Sending request to {self.provider}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        
        # Make the actual call
        response = super().complete(prompt, **kwargs)
        
        # Cache the response
        if self.cache_enabled and response:
            self.cache[cache_key] = response
            logger.info("Response cached for future use")
        
        # Track usage (simplified - real implementation would parse response)
        self.total_tokens_used += kwargs.get("max_tokens", 100)
        self._estimate_cost()
        
        return response
    
    def _estimate_cost(self):
        """Estimate the cost based on tokens used."""
        # Rough estimates (check current pricing)
        cost_per_1k_tokens = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "claude-2": 0.01
        }
        model = self.config.get("model_name")
        rate = cost_per_1k_tokens.get(model, 0.002)
        self.total_cost = (self.total_tokens_used / 1000) * rate
        logger.info(f"Estimated cost so far: ${self.total_cost:.4f}")
    
    def stream_complete(self, prompt: str, **kwargs):
        """
        Stream completions token by token.
        Useful for real-time interfaces.
        """
        if self.provider != "openai":
            logger.warning("Streaming only supported for OpenAI currently")
            yield self.complete(prompt, **kwargs)
            return
        
        try:
            kwargs["stream"] = True
            response = self.client.ChatCompletion.create(
                model=kwargs.get("model", self.config.get("model_name")),
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            for chunk in response:
                if chunk.choices[0].delta.get("content"):
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield None

# Example usage
prod_client = ProductionLLMClient(provider="openai")

# With retry and caching
# response = prod_client.complete_with_retry(
#     "What are the key principles of clean code?",
#     max_tokens=200
# )

# ================================================================================
# SECTION 4: HANDS-ON EXERCISES
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 4: EXERCISES")
print("=" * 80)

"""
EXERCISE 1: Build a Prompt Template System
------------------------------------------
Task: Create a class that manages prompt templates with variable substitution.

Requirements:
- Load templates from files
- Support variable substitution with {variable_name}
- Validate that all required variables are provided
- Support default values for optional variables

Hints:
- Use str.format() or Template from string module
- Store templates in a prompts/ directory
- Consider using YAML or JSON for template metadata

Expected Output:
template = PromptTemplate("greet_user")
result = template.render(name="Alice", language="Python")
# "Hello Alice! Welcome to Python programming."
"""

class PromptTemplate:
    """Your implementation here."""
    def __init__(self, template_name: str):
        # TODO: Load template from file
        pass
    
    def render(self, **variables) -> str:
        # TODO: Substitute variables and return formatted prompt
        pass

def exercise_1_solution():
    """
    Solution for Exercise 1.
    """
    class PromptTemplate:
        def __init__(self, template_name: str):
            self.template_name = template_name
            self.template = self._load_template()
        
        def _load_template(self) -> str:
            template_file = Path(f"prompts/{self.template_name}.txt")
            if template_file.exists():
                return template_file.read_text()
            else:
                # Default templates
                templates = {
                    "greet_user": "Hello {name}! Welcome to {language} programming.",
                    "summarize": "Summarize the following text in {sentences} sentences:\n\n{text}",
                    "translate": "Translate the following from {source_lang} to {target_lang}:\n\n{text}"
                }
                return templates.get(self.template_name, "")
        
        def render(self, **variables) -> str:
            try:
                return self.template.format(**variables)
            except KeyError as e:
                print(f"Missing required variable: {e}")
                return None
    
    # Test the solution
    template = PromptTemplate("greet_user")
    result = template.render(name="Alice", language="Python")
    print(f"Exercise 1 Result: {result}")

"""
EXERCISE 2: Token Counter and Cost Calculator
---------------------------------------------
Task: Build a utility that counts tokens and estimates costs for different models.

Requirements:
- Count tokens in a string (approximate: 1 token ≈ 4 characters)
- Calculate cost for different models
- Track cumulative usage across multiple calls
- Provide usage reports

Note: Real token counting requires tiktoken library, but we'll approximate.
"""

class TokenCounter:
    """Your implementation here."""
    def __init__(self):
        # TODO: Initialize counters
        pass
    
    def count_tokens(self, text: str) -> int:
        # TODO: Count tokens (approximate)
        pass
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        # TODO: Calculate cost based on model
        pass

def exercise_2_solution():
    """Solution for Exercise 2."""
    class TokenCounter:
        def __init__(self):
            self.total_tokens = 0
            self.total_cost = 0.0
            self.usage_by_model = {}
            
            # Pricing per 1K tokens (example rates)
            self.pricing = {
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "claude-2": {"input": 0.01, "output": 0.02}
            }
        
        def count_tokens(self, text: str) -> int:
            # Rough approximation: ~4 characters per token
            return len(text) // 4
        
        def calculate_cost(self, tokens: int, model: str, is_output: bool = False) -> float:
            if model not in self.pricing:
                model = "gpt-3.5-turbo"  # Default
            
            token_type = "output" if is_output else "input"
            rate = self.pricing[model][token_type]
            cost = (tokens / 1000) * rate
            
            # Track usage
            self.total_tokens += tokens
            self.total_cost += cost
            
            if model not in self.usage_by_model:
                self.usage_by_model[model] = {"tokens": 0, "cost": 0.0}
            self.usage_by_model[model]["tokens"] += tokens
            self.usage_by_model[model]["cost"] += cost
            
            return cost
        
        def get_report(self) -> str:
            report = f"=== Token Usage Report ===\n"
            report += f"Total Tokens: {self.total_tokens:,}\n"
            report += f"Total Cost: ${self.total_cost:.4f}\n"
            report += "\nBy Model:\n"
            for model, usage in self.usage_by_model.items():
                report += f"  {model}: {usage['tokens']:,} tokens (${usage['cost']:.4f})\n"
            return report
    
    # Test the solution
    counter = TokenCounter()
    text = "This is a sample text for token counting and cost calculation."
    tokens = counter.count_tokens(text)
    cost = counter.calculate_cost(tokens, "gpt-3.5-turbo")
    print(f"Exercise 2: {tokens} tokens, ${cost:.4f} cost")
    print(counter.get_report())

"""
EXERCISE 3: Build a Conversation Manager [Challenge]
---------------------------------------------------
Task: Create a system that manages multi-turn conversations with context.

This exercise combines:
- Message history management
- Context window management (trim when too long)
- System prompt handling
- Conversation serialization/deserialization

Requirements:
- Store conversation history
- Automatically manage context length
- Support different message roles (system, user, assistant)
- Save/load conversations to/from JSON
"""

class ConversationManager:
    """Your implementation here."""
    def __init__(self, system_prompt: str = None, max_context_length: int = 4000):
        # TODO: Initialize conversation
        pass
    
    def add_message(self, role: str, content: str):
        # TODO: Add message to history
        pass
    
    def get_context(self) -> List[Dict[str, str]]:
        # TODO: Return messages within context window
        pass
    
    def save(self, filename: str):
        # TODO: Save conversation to JSON
        pass

def exercise_3_solution():
    """Solution for Exercise 3 (Challenge)."""
    class ConversationManager:
        def __init__(self, system_prompt: str = None, max_context_length: int = 4000):
            self.max_context_length = max_context_length
            self.messages = []
            
            if system_prompt:
                self.add_message("system", system_prompt)
        
        def add_message(self, role: str, content: str):
            """Add a message to the conversation."""
            if role not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {role}")
            
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
        
        def get_context(self) -> List[Dict[str, str]]:
            """Get messages that fit within context window."""
            # Always include system message if present
            context = []
            total_length = 0
            
            # Add system message first
            for msg in self.messages:
                if msg["role"] == "system":
                    context.append({"role": msg["role"], "content": msg["content"]})
                    total_length += len(msg["content"])
                    break
            
            # Add recent messages in reverse order
            for msg in reversed(self.messages):
                if msg["role"] != "system":
                    msg_length = len(msg["content"])
                    if total_length + msg_length < self.max_context_length:
                        context.insert(1, {"role": msg["role"], "content": msg["content"]})
                        total_length += msg_length
                    else:
                        break
            
            return context
        
        def save(self, filename: str):
            """Save conversation to JSON."""
            filepath = Path(f"data/{filename}")
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.messages, f, indent=2)
            print(f"Conversation saved to {filepath}")
        
        def load(self, filename: str):
            """Load conversation from JSON."""
            filepath = Path(f"data/{filename}")
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.messages = json.load(f)
                print(f"Loaded {len(self.messages)} messages")
        
        def summarize(self) -> str:
            """Get a summary of the conversation."""
            summary = f"Conversation with {len(self.messages)} messages\n"
            summary += f"Roles: {set(m['role'] for m in self.messages)}\n"
            summary += f"Total length: {sum(len(m['content']) for m in self.messages)} chars"
            return summary
    
    # Test the solution
    conv = ConversationManager(system_prompt="You are a helpful Python tutor.")
    conv.add_message("user", "What are Python decorators?")
    conv.add_message("assistant", "Decorators are functions that modify other functions...")
    conv.add_message("user", "Can you show me an example?")
    print(f"Exercise 3: {conv.summarize()}")
    context = conv.get_context()
    print(f"Context has {len(context)} messages")

# ================================================================================
# SECTION 5: MINI-PROJECT
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 5: MINI-PROJECT - Build a Smart Q&A System")
print("=" * 80)

"""
PROJECT: Smart Q&A System with Context
======================================
Build a question-answering system that maintains context across questions.

This project will help you:
- Apply prompt engineering techniques
- Manage conversation context
- Handle different types of questions
- Integrate with real LLMs

Project Requirements:
--------------------
1. Accept questions from the user
2. Maintain conversation context
3. Detect question types (factual, opinion, follow-up)
4. Generate appropriate prompts for each type
5. Handle errors gracefully
6. Provide usage statistics

Bonus Challenges:
----------------
• Add a knowledge base for factual questions
• Implement response caching
• Add sentiment analysis to responses
• Export conversations to markdown
"""

class SmartQASystem:
    """
    A smart Q&A system that maintains context and handles various question types.
    """
    
    def __init__(self, llm_provider: str = "openai"):
        """Initialize the Q&A system."""
        self.llm_client = ProductionLLMClient(provider=llm_provider)
        self.conversation = ConversationManager(
            system_prompt="""You are a helpful assistant specialized in Python and LLM development.
            Provide clear, concise answers with code examples when appropriate."""
        )
        self.token_counter = TokenCounter()
        self.question_count = 0
        
    def detect_question_type(self, question: str) -> str:
        """
        Detect the type of question being asked.
        
        Returns:
            One of: "factual", "opinion", "follow_up", "code", "explanation"
        """
        question_lower = question.lower()
        
        # Simple rule-based detection (could use LLM for this too!)
        if any(word in question_lower for word in ["what is", "define", "how many", "when"]):
            return "factual"
        elif any(word in question_lower for word in ["should", "better", "recommend", "opinion"]):
            return "opinion"
        elif any(word in question_lower for word in ["code", "example", "implement", "function"]):
            return "code"
        elif any(word in question_lower for word in ["why", "how does", "explain"]):
            return "explanation"
        elif len(self.conversation.messages) > 1 and len(question.split()) < 10:
            return "follow_up"
        else:
            return "general"
    
    def generate_prompt(self, question: str, question_type: str) -> str:
        """
        Generate an optimized prompt based on question type.
        """
        prompts = {
            "factual": f"Provide a clear, factual answer to: {question}\nBe concise and accurate.",
            "opinion": f"Provide a balanced perspective on: {question}\nConsider multiple viewpoints.",
            "code": f"Provide a Python code example for: {question}\nInclude comments and explanation.",
            "explanation": f"Explain in detail: {question}\nUse examples and analogies if helpful.",
            "follow_up": f"{question}",  # Use context from conversation
            "general": f"{question}"
        }
        
        return prompts.get(question_type, question)
    
    def ask(self, question: str) -> str:
        """
        Process a question and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            The system's answer
        """
        try:
            # Track the question
            self.question_count += 1
            self.conversation.add_message("user", question)
            
            # Detect question type
            q_type = self.detect_question_type(question)
            logger.info(f"Question #{self.question_count} detected as: {q_type}")
            
            # Generate optimized prompt
            prompt = self.generate_prompt(question, q_type)
            
            # Get context for follow-up questions
            if q_type == "follow_up":
                context = self.conversation.get_context()
                # Format context for LLM
                context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
                prompt = f"Given this conversation:\n{context_str}\n\nAnswer: {question}"
            
            # Get response from LLM
            response = self.llm_client.complete_with_retry(prompt, max_tokens=500)
            
            if response:
                # Track the response
                self.conversation.add_message("assistant", response)
                tokens = self.token_counter.count_tokens(prompt + response)
                cost = self.token_counter.calculate_cost(tokens, "gpt-3.5-turbo")
                
                return response
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error in ask(): {e}")
            return f"An error occurred: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "questions_asked": self.question_count,
            "messages_in_context": len(self.conversation.messages),
            "total_tokens": self.token_counter.total_tokens,
            "estimated_cost": f"${self.token_counter.total_cost:.4f}",
            "cache_size": len(self.llm_client.cache) if hasattr(self.llm_client, 'cache') else 0
        }
    
    def export_conversation(self, filename: str):
        """Export the conversation to markdown."""
        filepath = Path(f"outputs/{filename}")
        filepath.parent.mkdir(exist_ok=True)
        
        markdown_content = f"# Q&A Session\n\n"
        markdown_content += f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n"
        markdown_content += f"Questions: {self.question_count}\n\n"
        markdown_content += "---\n\n"
        
        for msg in self.conversation.messages:
            if msg["role"] == "user":
                markdown_content += f"## Q: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                markdown_content += f"**A:** {msg['content']}\n\n"
                markdown_content += "---\n\n"
        
        filepath.write_text(markdown_content)
        print(f"Conversation exported to {filepath}")

# Example usage (uncomment when you have API keys):
"""
qa_system = SmartQASystem(llm_provider="openai")

# Ask questions
response1 = qa_system.ask("What are Python decorators?")
print(f"Answer 1: {response1}\n")

response2 = qa_system.ask("Can you show me a simple example?")
print(f"Answer 2: {response2}\n")

response3 = qa_system.ask("How do they compare to Java annotations?")
print(f"Answer 3: {response3}\n")

# Get statistics
stats = qa_system.get_stats()
print(f"Session Stats: {stats}")

# Export conversation
qa_system.export_conversation("python_decorators_qa.md")
"""

# ================================================================================
# SECTION 6: KEY TAKEAWAYS
# ================================================================================

"""
🎯 KEY TAKEAWAYS
================

✅ What You Learned:
--------------------
1. How to set up a professional Python environment for LLM development
2. Secure API key management using environment variables
3. Building robust LLM clients with error handling and retries
4. Managing conversation context and message history
5. Implementing production patterns like caching and token counting

📚 Further Reading:
------------------
- LLM Guide: https://llm-knowledge-hub.onrender.com/llm-guide
- OpenAI Documentation: https://platform.openai.com/docs
- Anthropic Documentation: https://docs.anthropic.com
- Python Async Programming: https://docs.python.org/3/library/asyncio.html

🚀 Next Steps:
-------------
- Complete all three exercises
- Build and extend the Smart Q&A System project
- Experiment with different LLM providers
- Move on to Tutorial #2: Understanding LLM APIs in Python

💡 Pro Tips:
-----------
• Always use environment variables for API keys - never hardcode them
• Implement retry logic for production systems - APIs can be unreliable
• Cache responses when possible to reduce costs and latency
• Monitor token usage to control costs
• Use streaming for real-time interfaces
• Test with smaller models first, then scale to larger ones

================================================================================
"""

# ================================================================================
# APPENDIX: COMMON ISSUES AND SOLUTIONS
# ================================================================================

"""
TROUBLESHOOTING GUIDE
====================

Issue 1: "API key not found" Error
----------------------------------
Symptom: Error message about missing API key
Solution: 
1. Check that .env file exists in project root
2. Verify API key is correctly formatted (no extra spaces)
3. Ensure python-dotenv is installed: pip install python-dotenv
4. Call load_dotenv() before accessing environment variables
Prevention: Always validate configuration on startup

Issue 2: Rate Limiting Errors
-----------------------------
Symptom: "Rate limit exceeded" or 429 status code
Solution:
1. Implement exponential backoff retry logic
2. Add delays between requests: time.sleep(1)
3. Use different API keys for development and production
4. Consider upgrading your API plan
Prevention: Track request rate and implement queuing

Issue 3: Timeout or Connection Errors
-------------------------------------
Symptom: Requests hang or timeout
Solution:
1. Set explicit timeouts: timeout=30
2. Implement retry logic with backoff
3. Check your internet connection
4. Verify API service status
Prevention: Always set timeouts and handle network errors

Issue 4: Unexpected Token Costs
-------------------------------
Symptom: Higher than expected API bills
Solution:
1. Implement token counting before sending requests
2. Set max_tokens parameter appropriately
3. Cache responses to avoid duplicate calls
4. Use smaller models for development
Prevention: Monitor usage and set up billing alerts

Issue 5: Context Length Exceeded
--------------------------------
Symptom: "Context length exceeded" error
Solution:
1. Implement conversation pruning
2. Summarize older messages
3. Use a sliding window for context
4. Choose models with larger context windows
Prevention: Track context size before each request

Need Help?
----------
- Check the Q&A section: https://llm-knowledge-hub.onrender.com/llm-qa
- Open an issue: https://github.com/OmarKAly22/llm-knowledge-hub/issues
- Read the guides: https://llm-knowledge-hub.onrender.com/guides
- Join the community discussions on GitHub

================================================================================
END OF TUTORIAL 01
================================================================================
"""

# Run the exercise solutions to verify everything works
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXERCISE SOLUTIONS FOR VERIFICATION")
    print("=" * 80)
    
    print("\nExercise 1: Prompt Templates")
    exercise_1_solution()
    
    print("\nExercise 2: Token Counter")
    exercise_2_solution()
    
    print("\nExercise 3: Conversation Manager")
    exercise_3_solution()
    
    print("\n✅ All exercises completed successfully!")
    print("\n📚 Remember to:")
    print("1. Add your API keys to .env file")
    print("2. Try building the Smart Q&A System project")
    print("3. Experiment with different prompts and models")
    print("\nHappy learning! 🚀")
