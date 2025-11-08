import tiktoken
from typing import List, Dict


class TokenManager:
    """
    Manages token counting and cost calculation for different models.
    This is essential for any production LLM application.
    """

    def __init__(self, model: str = "gpt-4"):
        # Initialize the tokenizer for the specific model
        self.encoder = tiktoken.encoding_for_model(model)
        self.model = model

        # Define token limits for different models
        self.limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "claude-3": 200000,
        }

        # Pricing per 1000 tokens (as of 2024)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoder.encode(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a conversation (OpenAI format).
        Each message has overhead tokens for formatting.
        """
        tokens = 0
        for message in messages:
            tokens += 4  # Message formatting overhead
            for key, value in message.items():
                tokens += len(self.encoder.encode(str(value)))
        tokens += 2  # Reply priming tokens
        return tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a API call."""
        rates = self.pricing.get(self.model, self.pricing["gpt-4"])
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        return input_cost + output_cost

    def can_fit_response(
        self, messages: List[Dict], response_tokens: int = 500
    ) -> bool:
        """Check if there's room for a response."""
        current_tokens = self.count_messages(messages)
        max_tokens = self.limits.get(self.model, 4096)
        return (current_tokens + response_tokens) <= max_tokens


# Example usage
manager = TokenManager("gpt-4")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
]
tokens_used = manager.count_messages(messages)
cost = manager.estimate_cost(tokens_used, 200)  # Expecting 200 token response
print(f"Tokens used: {tokens_used}")
print(f"Estimated cost: ${cost:.4f}")
print(f"Can fit 500-token response: {manager.can_fit_response(messages)}")
