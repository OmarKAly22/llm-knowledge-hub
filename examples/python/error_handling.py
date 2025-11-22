#!/usr/bin/env python3
"""
AI Error Handling System
Production-ready error handling with retry strategies, circuit breakers,
fallbacks, and intelligent recovery for resilient AI applications.
"""
import asyncio
import time
import random
import traceback
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ============== Error Types and Classification ==============
class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
class ErrorCategory(Enum):
    """Categories of errors in AI systems."""
    TRANSIENT = "transient"  # Temporary, can retry
    PERMANENT = "permanent"  # Won't resolve, need fallback
    DEGRADABLE = "degradable"  # Can provide partial service
    CRITICAL = "critical"  # System failure, need immediate action

@dataclass
class AIError(Exception):
    """Base exception for AI system errors."""
    message: str
    error_code: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.error_code}: {self.message}"
    
    def should_retry(self) -> bool:
        """Check if error should be retried."""
        return self.category == ErrorCategory.TRANSIENT and self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry counter."""
        self.retry_count += 1

# Specific error types
class LLMError(AIError):
    """LLM-specific errors."""
    pass

class RateLimitError(AIError):
    """Rate limiting errors."""
    def __init__(self, message: str, retry_after: int = 60, **kwargs):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TRANSIENT,
            **kwargs
        )
        self.retry_after = retry_after

class TokenLimitError(AIError):
    """Token limit exceeded."""
    def __init__(self, message: str, tokens_used: int, **kwargs):
        super().__init__(
            message=message,
            error_code="TOKEN_LIMIT",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DEGRADABLE,
            **kwargs
        )
        self.tokens_used = tokens_used

class ValidationError(AIError):
    """Input/output validation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.PERMANENT,
            **kwargs
        )

class NetworkError(AIError):
    """Network-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="NETWORK",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRANSIENT,
            **kwargs
        )

# ============== Error Detection and Validation ==============
class ResponseValidator:
    """Validates AI responses for errors."""
    
    def __init__(self):
        self.validation_rules = {
            "format": self._validate_format,
            "content": self._validate_content,
            "safety": self._validate_safety,
            "consistency": self._validate_consistency
        }
    
    def validate(self, response: Any, expected_schema: Dict[str, Any] = None) -> bool:
        """Validate response against rules."""
        try:
            for rule_name, rule_func in self.validation_rules.items():
                if not rule_func(response, expected_schema):
                    raise ValidationError(f"Validation failed: {rule_name}")
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise
    
    def _validate_format(self, response: Any, schema: Dict[str, Any]) -> bool:
        """Check response format."""
        if schema and isinstance(response, dict):
            required_keys = schema.get("required", [])
            return all(key in response for key in required_keys)
        return True
    
    def _validate_content(self, response: Any, schema: Dict[str, Any]) -> bool:
        """Check content validity."""
        if isinstance(response, str):
            # Check for common LLM failure patterns
            failure_patterns = [
                "I cannot", "I'm unable to", "Error:", "Exception:",
                "undefined", "null", "NaN"
            ]
            return not any(pattern in response for pattern in failure_patterns)
        return True
    
    def _validate_safety(self, response: Any, schema: Dict[str, Any]) -> bool:
        """Check for safety issues."""
        if isinstance(response, str):
            # Check for potential harmful content
            unsafe_patterns = ["<script>", "DROP TABLE", "../../", "__proto__"]
            return not any(pattern in response for pattern in unsafe_patterns)
        return True
    
    def _validate_consistency(self, response: Any, schema: Dict[str, Any]) -> bool:
        """Check logical consistency."""
        # Check if response is logically consistent
        if isinstance(response, dict):
            # Example: check if sum equals total
            if "items" in response and "total" in response:
                calculated_total = sum(item.get("value", 0) for item in response["items"])
                return abs(calculated_total - response["total"]) < 0.01
        return True

class AnomalyDetector:
    """Detects anomalies in AI behavior."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.thresholds = {
            "latency": {"mean": 2.0, "std_dev": 1.0},
            "token_usage": {"mean": 1000, "std_dev": 500},
            "error_rate": {"threshold": 0.1}
        }
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        self.metrics_history[metric_name].append(value)
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> bool:
        """Detect if current value is anomalous."""
        history = self.metrics_history[metric_name]
        
        if len(history) < 10:  # Not enough data
            return False
        
        # Calculate statistics
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std_dev = variance ** 0.5
        
        # Check if current value is outside normal range
        z_score = abs((current_value - mean) / std_dev) if std_dev > 0 else 0
        
        return z_score > 3  # More than 3 standard deviations

# ============== Retry Strategies ==============
class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        pass
    
    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if should retry."""
        pass

class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff with jitter."""
    
    def __init__(self, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 max_attempts: int = 5,
                 jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if should retry based on error type and attempts."""
        if attempt >= self.max_attempts:
            return False
        
        # Retry transient errors
        if isinstance(error, AIError):
            return error.should_retry()
        
        # Retry specific exceptions
        retryable_exceptions = [
            ConnectionError, TimeoutError, asyncio.TimeoutError
        ]
        return any(isinstance(error, exc) for exc in retryable_exceptions)

class AdaptiveRetry(RetryStrategy):
    """Adaptive retry based on success rate."""
    
    def __init__(self):
        self.success_history = deque(maxlen=100)
        self.base_delay = 1.0
        self.max_attempts = 5
    
    def get_delay(self, attempt: int) -> float:
        """Adapt delay based on recent success rate."""
        if not self.success_history:
            return self.base_delay
        
        success_rate = sum(self.success_history) / len(self.success_history)
        
        # Adjust delay based on success rate
        if success_rate < 0.3:
            # Low success rate, increase delay
            return self.base_delay * (2 ** attempt) * 2
        elif success_rate < 0.7:
            # Medium success rate, normal delay
            return self.base_delay * (2 ** attempt)
        else:
            # High success rate, reduce delay
            return self.base_delay * (1.5 ** attempt)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Adaptively decide whether to retry."""
        if attempt >= self.max_attempts:
            return False
        
        # Record failure
        self.success_history.append(False)
        
        # Don't retry if success rate is too low
        if len(self.success_history) >= 10:
            recent_success_rate = sum(list(self.success_history)[-10:]) / 10
            if recent_success_rate < 0.1:
                return False
        
        return isinstance(error, (AIError, ConnectionError, TimeoutError))
    
    def record_success(self):
        """Record successful attempt."""
        self.success_history.append(True)

# ============== Circuit Breaker ==============
class CircuitBreakerState(Enum):
    """States of circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.stats["total_calls"] += 1
        
        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                self.stats["rejected_calls"] += 1
                raise AIError(
                    message="Circuit breaker is OPEN",
                    error_code="CIRCUIT_OPEN",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.TRANSIENT
                )
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.stats["successful_calls"] += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to fully close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED")
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call."""
        self.stats["failed_calls"] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker reopened due to failure in HALF_OPEN state")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "stats": self.stats
        }

# ============== Fallback Handlers ==============
class FallbackHandler:
    """Manages fallback strategies for failures."""
    
    def __init__(self):
        self.fallback_chains = {}
        self.cache = {}  # Simple cache for fallback responses
        
    def register_fallback(self, 
                          operation: str,
                          fallbacks: List[Callable],
                          cache_duration: int = 300):
        """Register fallback chain for an operation."""
        self.fallback_chains[operation] = {
            "fallbacks": fallbacks,
            "cache_duration": cache_duration
        }
    
    async def execute_with_fallback(self,
                                   operation: str,
                                   primary: Callable,
                                   *args,
                                   **kwargs) -> Any:
        """Execute operation with fallback chain."""
        # Try primary operation
        try:
            result = await primary(*args, **kwargs)
            # Cache successful result
            self._cache_result(operation, result)
            return result
        except Exception as primary_error:
            logger.warning(f"Primary operation failed: {primary_error}")
            
            # Try fallbacks
            if operation in self.fallback_chains:
                for i, fallback in enumerate(self.fallback_chains[operation]["fallbacks"]):
                    try:
                        logger.info(f"Attempting fallback {i+1} for {operation}")
                        result = await fallback(*args, **kwargs)
                        return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {i+1} failed: {fallback_error}")
                        continue
            
            # Try cache as last resort
            cached = self._get_cached_result(operation)
            if cached is not None:
                logger.info(f"Returning cached result for {operation}")
                return cached
            
            # All fallbacks failed
            raise AIError(
                message=f"All fallbacks failed for {operation}",
                error_code="FALLBACK_EXHAUSTED",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PERMANENT,
                context={"original_error": str(primary_error)}
            )
    
    def _cache_result(self, operation: str, result: Any):
        """Cache a successful result."""
        self.cache[operation] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def _get_cached_result(self, operation: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if operation in self.cache:
            cached = self.cache[operation]
            duration = self.fallback_chains.get(operation, {}).get("cache_duration", 300)
            
            if time.time() - cached["timestamp"] < duration:
                return cached["result"]
        
        return None

# ============== Error Recovery Orchestrator ==============
class ErrorRecoveryOrchestrator:
    """Orchestrates error recovery strategies."""
    
    def __init__(self):
        self.retry_strategy = ExponentialBackoffRetry()
        self.circuit_breakers = {}
        self.fallback_handler = FallbackHandler()
        self.validator = ResponseValidator()
        self.anomaly_detector = AnomalyDetector()
        self.error_history = deque(maxlen=1000)
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "recovered_requests": 0,
            "fallback_used": 0
        }
    
    async def execute_with_resilience(self,
                                     operation_name: str,
                                     operation: Callable,
                                     *args,
                                     fallbacks: List[Callable] = None,
                                     validate_response: bool = True,
                                     expected_schema: Dict[str, Any] = None,
                                     **kwargs) -> Any:
        """
        Execute operation with comprehensive error handling.
        """
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        # Get or create circuit breaker for this operation
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[operation_name]
        
        # Register fallbacks if provided
        if fallbacks:
            self.fallback_handler.register_fallback(operation_name, fallbacks)
        
        attempt = 0
        last_error = None
        
        while attempt < self.retry_strategy.max_attempts:
            try:
                # Execute with circuit breaker
                result = await circuit_breaker.call(
                    self._execute_operation,
                    operation_name,
                    operation,
                    *args,
                    **kwargs
                )
                
                # Validate response
                if validate_response:
                    self.validator.validate(result, expected_schema)
                
                # Check for anomalies
                latency = time.time() - start_time
                self.anomaly_detector.record_metric(f"{operation_name}_latency", latency)
                
                if self.anomaly_detector.detect_anomaly(f"{operation_name}_latency", latency):
                    logger.warning(f"Anomaly detected in {operation_name}: high latency {latency}s")
                
                # Success
                self.metrics["successful_requests"] += 1
                if attempt > 0:
                    self.metrics["recovered_requests"] += 1
                
                return result
                
            except AIError as e:
                last_error = e
                self._record_error(e)
                
                if not e.should_retry():
                    # Try fallbacks for non-retryable errors
                    if fallbacks:
                        try:
                            self.metrics["fallback_used"] += 1
                            return await self.fallback_handler.execute_with_fallback(
                                operation_name,
                                operation,
                                *args,
                                **kwargs
                            )
                        except Exception as fallback_error:
                            logger.error(f"Fallback failed: {fallback_error}")
                            raise
                    raise
                
                # Calculate retry delay
                delay = self.retry_strategy.get_delay(attempt)
                logger.info(f"Retrying {operation_name} after {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                
                e.increment_retry()
                attempt += 1
                
            except Exception as e:
                last_error = e
                self._record_error(e)
                
                if not self.retry_strategy.should_retry(e, attempt):
                    # Try fallbacks
                    if fallbacks:
                        try:
                            self.metrics["fallback_used"] += 1
                            return await self.fallback_handler.execute_with_fallback(
                                operation_name,
                                operation,
                                *args,
                                **kwargs
                            )
                        except Exception as fallback_error:
                            logger.error(f"Fallback failed: {fallback_error}")
                            raise
                    raise
                
                delay = self.retry_strategy.get_delay(attempt)
                logger.info(f"Retrying {operation_name} after {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                attempt += 1
        
        # Max retries exceeded
        self.metrics["failed_requests"] += 1
        raise AIError(
            message=f"Max retries exceeded for {operation_name}",
            error_code="MAX_RETRIES",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PERMANENT,
            context={"last_error": str(last_error), "attempts": attempt}
        )
    
    async def _execute_operation(self, 
                                operation_name: str,
                                operation: Callable,
                                *args,
                                **kwargs) -> Any:
        """Execute the actual operation."""
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            # Wrap in AIError if needed
            if not isinstance(e, AIError):
                raise AIError(
                    message=str(e),
                    error_code="OPERATION_FAILED",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.TRANSIENT,
                    context={"operation": operation_name}
                )
            raise
    
    def _record_error(self, error: Exception):
        """Record error for analysis."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
        
        if isinstance(error, AIError):
            error_record.update({
                "code": error.error_code,
                "severity": error.severity.value,
                "category": error.category.value
            })
        
        self.error_history.append(error_record)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics."""
        metrics = self.metrics.copy()
        
        # Add circuit breaker states
        metrics["circuit_breakers"] = {
            name: breaker.get_state()
            for name, breaker in self.circuit_breakers.items()
        }
        
        # Calculate rates
        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
            metrics["recovery_rate"] = metrics["recovered_requests"] / metrics["total_requests"]
            metrics["fallback_rate"] = metrics["fallback_used"] / metrics["total_requests"]
        
        return metrics

# ============== Mock AI Services for Testing ==============
class MockLLMService:
    """Mock LLM service with configurable failure rates."""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    async def generate(self, prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
        """Simulate LLM generation with failures."""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Simulate failures
        if random.random() < self.failure_rate:
            error_type = random.choice([
                "rate_limit", "timeout", "invalid_response", "token_limit"
            ])
            
            if error_type == "rate_limit":
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=random.randint(10, 60)
                )
            elif error_type == "timeout":
                raise NetworkError("Request timeout")
            elif error_type == "invalid_response":
                raise LLMError(
                    message="Invalid response format",
                    error_code="INVALID_RESPONSE",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.PERMANENT
                )
            elif error_type == "token_limit":
                raise TokenLimitError(
                    "Token limit exceeded",
                    tokens_used=random.randint(8000, 10000)
                )
        
        # Return successful response
        return {
            "response": f"Generated response for: {prompt[:50]}...",
            "model": model,
            "tokens": len(prompt.split()),
            "cost": 0.001
        }

class MockVectorDB:
    """Mock vector database with failure simulation."""
    
    def __init__(self, failure_rate: float = 0.2):
        self.failure_rate = failure_rate
        self.data = [
            {"id": i, "content": f"Document {i}", "embedding": [random.random() for _ in range(128)]}
            for i in range(100)
        ]
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulate vector search with failures."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        if random.random() < self.failure_rate:
            raise NetworkError("Vector DB connection failed")
        
        # Return mock results
        results = random.sample(self.data, min(top_k, len(self.data)))
        return [
            {"id": r["id"], "content": r["content"], "score": random.random()}
            for r in results
        ]

# ============== Demonstration System ==============
class ResilientAISystem:
    """AI system with comprehensive error handling."""
    
    def __init__(self):
        self.orchestrator = ErrorRecoveryOrchestrator()
        self.llm_service = MockLLMService(failure_rate=0.4)
        self.vector_db = MockVectorDB(failure_rate=0.3)
        
        # Fallback services with lower failure rates
        self.fallback_llm = MockLLMService(failure_rate=0.1)
        self.fallback_vector_db = MockVectorDB(failure_rate=0.05)
        
        # Cache for simple fallbacks
        self.static_responses = {
            "greeting": "Hello! How can I help you today?",
            "error": "I'm experiencing technical difficulties. Please try again later.",
            "summary": "Here's a summary based on available information."
        }
        
        print(f"[ResilientAISystem] Initialized with error handling")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query with full error handling."""
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")
        
        # Define fallback functions
        async def llm_fallback(prompt: str, model: str = "gpt-3.5"):
            """Fallback to different LLM."""
            print("Using fallback LLM service")
            return await self.fallback_llm.generate(prompt, model)
        
        async def static_fallback(prompt: str, model: str = None):
            """Fallback to static response."""
            print("Using static response fallback")
            return {
                "response": self.static_responses.get("summary"),
                "model": "static",
                "tokens": 0,
                "cost": 0
            }
        
        # Search with error handling
        search_results = await self.orchestrator.execute_with_resilience(
            operation_name="vector_search",
            operation=self.vector_db.search,
            query=query,
            top_k=5,
            fallbacks=[self.fallback_vector_db.search]
        )
        
        print(f"Search completed: {len(search_results)} results")
        
        # Generate response with error handling
        context = "\n".join([r["content"] for r in search_results])
        prompt = f"Query: {query}\nContext: {context}\nGenerate a helpful response."
        
        response = await self.orchestrator.execute_with_resilience(
            operation_name="llm_generation",
            operation=self.llm_service.generate,
            prompt=prompt,
            model="gpt-4",
            fallbacks=[llm_fallback, static_fallback],
            validate_response=True,
            expected_schema={"required": ["response", "model"]}
        )
        
        print(f"Generation completed: {response['model']}")
        
        return {
            "query": query,
            "response": response["response"],
            "sources": search_results,
            "metadata": {
                "model": response["model"],
                "tokens": response.get("tokens", 0),
                "cost": response.get("cost", 0)
            }
        }
    
    async def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries with error handling."""
        print(f"\n{'='*60}")
        print(f"Batch processing {len(queries)} queries")
        print(f"{'='*60}")
        
        tasks = [self.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Query {i+1} failed: {result}")
                processed_results.append({
                    "query": queries[i],
                    "error": str(result),
                    "fallback_used": True,
                    "response": self.static_responses["error"]
                })
            else:
                processed_results.append(result)
        
        return processed_results

# ============== Demonstration ==============
async def demonstrate_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    system = ResilientAISystem()
    
    # Test 1: Single query with retries
    print("\nTEST 1: Single Query with Error Recovery")
    print("-" * 40)
    
    result1 = await system.process_query(
        "What are the best practices for error handling in AI systems?"
    )
    
    print(f"\nResult: {result1['response'][:100]}...")
    
    # Test 2: Batch processing with failures
    print("\nTEST 2: Batch Processing with Mixed Failures")
    print("-" * 40)
    
    queries = [
        "Explain circuit breakers",
        "What is exponential backoff?",
        "How do fallback strategies work?",
        "Describe retry patterns",
        "What is graceful degradation?"
    ]
    
    results = await system.batch_process(queries)
    
    successful = sum(1 for r in results if "error" not in r)
    print(f"\nBatch results: {successful}/{len(queries)} successful")
    
    # Test 3: Circuit breaker demonstration
    print("\nTEST 3: Circuit Breaker Behavior")
    print("-" * 40)
    
    # Force failures to trigger circuit breaker
    system.llm_service.failure_rate = 1.0  # 100% failure rate
    
    for i in range(7):
        try:
            print(f"\nAttempt {i+1}:")
            await system.process_query(f"Test query {i+1}")
        except AIError as e:
            print(f"  Circuit state: {e.error_code}")
    
    # Test 4: Recovery demonstration
    print("\nTEST 4: System Recovery")
    print("-" * 40)
    
    # Reduce failure rate to allow recovery
    system.llm_service.failure_rate = 0.2
    await asyncio.sleep(61)  # Wait for circuit recovery timeout
    
    print("\nAttempting recovery...")
    recovery_result = await system.process_query("Testing system recovery")
    print(f"System recovered: {recovery_result['response'][:50]}...")
    
    # Display metrics
    print("\nERROR HANDLING METRICS")
    print("-" * 40)
    
    metrics = system.orchestrator.get_metrics()
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Success rate: {metrics.get('success_rate', 0):.1%}")
    print(f"Recovery rate: {metrics.get('recovery_rate', 0):.1%}")
    print(f"Fallback usage: {metrics.get('fallback_rate', 0):.1%}")
    
    print("\nCircuit Breaker States:")
    for name, state in metrics.get("circuit_breakers", {}).items():
        print(f"  {name}: {state['state']} (failures: {state['failure_count']})")
    
    # Test 5: Adaptive behavior
    print("\nTEST 5: Adaptive Error Handling")
    print("-" * 40)
    
    adaptive_orchestrator = ErrorRecoveryOrchestrator()
    adaptive_orchestrator.retry_strategy = AdaptiveRetry()
    
    # Simulate varying failure patterns
    for phase in ["high_failure", "recovering", "stable"]:
        print(f"\nPhase: {phase}")
        
        if phase == "high_failure":
            system.llm_service.failure_rate = 0.8
        elif phase == "recovering":
            system.llm_service.failure_rate = 0.4
        else:
            system.llm_service.failure_rate = 0.1
        
        for _ in range(3):
            try:
                await system.process_query(f"Query in {phase} phase")
                print("Success")
            except Exception as e:
                print(f"Failed: {e}")

# ============== Main Execution ==============
if __name__ == "__main__":
    print("Starting AI Error Handling System...")
    
    # Run demonstration
    asyncio.run(demonstrate_error_handling())