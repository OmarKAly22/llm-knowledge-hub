#!/usr/bin/env python3
"""
AI Performance Optimization System
Production-ready optimization with caching, batching, parallelization,
and intelligent resource management for high-performance AI applications.
"""

import asyncio
import time
import hashlib
import random
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle

# ============== Performance Metrics ==============

@dataclass
class PerformanceMetrics:
    """Track performance metrics for optimization."""
    request_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency: float = 0
    total_tokens: int = 0
    total_cost: float = 0
    batch_count: int = 0
    parallel_executions: int = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        return {
            "requests": self.request_count,
            "cache_hit_rate": cache_rate,
            "avg_latency_ms": avg_latency * 1000,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "batches_processed": self.batch_count,
            "parallel_executions": self.parallel_executions
        }

# ============== Multi-Tier Caching System ==============

class CacheEntry:
    """Single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, cost: float = 0, size: int = 0):
        self.key = key
        self.value = value
        self.cost = cost
        self.size = size
        self.hits = 0
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def access(self):
        """Record cache access."""
        self.hits += 1
        self.last_accessed = time.time()
    
    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at
    
    def score(self) -> float:
        """Calculate cache entry score for eviction."""
        # Higher score = more valuable to keep
        # Consider: frequency, recency, cost to regenerate
        recency_score = 1 / (time.time() - self.last_accessed + 1)
        frequency_score = self.hits
        cost_score = self.cost
        
        return recency_score * frequency_score * cost_score

class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.metrics = PerformanceMetrics()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(key)
            entry = self.cache[key]
            entry.access()
            self.metrics.cache_hits += 1
            return entry.value
        
        self.metrics.cache_misses += 1
        return None
    
    def put(self, key: str, value: Any, cost: float = 0):
        """Add value to cache."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key].value = value
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = CacheEntry(key, value, cost)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()

class SemanticCache:
    """Cache based on semantic similarity of queries."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.embeddings = {}  # key -> embedding
        self.cache = {}  # key -> value
        self.metrics = PerformanceMetrics()
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        # Simplified - real implementation would use numpy
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def get(self, key: str, embedding: List[float]) -> Optional[Any]:
        """Get semantically similar cached result."""
        best_match = None
        best_similarity = 0
        
        for cached_key, cached_embedding in self.embeddings.items():
            similarity = self._compute_similarity(embedding, cached_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cached_key
        
        if best_match:
            self.metrics.cache_hits += 1
            return self.cache[best_match]
        
        self.metrics.cache_misses += 1
        return None
    
    def put(self, key: str, value: Any, embedding: List[float]):
        """Add to semantic cache."""
        self.embeddings[key] = embedding
        self.cache[key] = value

class MultiTierCache:
    """Multi-tier caching system with L1, L2, L3 caches."""
    
    def __init__(self):
        self.l1_cache = LRUCache(max_size=100)  # Memory - very fast, small
        self.l2_cache = LRUCache(max_size=1000)  # Redis simulation - fast, medium
        self.l3_cache = LRUCache(max_size=10000)  # Disk/CDN - slower, large
        self.semantic_cache = SemanticCache()
        self.metrics = PerformanceMetrics()
    
    async def get(self, key: str, embedding: Optional[List[float]] = None) -> Optional[Any]:
        """Get from cache hierarchy."""
        start_time = time.time()
        
        # Try L1 (memory)
        value = self.l1_cache.get(key)
        if value is not None:
            self.metrics.cache_hits += 1
            self.metrics.total_latency += time.time() - start_time
            return value
        
        # Try L2 (Redis)
        await asyncio.sleep(0.001)  # Simulate network latency
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value)
            self.metrics.cache_hits += 1
            self.metrics.total_latency += time.time() - start_time
            return value
        
        # Try L3 (Disk/CDN)
        await asyncio.sleep(0.01)  # Simulate disk I/O
        value = self.l3_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            self.l2_cache.put(key, value)
            self.l1_cache.put(key, value)
            self.metrics.cache_hits += 1
            self.metrics.total_latency += time.time() - start_time
            return value
        
        # Try semantic cache if embedding provided
        if embedding:
            value = self.semantic_cache.get(key, embedding)
            if value is not None:
                # Add to regular caches
                self.l1_cache.put(key, value)
                self.metrics.cache_hits += 1
                self.metrics.total_latency += time.time() - start_time
                return value
        
        self.metrics.cache_misses += 1
        self.metrics.total_latency += time.time() - start_time
        return None
    
    async def put(self, key: str, value: Any, cost: float = 0, embedding: Optional[List[float]] = None):
        """Add to cache hierarchy."""
        # Add to all tiers
        self.l1_cache.put(key, value, cost)
        self.l2_cache.put(key, value, cost)
        self.l3_cache.put(key, value, cost)
        
        if embedding:
            self.semantic_cache.put(key, value, embedding)

# ============== Request Batching ==============

@dataclass
class BatchRequest:
    """Single request in a batch."""
    request_id: str
    data: Any
    callback: Optional[Callable] = None
    priority: int = 0  # Higher = more important
    timestamp: float = field(default_factory=time.time)

class RequestBatcher:
    """Batch multiple requests for efficient processing."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_wait_time: float = 0.1,  # seconds
                 min_batch_size: int = 5):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        self.pending_requests = []
        self.processing = False
        self.metrics = PerformanceMetrics()
    
    async def add_request(self, request: BatchRequest) -> Any:
        """Add request to batch and wait for result."""
        # Add to pending
        self.pending_requests.append(request)
        
        # Create future for this request
        future = asyncio.Future()
        request.callback = lambda result: future.set_result(result)
        
        # Check if should process
        if len(self.pending_requests) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        elif not self.processing:
            # Start timer for batch processing
            asyncio.create_task(self._wait_and_process())
        
        # Wait for result
        return await future
    
    async def _wait_and_process(self):
        """Wait for more requests or timeout."""
        if self.processing:
            return
        
        self.processing = True
        await asyncio.sleep(self.max_wait_time)
        
        if len(self.pending_requests) >= self.min_batch_size:
            await self._process_batch()
        else:
            # Wait a bit more for minimum batch
            await asyncio.sleep(self.max_wait_time)
            await self._process_batch()
        
        self.processing = False
    
    async def _process_batch(self):
        """Process accumulated batch."""
        if not self.pending_requests:
            return
        
        # Sort by priority
        self.pending_requests.sort(key=lambda x: x.priority, reverse=True)
        
        # Take batch
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        # Process batch
        self.metrics.batch_count += 1
        results = await self._execute_batch(batch)
        
        # Return results via callbacks
        for request, result in zip(batch, results):
            if request.callback:
                request.callback(result)
    
    async def _execute_batch(self, batch: List[BatchRequest]) -> List[Any]:
        """Execute batched requests."""
        # Simulate batch processing
        await asyncio.sleep(0.1)  # Batch processing time
        
        # Return mock results
        return [f"Result for {req.request_id}" for req in batch]

# ============== Parallel Processing ==============

class ParallelExecutor:
    """Execute tasks in parallel with resource management."""
    
    def __init__(self, 
                 max_concurrency: int = 10,
                 use_processes: bool = False):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.metrics = PerformanceMetrics()
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_concurrency)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with concurrency control."""
        async with self.semaphore:
            self.metrics.parallel_executions += 1
            
            # Run in executor for CPU-bound tasks
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                func,
                *args
            )
            
            return result
    
    async def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Apply function to items in parallel."""
        tasks = [self.execute(func, item) for item in items]
        return await asyncio.gather(*tasks)
    
    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)

# ============== Prompt Optimization ==============

class PromptOptimizer:
    """Optimize prompts to reduce token usage and improve performance."""
    
    def __init__(self):
        self.compression_strategies = {
            "remove_redundancy": self._remove_redundancy,
            "use_abbreviations": self._use_abbreviations,
            "compress_examples": self._compress_examples,
            "optimize_format": self._optimize_format
        }
        self.metrics = PerformanceMetrics()
    
    def optimize(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Optimize prompt for efficiency."""
        original_tokens = self._count_tokens(prompt)
        optimized = prompt
        
        # Apply optimization strategies
        for strategy_name, strategy_func in self.compression_strategies.items():
            optimized = strategy_func(optimized)
            
            # Check if within token limit
            if max_tokens and self._count_tokens(optimized) <= max_tokens:
                break
        
        final_tokens = self._count_tokens(optimized)
        self.metrics.total_tokens += final_tokens
        
        # Track optimization
        reduction = (original_tokens - final_tokens) / original_tokens if original_tokens > 0 else 0
        print(f"Prompt optimized: {original_tokens} → {final_tokens} tokens ({reduction:.1%} reduction)")
        
        return optimized
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simplified)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _remove_redundancy(self, prompt: str) -> str:
        """Remove redundant information."""
        lines = prompt.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            if line.strip() and line.strip() not in seen:
                unique_lines.append(line)
                seen.add(line.strip())
        
        return '\n'.join(unique_lines)
    
    def _use_abbreviations(self, prompt: str) -> str:
        """Replace common phrases with abbreviations."""
        abbreviations = {
            "for example": "e.g.",
            "that is": "i.e.",
            "et cetera": "etc.",
            "versus": "vs.",
            "approximately": "~",
            "greater than": ">",
            "less than": "<"
        }
        
        result = prompt
        for full, abbr in abbreviations.items():
            result = result.replace(full, abbr)
        
        return result
    
    def _compress_examples(self, prompt: str) -> str:
        """Compress examples to essential information."""
        # Simple strategy: keep first and last example
        if "Example" in prompt:
            parts = prompt.split("Example")
            if len(parts) > 3:
                # Keep first and last example
                compressed = parts[0] + "Example" + parts[1] + "Example" + parts[-1]
                return compressed
        
        return prompt
    
    def _optimize_format(self, prompt: str) -> str:
        """Optimize formatting for clarity and brevity."""
        # Remove excessive whitespace
        lines = [line.strip() for line in prompt.split('\n')]
        # Remove empty lines
        lines = [line for line in lines if line]
        # Join with single newline
        return '\n'.join(lines)

# ============== Model Selection ==============

@dataclass
class ModelProfile:
    """Profile for an AI model."""
    name: str
    provider: str
    cost_per_token: float
    latency_ms: float  # Average latency
    max_tokens: int
    capabilities: Set[str]
    accuracy_score: float  # 0-1 scale

class ModelSelector:
    """Select optimal model based on requirements."""
    
    def __init__(self):
        self.models = {
            "gpt-4-turbo": ModelProfile(
                name="gpt-4-turbo",
                provider="openai",
                cost_per_token=0.00003,
                latency_ms=2000,
                max_tokens=128000,
                capabilities={"reasoning", "coding", "creative", "analysis"},
                accuracy_score=0.95
            ),
            "gpt-3.5-turbo": ModelProfile(
                name="gpt-3.5-turbo",
                provider="openai",
                cost_per_token=0.000002,
                latency_ms=500,
                max_tokens=16000,
                capabilities={"general", "coding", "creative"},
                accuracy_score=0.85
            ),
            "claude-3-opus": ModelProfile(
                name="claude-3-opus",
                provider="anthropic",
                cost_per_token=0.00003,
                latency_ms=2500,
                max_tokens=200000,
                capabilities={"reasoning", "coding", "analysis", "long-context"},
                accuracy_score=0.96
            ),
            "llama-3-8b": ModelProfile(
                name="llama-3-8b",
                provider="local",
                cost_per_token=0.0,
                latency_ms=100,
                max_tokens=8000,
                capabilities={"general", "fast"},
                accuracy_score=0.75
            )
        }
        self.metrics = PerformanceMetrics()
    
    def select_model(self,
                     task_type: str,
                     required_accuracy: float = 0.8,
                     max_latency_ms: float = 5000,
                     max_cost_per_token: float = 1.0,
                     min_tokens: int = 1000) -> Optional[str]:
        """Select best model for requirements."""
        candidates = []
        
        for model_name, profile in self.models.items():
            # Check requirements
            if (profile.accuracy_score >= required_accuracy and
                profile.latency_ms <= max_latency_ms and
                profile.cost_per_token <= max_cost_per_token and
                profile.max_tokens >= min_tokens):
                
                # Check capabilities
                if task_type in profile.capabilities or "general" in profile.capabilities:
                    candidates.append((model_name, profile))
        
        if not candidates:
            return None
        
        # Score candidates (lower is better)
        scored = []
        for model_name, profile in candidates:
            score = (
                profile.cost_per_token * 10000 +  # Cost weight
                profile.latency_ms / 1000 +  # Latency weight
                (1 - profile.accuracy_score) * 10  # Accuracy weight (inverted)
            )
            scored.append((score, model_name))
        
        # Select best (lowest score)
        scored.sort()
        selected = scored[0][1]
        
        print(f"Selected model: {selected} for {task_type}")
        return selected

# ============== Connection Pooling ==============

class ConnectionPool:
    """Manage reusable connections to AI services."""
    
    def __init__(self, 
                 min_size: int = 5,
                 max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.available_connections = asyncio.Queue(maxsize=max_size)
        self.in_use = set()
        self.created_count = 0
        self.metrics = PerformanceMetrics()
    
    async def initialize(self):
        """Initialize connection pool."""
        for _ in range(self.min_size):
            conn = await self._create_connection()
            await self.available_connections.put(conn)
    
    async def acquire(self) -> Any:
        """Acquire connection from pool."""
        try:
            # Try to get available connection
            conn = await asyncio.wait_for(
                self.available_connections.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Create new if under max
            if self.created_count < self.max_size:
                conn = await self._create_connection()
            else:
                # Wait for available connection
                conn = await self.available_connections.get()
        
        self.in_use.add(conn)
        return conn
    
    async def release(self, conn: Any):
        """Release connection back to pool."""
        self.in_use.discard(conn)
        
        if self.available_connections.full():
            # Pool is full, close connection
            await self._close_connection(conn)
        else:
            # Return to pool
            await self.available_connections.put(conn)
    
    async def _create_connection(self) -> Any:
        """Create new connection."""
        self.created_count += 1
        # Simulate connection creation
        await asyncio.sleep(0.01)
        return f"Connection-{self.created_count}"
    
    async def _close_connection(self, conn: Any):
        """Close connection."""
        # Simulate connection cleanup
        await asyncio.sleep(0.001)
        self.created_count -= 1

# ============== Optimized AI System ==============

class OptimizedAISystem:
    """AI system with all optimization techniques."""
    
    def __init__(self):
        # Initialize components
        self.cache = MultiTierCache()
        self.batcher = RequestBatcher(batch_size=10, max_wait_time=0.05)
        self.parallel_executor = ParallelExecutor(max_concurrency=5)
        self.prompt_optimizer = PromptOptimizer()
        self.model_selector = ModelSelector()
        self.connection_pool = ConnectionPool()
        
        # System metrics
        self.metrics = PerformanceMetrics()
        
        print(f"[OptimizedAISystem] Initialized with all optimizations")
    
    async def initialize(self):
        """Initialize system components."""
        await self.connection_pool.initialize()
        print(f"[OptimizedAISystem] Connection pool ready")
    
    async def process_request(self, 
                            query: str,
                            use_cache: bool = True,
                            use_batching: bool = True,
                            optimize_prompt: bool = True) -> Dict[str, Any]:
        """Process request with optimizations."""
        start_time = time.time()
        self.metrics.request_count += 1
        
        # Generate cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                latency = time.time() - start_time
                self.metrics.total_latency += latency
                print(f"Cache hit! Latency: {latency*1000:.1f}ms")
                return cached_result
        
        # Optimize prompt
        if optimize_prompt:
            optimized_query = self.prompt_optimizer.optimize(query, max_tokens=1000)
        else:
            optimized_query = query
        
        # Select model
        model = self.model_selector.select_model(
            task_type="general",
            required_accuracy=0.8,
            max_latency_ms=3000
        )
        
        # Process with batching
        if use_batching:
            request = BatchRequest(
                request_id=cache_key,
                data={"query": optimized_query, "model": model},
                priority=random.randint(1, 10)
            )
            result = await self.batcher.add_request(request)
        else:
            # Direct processing
            result = await self._process_single(optimized_query, model)
        
        # Cache result
        if use_cache:
            await self.cache.put(
                cache_key,
                result,
                cost=0.001,
                embedding=[random.random() for _ in range(128)]  # Mock embedding
            )
        
        latency = time.time() - start_time
        self.metrics.total_latency += latency
        self.metrics.total_cost += 0.001
        
        return {
            "query": query,
            "result": result,
            "model": model,
            "latency_ms": latency * 1000,
            "optimizations": {
                "cache_used": use_cache,
                "batching_used": use_batching,
                "prompt_optimized": optimize_prompt
            }
        }
    
    async def _process_single(self, query: str, model: str) -> str:
        """Process single request."""
        # Acquire connection
        conn = await self.connection_pool.acquire()
        
        try:
            # Simulate processing
            await asyncio.sleep(random.uniform(0.1, 0.3))
            result = f"Response from {model}: {query[:50]}..."
            return result
        finally:
            # Release connection
            await self.connection_pool.release(conn)
    
    async def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries with parallelization."""
        print(f"\nProcessing batch of {len(queries)} queries")
        start_time = time.time()
        
        # Process in parallel
        tasks = [self.process_request(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        batch_time = time.time() - start_time
        print(f"Batch completed in {batch_time:.2f}s")
        print(f"Average: {batch_time/len(queries)*1000:.1f}ms per query")
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        metrics = self.metrics.get_stats()
        
        # Add component metrics
        metrics["cache_stats"] = {
            "l1": self.cache.l1_cache.metrics.get_stats(),
            "l2": self.cache.l2_cache.metrics.get_stats(),
            "l3": self.cache.l3_cache.metrics.get_stats(),
        }
        
        metrics["batching"] = {
            "batches_processed": self.batcher.metrics.batch_count,
            "pending_requests": len(self.batcher.pending_requests)
        }
        
        metrics["parallel"] = {
            "executions": self.parallel_executor.metrics.parallel_executions,
            "max_concurrency": self.parallel_executor.max_concurrency
        }
        
        return metrics

# ============== Performance Testing ==============

class PerformanceBenchmark:
    """Benchmark different optimization strategies."""
    
    def __init__(self):
        self.results = {}
    
    async def run_benchmark(self, system: OptimizedAISystem, queries: List[str]):
        """Run comprehensive benchmark."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Test 1: Baseline (no optimizations)
        print("\nTest 1: Baseline (No Optimizations)")
        print("-" * 40)
        
        start = time.time()
        baseline_results = []
        for query in queries[:10]:
            result = await system.process_request(
                query,
                use_cache=False,
                use_batching=False,
                optimize_prompt=False
            )
            baseline_results.append(result)
        
        baseline_time = time.time() - start
        baseline_avg = baseline_time / len(baseline_results) * 1000
        
        print(f"Total time: {baseline_time:.2f}s")
        print(f"Average latency: {baseline_avg:.1f}ms")
        
        self.results["baseline"] = {
            "total_time": baseline_time,
            "avg_latency_ms": baseline_avg
        }
        
        # Test 2: With Caching
        print("\nTest 2: With Caching")
        print("-" * 40)
        
        # Warm up cache
        for query in queries[:5]:
            await system.process_request(query)
        
        start = time.time()
        cached_results = []
        for query in queries[:10]:
            result = await system.process_request(
                query,
                use_cache=True,
                use_batching=False,
                optimize_prompt=False
            )
            cached_results.append(result)
        
        cache_time = time.time() - start
        cache_avg = cache_time / len(cached_results) * 1000
        cache_improvement = (baseline_avg - cache_avg) / baseline_avg
        
        print(f"Total time: {cache_time:.2f}s")
        print(f"Average latency: {cache_avg:.1f}ms")
        print(f"Improvement: {cache_improvement:.1%}")
        
        self.results["caching"] = {
            "total_time": cache_time,
            "avg_latency_ms": cache_avg,
            "improvement": cache_improvement
        }
        
        # Test 3: With Batching
        print("\nTest 3: With Batching")
        print("-" * 40)
        
        start = time.time()
        batch_tasks = [
            system.process_request(
                query,
                use_cache=False,
                use_batching=True,
                optimize_prompt=False
            )
            for query in queries[:20]
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        
        batch_time = time.time() - start
        batch_avg = batch_time / len(batch_results) * 1000
        batch_improvement = (baseline_avg - batch_avg) / baseline_avg
        
        print(f"Total time: {batch_time:.2f}s")
        print(f"Average latency: {batch_avg:.1f}ms")
        print(f"Improvement: {batch_improvement:.1%}")
        
        self.results["batching"] = {
            "total_time": batch_time,
            "avg_latency_ms": batch_avg,
            "improvement": batch_improvement
        }
        
        # Test 4: All Optimizations
        print("\nTest 4: All Optimizations Combined")
        print("-" * 40)
        
        start = time.time()
        optimized_results = await system.process_batch(queries[:30])
        
        optimized_time = time.time() - start
        optimized_avg = optimized_time / len(optimized_results) * 1000
        total_improvement = (baseline_avg - optimized_avg) / baseline_avg
        
        print(f"  Total time: {optimized_time:.2f}s")
        print(f"  Average latency: {optimized_avg:.1f}ms")
        print(f"  Total improvement: {total_improvement:.1%}")
        
        self.results["all_optimizations"] = {
            "total_time": optimized_time,
            "avg_latency_ms": optimized_avg,
            "improvement": total_improvement
        }
        
        return self.results

# ============== Demonstration ==============

async def demonstrate_optimization():
    """Demonstrate performance optimization techniques."""
    print("\n" + "="*60)
    print("AI PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Initialize system
    system = OptimizedAISystem()
    await system.initialize()
    
    # Generate test queries
    queries = [
        f"Explain {topic} in detail with examples and best practices"
        for topic in [
            "machine learning", "neural networks", "deep learning",
            "natural language processing", "computer vision",
            "reinforcement learning", "transformer models",
            "gradient descent", "backpropagation", "attention mechanism",
            "BERT", "GPT", "ResNet", "YOLO", "GAN",
            "transfer learning", "fine-tuning", "embeddings",
            "tokenization", "optimization algorithms"
        ] * 2  # Duplicate to test caching
    ]
    
    # Test 1: Individual optimizations
    print("\nTEST 1: Individual Optimization Techniques")
    print("-" * 40)
    
    # Prompt optimization
    print("\n1. Prompt Optimization:")
    long_prompt = """
    Please provide a comprehensive analysis of artificial intelligence,
    including its history, current applications, future prospects,
    ethical considerations, technical challenges, and societal impact.
    Include specific examples for each point, and make sure to cover
    machine learning, deep learning, neural networks, natural language
    processing, computer vision, and robotics. Also discuss the role
    of data, algorithms, and compute power in AI development.
    """
    optimizer = PromptOptimizer()
    optimized = optimizer.optimize(long_prompt, max_tokens=100)
    
    # Model selection
    print("\n2. Model Selection:")
    selector = ModelSelector()
    for task in ["reasoning", "coding", "fast", "long-context"]:
        model = selector.select_model(
            task_type=task,
            required_accuracy=0.7,
            max_latency_ms=2000
        )
    
    # Test 2: Cache performance
    print("\nTEST 2: Cache Performance")
    print("-" * 40)
    
    # First pass - cache miss
    print("\nFirst pass (cache cold):")
    for query in queries[:3]:
        result = await system.process_request(query)
    
    # Second pass - cache hit
    print("\nSecond pass (cache warm):")
    for query in queries[:3]:
        result = await system.process_request(query)
    
    # Test 3: Batching performance
    print("\nTEST 3: Batching Performance")
    print("-" * 40)
    
    print("Processing 10 requests with batching...")
    batch_results = await system.process_batch(queries[:10])
    
    # Test 4: Run comprehensive benchmark
    print("\nTEST 4: Comprehensive Benchmark")
    
    benchmark = PerformanceBenchmark()
    benchmark_results = await benchmark.run_benchmark(system, queries)
    
    # Display final metrics
    print("\nSYSTEM METRICS")
    print("-" * 40)
    
    metrics = system.get_metrics()
    print(f"Total requests: {metrics['requests']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"Total cost: ${metrics['total_cost']:.4f}")
    
    # Summary
    print("\nOPTIMIZATION SUMMARY")
    print("-" * 40)
    
    for test_name, results in benchmark.results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        print(f"Average latency: {results['avg_latency_ms']:.1f}ms")
        if 'improvement' in results:
            print(f"  Improvement: {results['improvement']:.1%}")
    
    # Calculate total speedup
    if "baseline" in benchmark.results and "all_optimizations" in benchmark.results:
        baseline = benchmark.results["baseline"]["avg_latency_ms"]
        optimized = benchmark.results["all_optimizations"]["avg_latency_ms"]
        speedup = baseline / optimized
        
        print(f"\nTotal speedup: {speedup:.1f}x faster")
        print(f"Latency reduction: {baseline:.1f}ms → {optimized:.1f}ms")
    
    print("\nDEMONSTRATION COMPLETE")

# ============== Main Execution ==============

if __name__ == "__main__":
    print("Starting AI Performance Optimization System...")
    
    # Run demonstration
    asyncio.run(demonstrate_optimization())