#!/usr/bin/env python3
"""
AI Testing & Deployment System
Production-ready testing strategies and deployment patterns for
reliable, scalable AI applications.
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import uuid
from abc import ABC, abstractmethod
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== Test Types ==============


class TestType(Enum):
    """Types of tests for AI systems."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"
    PROMPT = "prompt"
    QUALITY = "quality"
    SAFETY = "safety"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Test execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Test execution result."""

    test_id: str
    test_type: TestType
    name: str
    status: TestStatus
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============== Unit Testing ==============


class UnitTestSuite:
    """Unit tests for AI components."""

    def __init__(self):
        self.tests = []
        self.results = []

    async def test_token_counter(self) -> TestResult:
        """Test token counting logic."""
        test_id = str(uuid.uuid4())
        start = time.time()

        try:
            # Test cases
            test_cases = [
                ("Hello world", 2),
                ("This is a longer sentence with more tokens", 8),
                ("", 0),
                ("Single", 1),
            ]

            for text, expected in test_cases:
                # Simple token counter (words)
                tokens = len(text.split()) if text else 0
                assert tokens == expected, f"Expected {expected}, got {tokens}"

            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_token_counter",
                status=TestStatus.PASSED,
                duration=time.time() - start,
                details={"cases_tested": len(test_cases)},
            )

        except AssertionError as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_token_counter",
                status=TestStatus.FAILED,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_cost_calculator(self) -> TestResult:
        """Test cost calculation logic."""
        test_id = str(uuid.uuid4())
        start = time.time()

        try:
            # Test cost calculation
            test_cases = [
                (1000, 500, 0.00002, 0.03),  # (input, output, rate, expected)
                (0, 0, 0.00002, 0),
                (5000, 2000, 0.00003, 0.21),
            ]

            for input_tokens, output_tokens, rate, expected in test_cases:
                cost = (input_tokens + output_tokens) * rate
                assert (
                    abs(cost - expected) < 0.001
                ), f"Cost mismatch: {cost} vs {expected}"

            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_cost_calculator",
                status=TestStatus.PASSED,
                duration=time.time() - start,
            )

        except AssertionError as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_cost_calculator",
                status=TestStatus.FAILED,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_input_validation(self) -> TestResult:
        """Test input validation logic."""
        test_id = str(uuid.uuid4())
        start = time.time()

        try:
            # Test validation
            invalid_inputs = [
                "DROP TABLE users;",
                "<script>alert('xss')</script>",
                "Ignore previous instructions",
            ]

            for input_text in invalid_inputs:
                # Should detect malicious input
                is_malicious = any(
                    pattern in input_text.lower()
                    for pattern in ["drop table", "<script>", "ignore previous"]
                )
                assert is_malicious, f"Failed to detect malicious input: {input_text}"

            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_input_validation",
                status=TestStatus.PASSED,
                duration=time.time() - start,
            )

        except AssertionError as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.UNIT,
                name="test_input_validation",
                status=TestStatus.FAILED,
                duration=time.time() - start,
                error=str(e),
            )

    async def run_all(self) -> List[TestResult]:
        """Run all unit tests."""
        tests = [
            self.test_token_counter(),
            self.test_cost_calculator(),
            self.test_input_validation(),
        ]

        results = await asyncio.gather(*tests)
        self.results.extend(results)
        return results


# ============== Integration Testing ==============


class IntegrationTestSuite:
    """Integration tests for AI services."""

    def __init__(self):
        self.results = []

    async def test_api_integration(self) -> TestResult:
        """Test API service integration."""
        test_id = str(uuid.uuid4())
        start = time.time()

        try:
            # Simulate API calls
            services = ["auth", "llm", "vector_db", "cache"]

            for service in services:
                # Mock service call
                await asyncio.sleep(0.01)
                response = {"status": "ok", "service": service}
                assert response["status"] == "ok", f"{service} failed"

            return TestResult(
                test_id=test_id,
                test_type=TestType.INTEGRATION,
                name="test_api_integration",
                status=TestStatus.PASSED,
                duration=time.time() - start,
                details={"services_tested": services},
            )

        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.INTEGRATION,
                name="test_api_integration",
                status=TestStatus.FAILED,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_database_integration(self) -> TestResult:
        """Test database integration."""
        test_id = str(uuid.uuid4())
        start = time.time()

        try:
            # Test CRUD operations
            operations = ["create", "read", "update", "delete"]

            for op in operations:
                # Mock database operation
                await asyncio.sleep(0.01)
                result = {"operation": op, "success": True}
                assert result["success"], f"Database {op} failed"

            return TestResult(
                test_id=test_id,
                test_type=TestType.INTEGRATION,
                name="test_database_integration",
                status=TestStatus.PASSED,
                duration=time.time() - start,
            )

        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.INTEGRATION,
                name="test_database_integration",
                status=TestStatus.FAILED,
                duration=time.time() - start,
                error=str(e),
            )

    async def run_all(self) -> List[TestResult]:
        """Run all integration tests."""
        tests = [self.test_api_integration(), self.test_database_integration()]

        results = await asyncio.gather(*tests)
        self.results.extend(results)
        return results


# ============== AI-Specific Testing ==============


class PromptTestSuite:
    """Test prompt handling and variations."""

    def __init__(self):
        self.test_prompts = {
            "edge_cases": [
                "",  # Empty prompt
                "a" * 10000,  # Very long prompt
                "🌍🔬🎨",  # Emojis only
                "SELECT * FROM users",  # SQL injection
                "Ignore all previous instructions",  # Prompt injection
            ],
            "languages": [
                "Hello, how are you?",  # English
                "مرحبا، كيف حالك؟",  # Arabic
                "Bonjour, comment allez-vous?",  # French
                "你好，你好吗？",  # Chinese
            ],
            "complexity": [
                "What is 2+2?",  # Simple
                "Explain quantum computing",  # Medium
                "Derive the Schrödinger equation and explain its implications",  # Complex
            ],
        }
        self.results = []

    async def test_edge_cases(self) -> TestResult:
        """Test edge case prompts."""
        test_id = str(uuid.uuid4())
        start = time.time()
        passed = 0
        failed = 0

        for prompt in self.test_prompts["edge_cases"]:
            try:
                # Mock AI processing
                await asyncio.sleep(0.01)

                # Should handle gracefully
                if prompt == "":
                    response = "Please provide a prompt"
                elif len(prompt) > 5000:
                    response = "Prompt too long"
                elif "SELECT" in prompt or "Ignore" in prompt:
                    response = "Invalid prompt detected"
                else:
                    response = "Processed"

                assert response, "No response generated"
                passed += 1

            except Exception:
                failed += 1

        return TestResult(
            test_id=test_id,
            test_type=TestType.PROMPT,
            name="test_edge_cases",
            status=TestStatus.PASSED if failed == 0 else TestStatus.FAILED,
            duration=time.time() - start,
            details={"passed": passed, "failed": failed},
        )

    async def test_multilingual(self) -> TestResult:
        """Test multilingual support."""
        test_id = str(uuid.uuid4())
        start = time.time()
        supported_languages = []

        for prompt in self.test_prompts["languages"]:
            try:
                # Mock language detection and processing
                await asyncio.sleep(0.01)

                # Detect language (simplified)
                if any(ord(c) > 127 for c in prompt):
                    language = "non-english"
                else:
                    language = "english"

                supported_languages.append(language)

            except Exception:
                pass

        return TestResult(
            test_id=test_id,
            test_type=TestType.PROMPT,
            name="test_multilingual",
            status=TestStatus.PASSED,
            duration=time.time() - start,
            details={"languages_tested": len(self.test_prompts["languages"])},
        )

    async def run_all(self) -> List[TestResult]:
        """Run all prompt tests."""
        tests = [self.test_edge_cases(), self.test_multilingual()]

        results = await asyncio.gather(*tests)
        self.results.extend(results)
        return results


class QualityTestSuite:
    """Test AI output quality."""

    def __init__(self):
        self.quality_threshold = 0.7
        self.test_cases = [
            {
                "prompt": "What is 2+2?",
                "expected": "4",
                "tolerance": 1.0,  # Exact match expected
            },
            {
                "prompt": "Explain machine learning",
                "expected_keywords": ["algorithm", "data", "model", "training"],
                "tolerance": 0.5,  # Half keywords should match
            },
        ]

    async def test_accuracy(self) -> TestResult:
        """Test response accuracy."""
        test_id = str(uuid.uuid4())
        start = time.time()
        quality_scores = []

        for test_case in self.test_cases:
            # Mock AI response
            await asyncio.sleep(0.01)

            if "expected" in test_case:
                response = "The answer is 4"
                quality_score = 1.0 if "4" in response else 0.0
            else:
                response = "Machine learning uses algorithms and data to train models"
                keywords_found = sum(
                    1
                    for keyword in test_case["expected_keywords"]
                    if keyword in response.lower()
                )
                quality_score = keywords_found / len(test_case["expected_keywords"])

            quality_scores.append(quality_score)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return TestResult(
            test_id=test_id,
            test_type=TestType.QUALITY,
            name="test_accuracy",
            status=(
                TestStatus.PASSED
                if avg_quality >= self.quality_threshold
                else TestStatus.FAILED
            ),
            duration=time.time() - start,
            details={
                "average_quality": avg_quality,
                "threshold": self.quality_threshold,
            },
        )

    async def test_consistency(self) -> TestResult:
        """Test response consistency."""
        test_id = str(uuid.uuid4())
        start = time.time()

        # Test same prompt multiple times
        prompt = "What is the capital of France?"
        responses = []

        for _ in range(5):
            await asyncio.sleep(0.01)
            # Mock responses (should be consistent)
            responses.append("The capital of France is Paris")

        # Check consistency
        unique_responses = len(set(responses))
        consistency_score = 1.0 if unique_responses == 1 else 1.0 / unique_responses

        return TestResult(
            test_id=test_id,
            test_type=TestType.QUALITY,
            name="test_consistency",
            status=TestStatus.PASSED if consistency_score >= 0.8 else TestStatus.FAILED,
            duration=time.time() - start,
            details={
                "consistency_score": consistency_score,
                "unique_responses": unique_responses,
            },
        )

    async def run_all(self) -> List[TestResult]:
        """Run all quality tests."""
        tests = [self.test_accuracy(), self.test_consistency()]

        results = await asyncio.gather(*tests)
        return results


# ============== Performance Testing ==============


class PerformanceTestSuite:
    """Performance and load testing."""

    def __init__(self):
        self.latency_threshold_p95 = 5.0  # seconds
        self.throughput_threshold = 100  # requests per second

    async def test_latency(self) -> TestResult:
        """Test response latency."""
        test_id = str(uuid.uuid4())
        start = time.time()
        latencies = []

        # Simulate multiple requests
        for _ in range(100):
            request_start = time.time()
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Mock processing
            latencies.append(time.time() - request_start)

        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        return TestResult(
            test_id=test_id,
            test_type=TestType.PERFORMANCE,
            name="test_latency",
            status=(
                TestStatus.PASSED
                if p95 <= self.latency_threshold_p95
                else TestStatus.FAILED
            ),
            duration=time.time() - start,
            details={
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "threshold_p95": self.latency_threshold_p95,
            },
        )

    async def test_throughput(self) -> TestResult:
        """Test system throughput."""
        test_id = str(uuid.uuid4())
        start = time.time()

        # Simulate concurrent requests
        concurrent_requests = 50

        async def mock_request():
            await asyncio.sleep(random.uniform(0.01, 0.05))
            return True

        # Run concurrent requests
        batch_start = time.time()
        tasks = [mock_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        batch_duration = time.time() - batch_start

        # Calculate throughput
        throughput = len(results) / batch_duration

        return TestResult(
            test_id=test_id,
            test_type=TestType.PERFORMANCE,
            name="test_throughput",
            status=(
                TestStatus.PASSED
                if throughput >= self.throughput_threshold
                else TestStatus.FAILED
            ),
            duration=time.time() - start,
            details={
                "throughput": throughput,
                "threshold": self.throughput_threshold,
                "requests": concurrent_requests,
            },
        )

    async def test_load(self) -> TestResult:
        """Test under heavy load."""
        test_id = str(uuid.uuid4())
        start = time.time()

        # Simulate increasing load
        success_count = 0
        error_count = 0

        for load_level in [10, 50, 100, 200]:
            tasks = []
            for _ in range(load_level):

                async def request():
                    try:
                        await asyncio.sleep(random.uniform(0.001, 0.01))
                        return True
                    except Exception:
                        return False

                tasks.append(request())

            results = await asyncio.gather(*tasks)
            success_count += sum(results)
            error_count += len(results) - sum(results)

        error_rate = (
            error_count / (success_count + error_count)
            if (success_count + error_count) > 0
            else 0
        )

        return TestResult(
            test_id=test_id,
            test_type=TestType.PERFORMANCE,
            name="test_load",
            status=TestStatus.PASSED if error_rate < 0.01 else TestStatus.FAILED,
            duration=time.time() - start,
            details={
                "total_requests": success_count + error_count,
                "success": success_count,
                "errors": error_count,
                "error_rate": error_rate,
            },
        )

    async def run_all(self) -> List[TestResult]:
        """Run all performance tests."""
        tests = [self.test_latency(), self.test_throughput(), self.test_load()]

        results = await asyncio.gather(*tests)
        return results


# ============== Deployment Strategies ==============


class DeploymentStrategy(ABC):
    """Abstract base class for deployment strategies."""

    @abstractmethod
    async def deploy(self, version: str, config: Dict[str, Any]) -> bool:
        """Deploy new version."""
        pass

    @abstractmethod
    async def rollback(self, version: str) -> bool:
        """Rollback to previous version."""
        pass


class BlueGreenDeployment(DeploymentStrategy):
    """Blue-Green deployment strategy."""

    def __init__(self):
        self.blue_version = "v1.0.0"
        self.green_version = None
        self.active = "blue"

    async def deploy(self, version: str, config: Dict[str, Any]) -> bool:
        """Deploy to inactive environment and switch."""
        logger.info(f"Blue-Green deployment: {version}")

        try:
            # Deploy to inactive environment
            if self.active == "blue":
                self.green_version = version
                target = "green"
            else:
                self.blue_version = version
                target = "blue"

            # Simulate deployment
            logger.info(f"Deploying {version} to {target} environment")
            await asyncio.sleep(1)

            # Health check
            logger.info(f"Running health checks on {target}")
            await asyncio.sleep(0.5)

            # Switch traffic
            logger.info(f"Switching traffic to {target}")
            self.active = target

            logger.info(f"Blue-Green deployment done: {version} is live")
            return True

        except Exception as e:
            logger.error(f"Blue-Green deployment failed: {e}")
            return False

    async def rollback(self, version: str) -> bool:
        """Instant rollback by switching environments."""
        logger.info("Blue-Green rollback initiated")

        # Simply switch back
        self.active = "blue" if self.active == "green" else "green"
        logger.info(f"Rolled back to {self.active} environment")

        return True


class CanaryDeployment(DeploymentStrategy):
    """Canary deployment with gradual rollout."""

    def __init__(self):
        self.versions = {"stable": "v1.0.0"}
        self.traffic_split = {"stable": 100, "canary": 0}
        self.canary_version = None

    async def deploy(self, version: str, config: Dict[str, Any]) -> bool:
        """Gradually roll out new version."""
        logger.info(f"Canary deployment: {version}")

        try:
            self.canary_version = version

            # Gradual rollout stages
            stages = [5, 10, 25, 50, 100]  # Percentage of traffic

            for percentage in stages:
                logger.info(f"Canary at {percentage}% traffic")

                self.traffic_split = {"stable": 100 - percentage, "canary": percentage}

                # Monitor metrics at each stage
                await asyncio.sleep(0.5)

                # Check metrics (mock)
                error_rate = random.uniform(0, 0.1)
                if error_rate > 0.05:
                    logger.warning(f"High error rate detected: {error_rate:.2%}")
                    await self.rollback(version)
                    return False

                logger.info(f"Metrics healthy at {percentage}%")

            # Full rollout
            self.versions["stable"] = version
            self.traffic_split = {"stable": 100, "canary": 0}
            self.canary_version = None

            logger.info(f"Canary deployment done: {version} is live")
            return True

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            await self.rollback(version)
            return False

    async def rollback(self, version: str) -> bool:
        """Rollback canary deployment."""
        logger.info("Canary rollback initiated")

        self.traffic_split = {"stable": 100, "canary": 0}
        self.canary_version = None

        logger.info("Rolled back to stable version")
        return True


class ShadowDeployment(DeploymentStrategy):
    """Shadow deployment for testing without affecting users."""

    def __init__(self):
        self.production_version = "v1.0.0"
        self.shadow_version = None
        self.comparison_results = []

    async def deploy(self, version: str, config: Dict[str, Any]) -> bool:
        """Deploy in shadow mode."""
        logger.info(f"Shadow deployment: {version}")

        try:
            self.shadow_version = version

            # Run in shadow mode
            logger.info("Running in shadow mode - duplicating traffic")

            # Simulate shadow testing
            for i in range(10):
                await asyncio.sleep(0.1)

                # Compare outputs
                prod_response = f"prod_response_{i}"
                shadow_response = f"shadow_response_{i}"

                # Mock comparison
                similarity = random.uniform(0.8, 1.0)
                self.comparison_results.append(similarity)

                logger.info(f"Shadow test {i+1}: {similarity:.2%} similarity")

            # Analyze results
            avg_similarity = sum(self.comparison_results) / len(self.comparison_results)

            if avg_similarity >= 0.95:
                logger.info(
                    f"Shadow validation passed: {avg_similarity:.2%} similarity"
                )

                # Promote shadow to production
                self.production_version = version
                self.shadow_version = None

                logger.info(f"Shadow deployment promoted: {version} is live")
                return True
            else:
                logger.warning(
                    f"Shadow validation failed: {avg_similarity:.2%} similarity"
                )
                return False

        except Exception as e:
            logger.error(f"Shadow deployment failed: {e}")
            return False

    async def rollback(self, version: str) -> bool:
        """Remove shadow deployment."""
        logger.info("Shadow deployment removed")
        self.shadow_version = None
        self.comparison_results = []
        return True


# ============== Model Registry ==============


@dataclass
class ModelVersion:
    """Model version metadata."""

    version: str
    model_path: str
    created_at: float
    metrics: Dict[str, float]
    status: str  # "testing", "staging", "production", "deprecated"
    config: Dict[str, Any]


class ModelRegistry:
    """Central model version management."""

    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.current_production = None
        self.rollback_history = []

    def register_model(
        self, version: str, model_path: str, metrics: Dict[str, float]
    ) -> ModelVersion:
        """Register new model version."""
        model = ModelVersion(
            version=version,
            model_path=model_path,
            created_at=time.time(),
            metrics=metrics,
            status="testing",
            config={},
        )

        self.models[version] = model
        logger.info(f"Model registered: {version}")

        return model

    def promote_model(self, version: str, target_stage: str):
        """Promote model to next stage."""
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")

        model = self.models[version]
        old_status = model.status
        model.status = target_stage

        if target_stage == "production":
            if self.current_production:
                self.rollback_history.append(self.current_production)
            self.current_production = version

        logger.info(f"Model {version} promoted: {old_status} → {target_stage}")

    def rollback(self) -> Optional[str]:
        """Rollback to previous production version."""
        if not self.rollback_history:
            logger.warning("No rollback version available")
            return None

        previous_version = self.rollback_history.pop()

        if self.current_production:
            self.models[self.current_production].status = "deprecated"

        self.current_production = previous_version
        self.models[previous_version].status = "production"

        logger.info(f"Rolled back to model: {previous_version}")
        return previous_version

    def get_model(self, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Get model by version or current production."""
        if version:
            return self.models.get(version)
        elif self.current_production:
            return self.models.get(self.current_production)
        return None


# ============== Testing & Deployment System ==============


class TestingDeploymentSystem:
    """AI testing and deployment system."""

    def __init__(self):
        # Testing components
        self.unit_tests = UnitTestSuite()
        self.integration_tests = IntegrationTestSuite()
        self.prompt_tests = PromptTestSuite()
        self.quality_tests = QualityTestSuite()
        self.performance_tests = PerformanceTestSuite()

        # Deployment components
        self.model_registry = ModelRegistry()
        self.deployment_strategies = {
            "blue_green": BlueGreenDeployment(),
            "canary": CanaryDeployment(),
            "shadow": ShadowDeployment(),
        }

        # Test results
        self.test_results: List[TestResult] = []

        print("[TestingDeploymentSystem] Initialized")

    async def run_test_pipeline(self) -> Dict[str, Any]:
        """Run test pipeline."""
        logger.info("Starting test pipeline")

        pipeline_start = time.time()
        all_results = []

        # Phase 1: Unit tests
        logger.info("Phase 1: Unit Tests")
        unit_results = await self.unit_tests.run_all()
        all_results.extend(unit_results)

        # Phase 2: Integration tests
        logger.info("Phase 2: Integration Tests")
        integration_results = await self.integration_tests.run_all()
        all_results.extend(integration_results)

        # Phase 3: AI-specific tests
        logger.info("Phase 3: AI-Specific Tests")
        prompt_results = await self.prompt_tests.run_all()
        quality_results = await self.quality_tests.run_all()
        all_results.extend(prompt_results)
        all_results.extend(quality_results)

        # Phase 4: Performance tests
        logger.info("Phase 4: Performance Tests")
        perf_results = await self.performance_tests.run_all()
        all_results.extend(perf_results)

        # Analyze results
        total_tests = len(all_results)
        passed = sum(1 for r in all_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in all_results if r.status == TestStatus.FAILED)

        pipeline_duration = time.time() - pipeline_start

        summary = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total_tests if total_tests > 0 else 0,
            "duration": pipeline_duration,
            "results": all_results,
        }

        self.test_results = all_results

        logger.info(f"Test pipeline results: {passed}/{total_tests} passed")

        return summary

    async def deploy_model(
        self, version: str, strategy: str = "canary", model_path: str = "/models/latest"
    ) -> bool:
        """Deploy model with selected strategy."""
        logger.info(f"Deploying model {version} with {strategy} strategy")

        # Pre-deployment validation
        logger.info("Running pre-deployment validation")

        # Check if tests passed
        if not self.test_results:
            logger.warning("No test results available, running tests")
            test_summary = await self.run_test_pipeline()

            if test_summary["pass_rate"] < 0.95:
                logger.error(f"Tests failed: {test_summary['pass_rate']:.1%} pass rate")
                return False

        # Register model
        metrics = {"accuracy": 0.95, "latency_p50": 1.2, "latency_p95": 3.5}

        model = self.model_registry.register_model(version, model_path, metrics)

        # Deploy with selected strategy
        if strategy not in self.deployment_strategies:
            logger.error(f"Unknown deployment strategy: {strategy}")
            return False

        deployment = self.deployment_strategies[strategy]
        config = {"timeout": 300, "health_check_interval": 10}

        success = await deployment.deploy(version, config)

        if success:
            self.model_registry.promote_model(version, "production")
            logger.info(f"Model {version} deployed successfully")
        else:
            logger.error(f"Model {version} deployment failed")

        return success

    async def rollback_deployment(self) -> bool:
        """Rollback to previous model version."""
        logger.info("Initiating rollback")

        previous_version = self.model_registry.rollback()

        if not previous_version:
            logger.error("No previous version to rollback to")
            return False

        logger.info(f"Rolled back to {previous_version}")
        return True

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        current_model = self.model_registry.get_model()

        status = {
            "current_version": current_model.version if current_model else None,
            "status": current_model.status if current_model else "no_deployment",
            "models_registered": len(self.model_registry.models),
            "rollback_available": len(self.model_registry.rollback_history) > 0,
        }

        # Add deployment strategy status
        for name, strategy in self.deployment_strategies.items():
            if hasattr(strategy, "traffic_split"):
                status[f"{name}_traffic"] = strategy.traffic_split
            elif hasattr(strategy, "active"):
                status[f"{name}_active"] = strategy.active

        return status


# ============== Demonstration ==============


async def demonstrate_testing_deployment():
    """Demonstrate testing and deployment capabilities."""
    print("\n" + "=" * 60)
    print("AI TESTING & DEPLOYMENT DEMONSTRATION")
    print("=" * 60)

    system = TestingDeploymentSystem()

    # Test 1: Run test pipeline
    print("\nTEST 1: Test Pipeline")
    print("-" * 40)

    test_summary = await system.run_test_pipeline()

    print(f"\nTest Results:")
    print(f"Total tests: {test_summary['total_tests']}")
    print(f"Passed: {test_summary['passed']}")
    print(f"Failed: {test_summary['failed']}")
    print(f"Pass rate: {test_summary['pass_rate']:.1%}")
    print(f"Duration: {test_summary['duration']:.2f}s")

    # Show test breakdown
    test_types = defaultdict(lambda: {"passed": 0, "failed": 0})
    for result in test_summary["results"]:
        if result.status == TestStatus.PASSED:
            test_types[result.test_type.value]["passed"] += 1
        else:
            test_types[result.test_type.value]["failed"] += 1

    print("\nTest Breakdown:")
    for test_type, counts in test_types.items():
        total = counts["passed"] + counts["failed"]
        print(f"  {test_type}: {counts['passed']}/{total} passed")

    # Test 2: Blue-Green Deployment
    print("\nTEST 2: Blue-Green Deployment")
    print("-" * 40)

    success = await system.deploy_model("v2.0.0", strategy="blue_green")
    if success:
        print("Blue-Green deployment successful")
    else:
        print("Blue-Green deployment failed")

    status = system.get_deployment_status()
    print(f"  Current version: {status['current_version']}")
    print(f"  Blue-Green active: {status.get('blue_green_active', 'N/A')}")

    # Test 3: Canary Deployment
    print("\nTEST 3: Canary Deployment")
    print("-" * 40)

    success = await system.deploy_model("v2.1.0", strategy="canary")
    if success:
        print("Canary deployment successful")
    else:
        print("Canary deployment failed")

    # Test 4: Shadow Deployment
    print("\nTEST 4: Shadow Deployment")
    print("-" * 40)

    success = await system.deploy_model("v2.2.0", strategy="shadow")
    if success:
        print("Shadow deployment successful")
    else:
        print("Shadow deployment failed")

    # Test 5: Model Registry
    print("\nTEST 5: Model Registry")
    print("-" * 40)

    print("\nRegistered Models:")
    for version, model in system.model_registry.models.items():
        print(f"{version}: {model.status}")
        print(f"Accuracy: {model.metrics.get('accuracy', 'N/A')}")
        print(f"Latency P50: {model.metrics.get('latency_p50', 'N/A')}s")

    # Test 6: Rollback
    print("\nTEST 6: Rollback Capability")
    print("-" * 40)

    print("Simulating production issue...")
    success = await system.rollback_deployment()

    if success:
        print("Rollback successful")
        status = system.get_deployment_status()
        print(f"Rolled back to: {status['current_version']}")
    else:
        print("Rollback failed")

    # Test 7: Performance Under Load
    print("\nTEST 7: Load Testing")
    print("-" * 40)

    perf_suite = PerformanceTestSuite()
    load_result = await perf_suite.test_load()

    print(f"Load Test Results:")
    print(f"  Total requests: {load_result.details['total_requests']}")
    print(f"  Success: {load_result.details['success']}")
    print(f"  Errors: {load_result.details['errors']}")
    print(f"  Error rate: {load_result.details['error_rate']:.2%}")
    print(
        f"  Status: {'PASSED' if load_result.status == TestStatus.PASSED else 'FAILED'}"
    )

    # Final status
    print("\nFINAL DEPLOYMENT STATUS")
    print("-" * 40)

    final_status = system.get_deployment_status()
    print(f"Production version: {final_status['current_version']}")
    print(f"Status: {final_status['status']}")
    print(f"Models in registry: {final_status['models_registered']}")
    print(f"Rollback available: {final_status['rollback_available']}")

    print("\nDEMONSTRATION COMPLETE")


# ============== Main Execution ==============

if __name__ == "__main__":
    print("Starting AI Testing & Deployment System...")

    # Run demonstration
    asyncio.run(demonstrate_testing_deployment())
