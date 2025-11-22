#!/usr/bin/env python3
"""
AI Monitoring System
Production-ready observability with metrics, tracing, logging,
and intelligent alerting for AI applications.
"""

import time
import random
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import logging
import numpy as np
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============== Metric Types ==============


class MetricType(Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Single metric data point."""

    name: str
    value: float
    type: MetricType
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        return f"{self.name}{{{labels_str}}} {self.value} {int(self.timestamp * 1000)}"


@dataclass
class Trace:
    """Distributed trace span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def duration(self) -> Optional[float]:
        """Calculate span duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


# ============== Metrics Collection ==============


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.collection_interval = 10  # seconds

    def increment(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """Increment counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] += value

        self.metrics[name].append(
            Metric(
                name=name,
                value=self.counters[key],
                type=MetricType.COUNTER,
                labels=labels or {},
            )
        )

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

        self.metrics[name].append(
            Metric(name=name, value=value, type=MetricType.GAUGE, labels=labels or {})
        )

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add value to histogram."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        self.metrics[name].append(
            Metric(
                name=name, value=value, type=MetricType.HISTOGRAM, labels=labels or {}
            )
        )

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistical summary of metric."""
        if name not in self.metrics:
            return {}

        values = [m.value for m in self.metrics[name][-1000:]]  # Last 1000 values

        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric."""
        if not labels:
            return name

        labels_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name},{labels_str}"


# ============== AI-Specific Metrics ==============


class AIMetricsCollector:
    """Collect AI-specific metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.request_tokens = defaultdict(list)
        self.response_tokens = defaultdict(list)
        self.latencies = defaultdict(list)
        self.costs = defaultdict(float)
        self.quality_scores = defaultdict(list)
        self.error_rates = defaultdict(lambda: {"errors": 0, "total": 0})

    def track_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        cost: float,
        quality_score: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Track AI request metrics."""

        labels = {"model": model}

        # Token metrics
        self.metrics.histogram("ai_input_tokens", input_tokens, labels)
        self.metrics.histogram("ai_output_tokens", output_tokens, labels)
        self.metrics.histogram("ai_total_tokens", input_tokens + output_tokens, labels)

        # Performance metrics
        self.metrics.histogram("ai_latency_seconds", latency, labels)

        # Cost metrics
        self.metrics.increment("ai_cost_dollars", cost, labels)
        self.costs[model] += cost

        # Quality metrics
        if quality_score is not None:
            self.metrics.histogram("ai_quality_score", quality_score, labels)
            self.quality_scores[model].append(quality_score)

        # Error tracking
        self.error_rates[model]["total"] += 1
        if error:
            self.metrics.increment("ai_errors_total", 1, {**labels, "error": error})
            self.error_rates[model]["errors"] += 1

        # Calculate rates
        error_rate = (
            self.error_rates[model]["errors"] / self.error_rates[model]["total"]
        )
        self.metrics.gauge("ai_error_rate", error_rate, labels)

        # Track for analysis
        self.request_tokens[model].append(input_tokens)
        self.response_tokens[model].append(output_tokens)
        self.latencies[model].append(latency)

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get comprehensive stats for a model."""

        stats = {
            "model": model,
            "total_requests": self.error_rates[model]["total"],
            "error_rate": self.error_rates[model]["errors"]
            / max(self.error_rates[model]["total"], 1),
            "total_cost": self.costs[model],
        }

        # Token statistics
        if self.request_tokens[model]:
            stats["input_tokens"] = {
                "mean": statistics.mean(self.request_tokens[model]),
                "p95": np.percentile(self.request_tokens[model], 95),
            }

        if self.response_tokens[model]:
            stats["output_tokens"] = {
                "mean": statistics.mean(self.response_tokens[model]),
                "p95": np.percentile(self.response_tokens[model], 95),
            }

        # Latency statistics
        if self.latencies[model]:
            stats["latency"] = {
                "p50": np.percentile(self.latencies[model], 50),
                "p95": np.percentile(self.latencies[model], 95),
                "p99": np.percentile(self.latencies[model], 99),
            }

        # Quality statistics
        if self.quality_scores[model]:
            stats["quality"] = {
                "mean": statistics.mean(self.quality_scores[model]),
                "min": min(self.quality_scores[model]),
            }

        return stats


# ============== Distributed Tracing ==============


class DistributedTracer:
    """Distributed tracing system."""

    def __init__(self):
        self.traces: Dict[str, List[Trace]] = defaultdict(list)
        self.active_spans: Dict[str, Trace] = {}

    def start_span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Trace:
        """Start a new trace span."""

        if not trace_id:
            trace_id = hashlib.md5(
                f"{time.time()}{random.random()}".encode()
            ).hexdigest()

        span_id = hashlib.md5(
            f"{trace_id}{time.time()}{random.random()}".encode()
        ).hexdigest()[:16]

        span = Trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
        )

        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span

        return span

    def end_span(self, span_id: str, tags: Dict[str, Any] = None):
        """End a trace span."""

        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end_time = time.time()

            if tags:
                span.tags.update(tags)

            del self.active_spans[span_id]

    def add_log(self, span_id: str, message: str, level: str = "info"):
        """Add log to span."""

        if span_id in self.active_spans:
            self.active_spans[span_id].logs.append(
                {"timestamp": time.time(), "level": level, "message": message}
            )

    def get_trace(self, trace_id: str) -> List[Trace]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])

    def get_trace_timeline(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get timeline view of trace."""

        spans = self.get_trace(trace_id)
        if not spans:
            return []

        timeline = []
        for span in sorted(spans, key=lambda s: s.start_time):
            duration = span.duration()
            timeline.append(
                {
                    "operation": span.operation,
                    "start": span.start_time,
                    "duration": duration,
                    "tags": span.tags,
                    "logs": span.logs,
                }
            )

        return timeline


# ============== Quality Monitoring ==============


class QualityMonitor:
    """Monitor AI output quality."""

    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.relevance_scorer = RelevanceScorer()
        self.toxicity_checker = ToxicityChecker()
        self.quality_history = deque(maxlen=1000)

    async def evaluate_quality(
        self, prompt: str, response: str, expected: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate response quality."""

        # Check for hallucinations
        hallucination_score = self.hallucination_detector.check(prompt, response)

        # Check relevance
        relevance_score = self.relevance_scorer.score(prompt, response)

        # Check toxicity
        toxicity_score = self.toxicity_checker.check(response)

        # Calculate overall quality
        quality_score = (
            (1 - hallucination_score) * 0.4
            + relevance_score * 0.4
            + (1 - toxicity_score) * 0.2
        )

        quality_result = {
            "overall_score": quality_score,
            "hallucination_score": hallucination_score,
            "relevance_score": relevance_score,
            "toxicity_score": toxicity_score,
            "timestamp": time.time(),
        }

        # Compare with expected if provided
        if expected:
            similarity = self._calculate_similarity(response, expected)
            quality_result["similarity_to_expected"] = similarity

        # Track history
        self.quality_history.append(quality_result)

        # Detect quality degradation
        if len(self.quality_history) >= 10:
            recent_scores = [
                q["overall_score"] for q in list(self.quality_history)[-10:]
            ]
            avg_score = statistics.mean(recent_scores)

            if avg_score < 0.7:
                quality_result["warning"] = "Quality degradation detected"

        return quality_result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class HallucinationDetector:
    """Detect hallucinations in AI responses."""

    def check(self, prompt: str, response: str) -> float:
        """Return hallucination score (0=none, 1=severe)."""

        # Simplified detection logic
        hallucination_indicators = [
            "I don't have access to",
            "I cannot verify",
            "This information may not be accurate",
            "[HALLUCINATION]",  # Mock indicator
        ]

        score = 0.0
        for indicator in hallucination_indicators:
            if indicator.lower() in response.lower():
                score += 0.25

        # Check for made-up statistics
        if "%" in response and "study" in response.lower():
            # Suspicious of specific statistics without sources
            score += 0.1

        return min(score, 1.0)


class RelevanceScorer:
    """Score response relevance to prompt."""

    def score(self, prompt: str, response: str) -> float:
        """Return relevance score (0=irrelevant, 1=perfect)."""

        # Extract key terms from prompt
        prompt_terms = set(prompt.lower().split())
        response_terms = set(response.lower().split())

        # Check term overlap
        common_terms = prompt_terms & response_terms
        relevance = len(common_terms) / max(len(prompt_terms), 1)

        # Boost score if response directly addresses prompt
        question_words = {"what", "why", "how", "when", "where", "who"}
        if any(word in prompt_terms for word in question_words):
            if len(response) > 50:  # Reasonable response length
                relevance = min(relevance + 0.3, 1.0)

        return relevance


class ToxicityChecker:
    """Check for toxic content."""

    def check(self, text: str) -> float:
        """Return toxicity score (0=clean, 1=toxic)."""

        # Simplified toxic content detection
        toxic_indicators = [
            "hate",
            "kill",
            "attack",
            "stupid",
            "idiot",
            # In production, use proper toxicity detection
        ]

        text_lower = text.lower()
        toxicity = 0.0

        for indicator in toxic_indicators:
            if indicator in text_lower:
                toxicity += 0.2

        return min(toxicity, 1.0)


# ============== Model Drift Detection ==============


class DriftDetector:
    """Detect model performance drift."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.drift_threshold = 0.2  # 20% deviation

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics."""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline set: {metrics}")

    def add_observation(self, metric_name: str, value: float) -> Optional[float]:
        """Add observation and check for drift."""

        self.current_metrics[metric_name].append(value)

        # Need enough observations
        if len(self.current_metrics[metric_name]) < 10:
            return None

        # Calculate drift
        if metric_name in self.baseline_metrics:
            baseline = self.baseline_metrics[metric_name]
            current_mean = statistics.mean(self.current_metrics[metric_name])

            if baseline > 0:
                drift = abs(current_mean - baseline) / baseline

                if drift > self.drift_threshold:
                    logger.warning(f"Drift detected in {metric_name}: {drift:.2%}")
                    return drift

        return None

    def get_drift_report(self) -> Dict[str, Any]:
        """Get comprehensive drift report."""

        report = {"timestamp": time.time(), "drifts": {}}

        for metric_name in self.baseline_metrics:
            if (
                metric_name in self.current_metrics
                and len(self.current_metrics[metric_name]) >= 10
            ):
                baseline = self.baseline_metrics[metric_name]
                current_mean = statistics.mean(self.current_metrics[metric_name])

                drift = abs(current_mean - baseline) / baseline if baseline > 0 else 0

                report["drifts"][metric_name] = {
                    "baseline": baseline,
                    "current": current_mean,
                    "drift_percentage": drift * 100,
                    "is_drifting": drift > self.drift_threshold,
                }

        return report


# ============== Alert Manager ==============


@dataclass
class Alert:
    """Alert definition."""

    name: str
    severity: AlertSeverity
    condition: str
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notifications_sent: Dict[str, float] = {}
        self.cooldown_period = 300  # 5 minutes

    def add_rule(
        self, name: str, condition: Callable, severity: AlertSeverity, message: str
    ):
        """Add alert rule."""

        self.alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "message": message,
        }

    def check_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check all rules and generate alerts."""

        new_alerts = []

        for rule_name, rule in self.alert_rules.items():
            try:
                if rule["condition"](metrics):
                    # Check cooldown
                    last_sent = self.notifications_sent.get(rule_name, 0)
                    if time.time() - last_sent > self.cooldown_period:

                        alert = Alert(
                            name=rule_name,
                            severity=rule["severity"],
                            condition=str(rule["condition"]),
                            message=rule["message"].format(**metrics),
                        )

                        self.alerts.append(alert)
                        new_alerts.append(alert)
                        self.notifications_sent[rule_name] = time.time()

                        logger.warning(
                            f"Alert triggered: {rule_name} - {alert.message}"
                        )

            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")

        return new_alerts

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]

    def resolve_alert(self, alert_name: str):
        """Mark alert as resolved."""
        for alert in self.alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_name}")


# ============== Dashboard & Visualization ==============


class Dashboard:
    """Dashboard for metrics visualization."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        ai_metrics: AIMetricsCollector,
        tracer: DistributedTracer,
    ):
        self.metrics = metrics_collector
        self.ai_metrics = ai_metrics
        self.tracer = tracer
        self.refresh_interval = 5  # seconds

    def get_overview(self) -> Dict[str, Any]:
        """Get dashboard overview."""

        overview = {
            "timestamp": time.time(),
            "status": "healthy",  # Simplified
            "metrics": {},
            "ai_performance": {},
            "active_traces": len(self.tracer.active_spans),
            "alerts": [],
        }

        # System metrics
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            stats = self.metrics.get_statistics(metric_name)
            if stats:
                overview["metrics"][metric_name] = {
                    "current": stats.get("mean", 0),
                    "p95": stats.get("p95", 0),
                }

        # AI metrics
        for model in ["gpt-4", "gpt-3.5", "claude"]:
            model_stats = self.ai_metrics.get_model_stats(model)
            if model_stats["total_requests"] > 0:
                overview["ai_performance"][model] = {
                    "requests": model_stats["total_requests"],
                    "error_rate": model_stats["error_rate"],
                    "latency_p50": model_stats.get("latency", {}).get("p50", 0),
                    "cost": model_stats["total_cost"],
                }

        return overview

    def get_detailed_metrics(self, metric_name: str) -> Dict[str, Any]:
        """Get detailed metrics for specific metric."""

        stats = self.metrics.get_statistics(metric_name)

        return {
            "metric": metric_name,
            "statistics": stats,
            "recent_values": [
                m.value for m in self.metrics.metrics.get(metric_name, [])[-20:]
            ],
        }


# ============== Monitoring System ==============


class MonitoringSystem:
    """AI monitoring system."""

    def __init__(self):
        # Core components
        self.metrics = MetricsCollector()
        self.ai_metrics = AIMetricsCollector(self.metrics)
        self.tracer = DistributedTracer()
        self.quality_monitor = QualityMonitor()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager()
        self.dashboard = Dashboard(self.metrics, self.ai_metrics, self.tracer)

        # Setup default alerts
        self._setup_default_alerts()

        # Set baseline metrics
        self.drift_detector.set_baseline(
            {"latency": 2.0, "error_rate": 0.01, "quality_score": 0.85}
        )

        print("[MonitoringSystem] Initialized with all components")

    def _setup_default_alerts(self):
        """Setup default alert rules."""

        # High error rate
        self.alert_manager.add_rule(
            "high_error_rate",
            lambda m: m.get("error_rate", 0) > 0.05,
            AlertSeverity.ERROR,
            "High error rate: {error_rate:.1%}",
        )

        # High latency
        self.alert_manager.add_rule(
            "high_latency",
            lambda m: m.get("latency_p95", 0) > 10,
            AlertSeverity.WARNING,
            "High latency: {latency_p95:.1f}s",
        )

        # High cost
        self.alert_manager.add_rule(
            "high_cost",
            lambda m: m.get("hourly_cost", 0) > 100,
            AlertSeverity.WARNING,
            "High cost: ${hourly_cost:.2f}/hour",
        )

        # Quality degradation
        self.alert_manager.add_rule(
            "quality_degradation",
            lambda m: m.get("quality_score", 1) < 0.7,
            AlertSeverity.ERROR,
            "Quality degradation: {quality_score:.2f}",
        )

    async def monitor_request(
        self, request_id: str, prompt: str, model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """Monitor AI request."""

        # Start trace
        trace = self.tracer.start_span("ai_request", trace_id=request_id)

        # Track start metrics
        self.metrics.increment("requests_total", labels={"model": model})

        try:
            # Input validation span
            validation_span = self.tracer.start_span(
                "input_validation", trace_id=request_id, parent_span_id=trace.span_id
            )

            await asyncio.sleep(0.01)  # Simulate validation
            self.tracer.end_span(validation_span.span_id)

            # Model inference span
            inference_span = self.tracer.start_span(
                "model_inference", trace_id=request_id, parent_span_id=trace.span_id
            )

            # Simulate AI processing
            start_time = time.time()
            await asyncio.sleep(random.uniform(0.5, 3.0))  # Simulate latency
            latency = time.time() - start_time

            # Mock response
            response = f"Response to: {prompt[:30]}..."
            input_tokens = len(prompt.split()) * 2
            output_tokens = len(response.split()) * 2
            cost = (input_tokens + output_tokens) * 0.00002

            self.tracer.end_span(
                inference_span.span_id,
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency": latency,
                },
            )

            # Quality evaluation span
            quality_span = self.tracer.start_span(
                "quality_evaluation", trace_id=request_id, parent_span_id=trace.span_id
            )

            quality_result = await self.quality_monitor.evaluate_quality(
                prompt, response
            )

            self.tracer.end_span(
                quality_span.span_id, {"quality_score": quality_result["overall_score"]}
            )

            # Track metrics
            self.ai_metrics.track_request(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                cost=cost,
                quality_score=quality_result["overall_score"],
            )

            # Check for drift
            self.drift_detector.add_observation("latency", latency)
            self.drift_detector.add_observation(
                "quality_score", quality_result["overall_score"]
            )

            # End main trace
            self.tracer.end_span(
                trace.span_id,
                {"status": "success", "total_tokens": input_tokens + output_tokens},
            )

            # Check alerts
            metrics_snapshot = {
                "error_rate": 0,
                "latency_p95": latency,
                "quality_score": quality_result["overall_score"],
            }
            self.alert_manager.check_rules(metrics_snapshot)

            return {
                "request_id": request_id,
                "response": response,
                "metrics": {
                    "latency": latency,
                    "tokens": input_tokens + output_tokens,
                    "cost": cost,
                    "quality": quality_result,
                },
                "trace_id": request_id,
            }

        except Exception as e:
            # Track error
            self.metrics.increment(
                "errors_total", labels={"model": model, "error": type(e).__name__}
            )
            self.ai_metrics.track_request(
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency=time.time() - trace.start_time,
                cost=0,
                error=str(e),
            )

            # Log to trace
            self.tracer.add_log(trace.span_id, f"Error: {e}", "error")
            self.tracer.end_span(trace.span_id, {"status": "error", "error": str(e)})

            raise

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""

        health = {"status": "healthy", "timestamp": time.time(), "components": {}}

        # Check each model
        for model in ["gpt-4", "gpt-3.5", "claude"]:
            stats = self.ai_metrics.get_model_stats(model)

            if stats["total_requests"] > 0:
                model_health = "healthy"

                if stats["error_rate"] > 0.1:
                    model_health = "degraded"
                elif stats["error_rate"] > 0.5:
                    model_health = "unhealthy"

                health["components"][model] = {
                    "status": model_health,
                    "error_rate": stats["error_rate"],
                    "requests": stats["total_requests"],
                }

        # Check for active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        if active_alerts:
            health["status"] = "degraded"
            health["active_alerts"] = [
                {"name": a.name, "severity": a.severity.value} for a in active_alerts
            ]

        # Check drift
        drift_report = self.drift_detector.get_drift_report()
        drifting_metrics = [
            name for name, info in drift_report["drifts"].items() if info["is_drifting"]
        ]

        if drifting_metrics:
            health["drift_detected"] = drifting_metrics

        return health


# ============== Demonstration ==============


async def demonstrate_monitoring():
    """Demonstrate monitoring capabilities."""
    print("\n" + "=" * 60)
    print("AI MONITORING DEMONSTRATION")
    print("=" * 60)

    system = MonitoringSystem()

    # Test 1: Normal operations
    print("\nTEST 1: Normal Request Monitoring")
    print("-" * 40)

    requests = [
        "Explain quantum computing",
        "What is machine learning?",
        "How does blockchain work?",
        "Describe neural networks",
        "What is cloud computing?",
    ]

    for i, prompt in enumerate(requests):
        request_id = f"req_{i:03d}"
        result = await system.monitor_request(request_id, prompt, model="gpt-4")

        print(f"\nRequest {request_id}:")
        print(f"Latency: {result['metrics']['latency']:.2f}s")
        print(f"Tokens: {result['metrics']['tokens']}")
        print(f"Cost: ${result['metrics']['cost']:.4f}")
        print(f"Quality: {result['metrics']['quality']['overall_score']:.2f}")

    # Test 2: Model comparison
    print("\nTEST 2: Multi-Model Comparison")
    print("-" * 40)

    models = ["gpt-4", "gpt-3.5", "claude"]

    for model in models:
        for _ in range(5):
            await system.monitor_request(
                f"{model}_{uuid.uuid4().hex[:8]}", "Test prompt", model=model
            )

    print("\nModel Performance Comparison:")
    for model in models:
        stats = system.ai_metrics.get_model_stats(model)
        if stats["total_requests"] > 0:
            print(f"\n{model}:")
            print(f"Requests: {stats['total_requests']}")
            print(f"Error rate: {stats['error_rate']:.1%}")
            print(f"Latency P50: {stats.get('latency', {}).get('p50', 0):.2f}s")
            print(f"Total cost: ${stats['total_cost']:.4f}")

    # Test 3: Quality monitoring
    print("\nTEST 3: Quality Monitoring")
    print("-" * 40)

    # Simulate quality degradation
    quality_prompts = [
        ("What is AI?", "Artificial Intelligence is..."),
        ("Explain Python", "Python is a programming language..."),
        ("What is data?", "[HALLUCINATION] Data is magical fairy dust..."),
        ("Describe ML", "Machine learning is..."),
    ]

    for prompt, response in quality_prompts:
        quality = await system.quality_monitor.evaluate_quality(prompt, response)
        print(f"\nPrompt: {prompt[:30]}...")
        print(f"Overall quality: {quality['overall_score']:.2f}")
        print(f"Hallucination: {quality['hallucination_score']:.2f}")
        print(f"Relevance: {quality['relevance_score']:.2f}")

        if "warning" in quality:
            print(f"{quality['warning']}")

    # Test 4: Drift detection
    print("\nTEST 4: Drift Detection")
    print("-" * 40)

    # Simulate drift
    print("Simulating performance drift...")
    for i in range(20):
        # Gradually increase latency
        drifting_latency = 2.0 + (i * 0.2)
        system.drift_detector.add_observation("latency", drifting_latency)

    drift_report = system.drift_detector.get_drift_report()

    for metric, info in drift_report["drifts"].items():
        print(f"\n{metric}:")
        print(f"Baseline: {info['baseline']:.2f}")
        print(f"Current: {info['current']:.2f}")
        print(f"Drift: {info['drift_percentage']:.1f}%")
        if info["is_drifting"]:
            print(f"DRIFT DETECTED")

    # Test 5: Distributed tracing
    print("\nTEST 5: Distributed Tracing")
    print("-" * 40)

    trace_id = "trace_demo_001"

    # Create complex trace
    main_span = system.tracer.start_span("complex_operation", trace_id=trace_id)

    # Child spans
    for operation in ["fetch_context", "process_data", "generate_response"]:
        child_span = system.tracer.start_span(
            operation, trace_id=trace_id, parent_span_id=main_span.span_id
        )
        await asyncio.sleep(random.uniform(0.1, 0.5))
        system.tracer.end_span(child_span.span_id)

    system.tracer.end_span(main_span.span_id)

    # Get trace timeline
    timeline = system.tracer.get_trace_timeline(trace_id)

    print(f"\nTrace {trace_id} Timeline:")
    for event in timeline:
        print(f"  {event['operation']}: {event['duration']:.3f}s")

    # Test 6: System health & dashboard
    print("\nTEST 6: System Health & Dashboard")
    print("-" * 40)

    health = system.get_system_health()

    print(f"\nSystem Status: {health['status']}")

    if "components" in health:
        print("\nComponent Health:")
        for component, status in health["components"].items():
            print(
                f"{component}: {status['status']} (error rate: {status['error_rate']:.1%})"
            )

    if "active_alerts" in health:
        print("\nActive Alerts:")
        for alert in health["active_alerts"]:
            print(f"  - {alert['name']} ({alert['severity']})")

    if "drift_detected" in health:
        print(f"\nDrift Detected: {', '.join(health['drift_detected'])}")

    # Dashboard overview
    overview = system.dashboard.get_overview()

    print("\nDashboard Overview:")
    print(f"Active traces: {overview['active_traces']}")

    if overview["ai_performance"]:
        print("\nAI Performance:")
        for model, perf in overview["ai_performance"].items():
            print(f"{model}:")
            print(f"Requests: {perf['requests']}")
            print(f"Latency P50: {perf['latency_p50']:.2f}s")
            print(f"Cost: ${perf['cost']:.4f}")

    print("\nDEMONSTRATION COMPLETE")


# ============== Main Execution ==============

if __name__ == "__main__":
    print("Starting AI Monitoring System...")

    # Run demonstration
    asyncio.run(demonstrate_monitoring())
