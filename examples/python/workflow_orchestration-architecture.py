#!/usr/bin/env python3
"""
Workflow Orchestration System
Production-ready orchestration for complex AI pipelines with
dependency management, parallel execution, and state persistence.
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
from abc import ABC, abstractmethod
# ============== Core Workflow Components ==============
class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
class TriggerType(Enum):
    """Types of workflow triggers."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    DEPENDENCY = "dependency"
@dataclass
class Task:
    """Represents a single task in a workflow."""
    task_id: str
    name: str
    task_type: str  # 'llm', 'tool', 'agent', 'transform', 'condition'
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_retries": 3, "backoff": 2})
    timeout: int = 300  # seconds
    condition: Optional[Callable] = None  # For conditional tasks
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def can_run(self, completed_tasks: Set[str]) -> bool:
        """Check if task can run based on dependencies."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.attempts < self.retry_policy["max_retries"]
@dataclass
class Workflow:
    """Represents a complete workflow."""
    workflow_id: str
    name: str
    description: str
    tasks: Dict[str, Task]
    trigger: TriggerType
    schedule: Optional[str] = None  # Cron expression
    created_at: datetime = field(default_factory=datetime.now)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_dag(self) -> Dict[str, List[str]]:
        """Get DAG representation of workflow."""
        return {task_id: task.dependencies for task_id, task in self.tasks.items()}
    
    def validate(self) -> bool:
        """Validate workflow for cycles and missing dependencies."""
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in self.tasks[task_id].dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Missing dependency: {dep}")
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError("Workflow contains cycles")
        
        return True
@dataclass
class WorkflowRun:
    """Represents a single execution of a workflow."""
    run_id: str
    workflow_id: str
    trigger_type: TriggerType
    triggered_by: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.RUNNING
    task_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
# ============== Task Implementations ==============
class TaskExecutor(ABC):
    """Abstract base class for task executors."""
    
    @abstractmethod
    async def execute(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a task."""
        pass
class LLMTaskExecutor(TaskExecutor):
    """Executor for LLM tasks."""
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute an LLM task."""
        print(f"[LLM] Executing: {task.name}")
        await asyncio.sleep(0.5)  # Simulate LLM call
        
        # Mock LLM response based on task parameters
        prompt = task.parameters.get("prompt", "")
        model = task.parameters.get("model", "gpt-4")
        
        return {
            "response": f"LLM response to: {prompt[:50]}...",
            "model": model,
            "tokens": len(prompt.split()),
            "cost": 0.001
        }
class ToolTaskExecutor(TaskExecutor):
    """Executor for tool tasks."""
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a tool task."""
        print(f"[Tool] Executing: {task.name}")
        await asyncio.sleep(0.3)  # Simulate tool execution
        
        tool_name = task.parameters.get("tool", "unknown")
        
        # Mock tool executions
        if tool_name == "database_query":
            return {
                "data": [
                    {"id": 1, "value": 100},
                    {"id": 2, "value": 200},
                    {"id": 3, "value": 150}
                ],
                "rows": 3
            }
        elif tool_name == "web_search":
            query = task.parameters.get("query", "")
            return {
                "results": [
                    {"title": f"Result 1 for {query}", "url": "http://example.com/1"},
                    {"title": f"Result 2 for {query}", "url": "http://example.com/2"}
                ],
                "count": 2
            }
        elif tool_name == "file_write":
            return {"success": True, "path": task.parameters.get("path", "/tmp/file.txt")}
        else:
            return {"result": f"Tool {tool_name} executed"}
class TransformTaskExecutor(TaskExecutor):
    """Executor for data transformation tasks."""
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a transformation task."""
        print(f"[Transform] Executing: {task.name}")
        await asyncio.sleep(0.2)
        
        # Get input data from context
        input_key = task.parameters.get("input", "data")
        transform_type = task.parameters.get("type", "passthrough")
        
        input_data = context.get(input_key, [])
        
        # Apply transformation
        if transform_type == "aggregate":
            if isinstance(input_data, list):
                values = [item.get("value", 0) for item in input_data if isinstance(item, dict)]
                return {
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "count": len(values)
                }
        elif transform_type == "filter":
            threshold = task.parameters.get("threshold", 100)
            return [item for item in input_data if isinstance(item, dict) and item.get("value", 0) > threshold]
        elif transform_type == "format":
            return {"formatted": json.dumps(input_data, indent=2)}
        else:
            return input_data
class ConditionalTaskExecutor(TaskExecutor):
    """Executor for conditional tasks."""
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Any:
        print(f"[Condition] Evaluating: {task.name}")
        
        # Evaluate condition
        if task.condition:
            result = await task.condition(context)
        else:
            # Default condition based on parameters
            check_key = task.parameters.get("check", "value")
            operator = task.parameters.get("operator", ">")
            threshold = task.parameters.get("threshold", 0)
            
            # Get the value from context - handle nested results
            value = context.get(check_key, 0)
            
            # If value is a dict (from previous task result), extract a numeric value
            if isinstance(value, dict):
                # Priority-ordered extractors
                extractors = [
                    ("count", lambda v: v["count"]),
                    ("rows", lambda v: v["rows"]),
                    ("success", lambda v: 1 if v["success"] else 0),
                    ("results", lambda v: len(v["results"]) if isinstance(v["results"], list) else 0),
                    (None, lambda v: 1 if v else 0)  # Default
                ]
                
                for key, extractor in extractors:
                    if key is None or key in value:
                        value = extractor(value)
                        break
                    
            # Operator mapping
            operators = {
                ">": lambda v, t: v > t,
                "<": lambda v, t: v < t,
                "==": lambda v, t: v == t,
                ">=": lambda v, t: v >= t,
                "<=": lambda v, t: v <= t,
                "!=": lambda v, t: v != t
            }
            
            result = operators.get(operator, lambda v, t: True)(value, threshold)
        
        return {"condition_met": result, "branch": "true" if result else "false"}
# ============== Workflow Engine ==============
class WorkflowEngine:
    """
    Main workflow orchestration engine.
    Manages workflow execution, state, and coordination.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.workflows: Dict[str, Workflow] = {}
        self.runs: Dict[str, WorkflowRun] = {}
        self.executors: Dict[str, TaskExecutor] = {
            "llm": LLMTaskExecutor(),
            "tool": ToolTaskExecutor(),
            "transform": TransformTaskExecutor(),
            "condition": ConditionalTaskExecutor()
        }
        
        # State management
        self.state_store: Dict[str, Any] = {}
        
        # Metrics
        self.metrics = {
            "total_workflows": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_tasks": 0,
            "avg_run_time": 0
        }
        
        print(f"[WorkflowEngine] Initialized with {max_workers} workers")
    
    def register_workflow(self, workflow: Workflow) -> str:
        """Register a workflow in the engine."""
        workflow.validate()
        self.workflows[workflow.workflow_id] = workflow
        self.metrics["total_workflows"] += 1
        print(f"[WorkflowEngine] Registered workflow: {workflow.name}")
        return workflow.workflow_id
    
    async def run_workflow(self, 
                          workflow_id: str,
                          trigger_type: TriggerType = TriggerType.MANUAL,
                          triggered_by: str = "user",
                          initial_context: Dict[str, Any] = None) -> WorkflowRun:
        """
        Execute a workflow.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        run_id = f"run_{workflow_id}_{int(time.time())}"
        
        # Create run record
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=workflow_id,
            trigger_type=trigger_type,
            triggered_by=triggered_by
        )
        self.runs[run_id] = run
        
        print(f"\n{'='*60}")
        print(f"WORKFLOW RUN: {workflow.name}")
        print(f"Run ID: {run_id}")
        print(f"{'='*60}")
        
        # Initialize context
        context = initial_context or {}
        context["workflow_id"] = workflow_id
        context["run_id"] = run_id
        
        # Track metrics
        start_time = time.time()
        self.metrics["total_runs"] += 1
        
        try:
            # Execute workflow
            await self._execute_workflow(workflow, run, context)
            
            # Mark as successful
            run.status = TaskStatus.SUCCESS
            run.completed_at = datetime.now()
            self.metrics["successful_runs"] += 1
            
            print(f"\nWorkflow completed successfully")
            
        except Exception as e:
            # Mark as failed
            run.status = TaskStatus.FAILED
            run.completed_at = datetime.now()
            self.metrics["failed_runs"] += 1
            
            print(f"\nWorkflow failed: {e}")
            raise
        
        finally:
            # Update metrics
            run_time = time.time() - start_time
            run.metrics["run_time"] = run_time
            
            # Update average
            total_runs = self.metrics["successful_runs"] + self.metrics["failed_runs"]
            self.metrics["avg_run_time"] = (
                (self.metrics["avg_run_time"] * (total_runs - 1) + run_time) / total_runs
            )
        
        return run
    
    async def _execute_workflow(self, 
                               workflow: Workflow,
                               run: WorkflowRun,
                               context: Dict[str, Any]):
        """Execute workflow tasks respecting dependencies."""
        
        # Track completed tasks
        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        task_results = {}
        
        # Create task queue
        pending_tasks = list(workflow.tasks.keys())
        running_tasks = {}
        
        print("\nExecution Plan:")
        for task_id, task in workflow.tasks.items():
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  • {task.name}{deps}")
        
        print("\nExecution:")
        print("-" * 40)
        
        # Execute tasks
        while pending_tasks or running_tasks:
            # Start tasks that can run
            tasks_to_start = []
            for task_id in pending_tasks[:]:
                task = workflow.tasks[task_id]
                
                # Check if dependencies are met
                if task.can_run(completed_tasks):
                    # Check if dependencies failed
                    if any(dep in failed_tasks for dep in task.dependencies):
                        task.status = TaskStatus.SKIPPED
                        failed_tasks.add(task_id)
                        pending_tasks.remove(task_id)
                        print(f"Skipped: {task.name} (dependency failed)")
                    else:
                        tasks_to_start.append(task_id)
                        pending_tasks.remove(task_id)
            
            # Start parallel tasks (up to max_workers)
            for task_id in tasks_to_start[:self.max_workers - len(running_tasks)]:
                task = workflow.tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.start_time = datetime.now()
                
                # Update context with previous results
                for dep in task.dependencies:
                    if dep in task_results:
                        context[f"{dep}_result"] = task_results[dep]
                
                # Start task
                print(f"Starting: {task.name}")
                task_future = asyncio.create_task(
                    self._execute_task(task, context)
                )
                running_tasks[task_id] = task_future
            
            # Wait for any task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_future in done:
                    # Find which task completed
                    task_id = None
                    for tid, future in running_tasks.items():
                        if future == task_future:
                            task_id = tid
                            break
                    
                    if task_id:
                        task = workflow.tasks[task_id]
                        del running_tasks[task_id]
                        
                        try:
                            result = await task_future
                            task.status = TaskStatus.SUCCESS
                            task.result = result
                            task_results[task_id] = result
                            completed_tasks.add(task_id)
                            context[task_id] = result
                            
                            print(f"Completed: {task.name}")
                            
                            # Handle conditional branches
                            if task.task_type == "condition":
                                branch = result.get("branch", "true")
                                print(f"     → Branch: {branch}")
                                
                        except Exception as e:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            failed_tasks.add(task_id)
                            
                            print(f"Failed: {task.name} - {e}")
                            
                            # Retry logic
                            if task.should_retry():
                                task.attempts += 1
                                task.status = TaskStatus.RETRYING
                                pending_tasks.append(task_id)
                                print(f"     → Retrying ({task.attempts}/{task.retry_policy['max_retries']})")
                        
                        finally:
                            task.end_time = datetime.now()
                            self.metrics["total_tasks"] += 1
            
            # Small delay to prevent busy waiting
            if not running_tasks and pending_tasks:
                await asyncio.sleep(0.1)
        
        # Store results
        run.task_results = task_results
    
    async def _execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a single task."""
        # Get appropriate executor
        executor = self.executors.get(task.task_type)
        if not executor:
            raise ValueError(f"No executor for task type: {task.task_type}")
        
        # Apply timeout
        try:
            result = await asyncio.wait_for(
                executor.execute(task, context),
                timeout=task.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task timed out after {task.timeout}s")
# ============== Workflow Builder ==============
class WorkflowBuilder:
    """Helper class to build workflows programmatically."""
    
    def __init__(self, name: str, description: str = ""):
        self.workflow_id = f"wf_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        self.name = name
        self.description = description
        self.tasks = {}
        self.trigger = TriggerType.MANUAL
        self.schedule = None
    
    def add_task(self, 
                 task_id: str,
                 name: str,
                 task_type: str,
                 parameters: Dict[str, Any] = None,
                 dependencies: List[str] = None,
                 condition: Callable = None) -> 'WorkflowBuilder':
        """Add a task to the workflow."""
        self.tasks[task_id] = Task(
            task_id=task_id,
            name=name,
            task_type=task_type,
            parameters=parameters or {},
            dependencies=dependencies or [],
            condition=condition
        )
        return self
    
    def set_trigger(self, trigger: TriggerType, schedule: str = None) -> 'WorkflowBuilder':
        """Set workflow trigger."""
        self.trigger = trigger
        self.schedule = schedule
        return self
    
    def build(self) -> Workflow:
        """Build the workflow."""
        return Workflow(
            workflow_id=self.workflow_id,
            name=self.name,
            description=self.description,
            tasks=self.tasks,
            trigger=self.trigger,
            schedule=self.schedule
        )
# ============== Example Workflows ==============
def create_data_pipeline_workflow() -> Workflow:
    """Create a data processing pipeline workflow."""
    builder = WorkflowBuilder(
        "Data Processing Pipeline",
        "Extract, transform, and load data with analysis"
    )
    
    # Extract phase
    builder.add_task(
        "extract_sales",
        "Extract Sales Data",
        "tool",
        {"tool": "database_query", "query": "SELECT * FROM sales"}
    )
    
    builder.add_task(
        "extract_customers",
        "Extract Customer Data",
        "tool",
        {"tool": "database_query", "query": "SELECT * FROM customers"}
    )
    
    # Transform phase
    builder.add_task(
        "transform_sales",
        "Transform Sales Data",
        "transform",
        {"input": "extract_sales", "type": "aggregate"},
        dependencies=["extract_sales"]
    )
    
    builder.add_task(
        "filter_customers",
        "Filter High-Value Customers",
        "transform",
        {"input": "extract_customers", "type": "filter", "threshold": 100},
        dependencies=["extract_customers"]
    )
    
    # Analysis phase
    builder.add_task(
        "analyze_data",
        "Analyze Combined Data",
        "llm",
        {"prompt": "Analyze the sales and customer data", "model": "gpt-4"},
        dependencies=["transform_sales", "filter_customers"]
    )
    
    # Load phase
    builder.add_task(
        "save_results",
        "Save Analysis Results",
        "tool",
        {"tool": "file_write", "path": "/reports/analysis.json"},
        dependencies=["analyze_data"]
    )
    
    return builder.build()
def create_content_generation_workflow() -> Workflow:
    """Create a content generation workflow with conditional logic."""
    builder = WorkflowBuilder(
        "Content Generation Pipeline",
        "Research, write, and publish content"
    )
    
    # Research phase
    builder.add_task(
        "research",
        "Research Topic",
        "tool",
        {"tool": "web_search", "query": "latest AI trends"}
    )
    
    # Quality check
    builder.add_task(
        "check_quality",
        "Check Research Quality",
        "condition",
        {"check": "research_result", "operator": ">", "threshold": 0},
        dependencies=["research"]
    )
    
    # Content generation (conditional)
    builder.add_task(
        "generate_article",
        "Generate Article",
        "llm",
        {"prompt": "Write an article about AI trends", "model": "gpt-4"},
        dependencies=["check_quality"]
    )
    
    builder.add_task(
        "generate_summary",
        "Generate Summary",
        "llm",
        {"prompt": "Create a summary", "model": "gpt-3.5-turbo"},
        dependencies=["generate_article"]
    )
    
    # Publishing
    builder.add_task(
        "publish",
        "Publish Content",
        "tool",
        {"tool": "file_write", "path": "/content/article.md"},
        dependencies=["generate_article", "generate_summary"]
    )
    
    return builder.build()
def create_ml_training_workflow() -> Workflow:
    """Create an ML model training workflow."""
    builder = WorkflowBuilder(
        "ML Training Pipeline",
        "Train and evaluate machine learning models"
    )
    
    # Data preparation
    builder.add_task(
        "load_data",
        "Load Training Data",
        "tool",
        {"tool": "database_query", "query": "SELECT * FROM training_data"}
    )
    
    builder.add_task(
        "preprocess",
        "Preprocess Data",
        "transform",
        {"input": "load_data", "type": "format"},
        dependencies=["load_data"]
    )
    
    # Parallel model training
    builder.add_task(
        "train_model_a",
        "Train Model A",
        "llm",
        {"prompt": "Train classification model A", "model": "custom-ml"},
        dependencies=["preprocess"]
    )
    
    builder.add_task(
        "train_model_b",
        "Train Model B",
        "llm",
        {"prompt": "Train classification model B", "model": "custom-ml"},
        dependencies=["preprocess"]
    )
    
    # Evaluation
    builder.add_task(
        "evaluate",
        "Evaluate Models",
        "transform",
        {"input": "models", "type": "aggregate"},
        dependencies=["train_model_a", "train_model_b"]
    )
    
    # Select best model
    builder.add_task(
        "select_best",
        "Select Best Model",
        "condition",
        {"check": "accuracy", "operator": ">", "threshold": 0.9},
        dependencies=["evaluate"]
    )
    
    # Deploy
    builder.add_task(
        "deploy",
        "Deploy Model",
        "tool",
        {"tool": "model_deploy", "endpoint": "/api/model"},
        dependencies=["select_best"]
    )
    
    return builder.build()
# ============== Demo & Testing ==============
async def demonstrate_orchestration():
    """Demonstrate the workflow orchestration system."""
    print("\n" + "="*60)
    print("WORKFLOW ORCHESTRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize engine
    engine = WorkflowEngine(max_workers=5)
    
    # Test 1: Data Pipeline
    print("\nTEST 1: Data Processing Pipeline")
    print("-" * 40)
    
    data_workflow = create_data_pipeline_workflow()
    engine.register_workflow(data_workflow)
    
    run1 = await engine.run_workflow(
        data_workflow.workflow_id,
        initial_context={"environment": "production"}
    )
    
    print(f"\nResults: {len(run1.task_results)} tasks completed")
    print(f"Status: {run1.status.value}")
    print(f"Runtime: {run1.metrics.get('run_time', 0):.2f}s")
    
    # Test 2: Content Generation with Conditions
    print("\nTEST 2: Content Generation Pipeline")
    print("-" * 40)
    
    content_workflow = create_content_generation_workflow()
    engine.register_workflow(content_workflow)
    
    run2 = await engine.run_workflow(
        content_workflow.workflow_id,
        trigger_type=TriggerType.SCHEDULED,
        triggered_by="scheduler"
    )
    
    print(f"\nResults: {len(run2.task_results)} tasks completed")
    print(f"Status: {run2.status.value}")
    
    # Test 3: ML Training Pipeline
    print("\nTEST 3: ML Training Pipeline")
    print("-" * 40)
    
    ml_workflow = create_ml_training_workflow()
    engine.register_workflow(ml_workflow)
    
    run3 = await engine.run_workflow(
        ml_workflow.workflow_id,
        trigger_type=TriggerType.EVENT,
        triggered_by="data_update_event"
    )
    
    print(f"\nResults: {len(run3.task_results)} tasks completed")
    print(f"Status: {run3.status.value}")
    
    # Test 4: Parallel Workflow Execution
    print("\nTEST 4: Parallel Workflow Execution")
    print("-" * 40)
    
    # Run multiple workflows in parallel
    parallel_runs = await asyncio.gather(
        engine.run_workflow(data_workflow.workflow_id),
        engine.run_workflow(content_workflow.workflow_id),
        engine.run_workflow(ml_workflow.workflow_id)
    )
    
    print(f"\nExecuted {len(parallel_runs)} workflows in parallel")
    successful = sum(1 for run in parallel_runs if run.status == TaskStatus.SUCCESS)
    print(f"Successful: {successful}/{len(parallel_runs)}")
    
    # Display metrics
    print("\nENGINE METRICS")
    print("-" * 40)
    for metric, value in engine.metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    # Visualize a workflow
    print("\nWORKFLOW STRUCTURE")
    print("-" * 40)
    print(f"Workflow: {data_workflow.name}")
    dag = data_workflow.get_dag()
    for task_id, deps in dag.items():
        task = data_workflow.tasks[task_id]
        if deps:
            print(f"  {task.name} <- {', '.join(deps)}")
        else:
            print(f"  {task.name} (entry point)")
    
# ============== Advanced Features ==============
class WorkflowScheduler:
    """Schedule workflows based on cron expressions or events."""
    
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self.scheduled_jobs = {}
        self.running = False
        
    async def start(self):
        """Start the scheduler."""
        self.running = True
        print("[Scheduler] Started")
        
        while self.running:
            # Check scheduled workflows
            now = datetime.now()
            
            for workflow_id, schedule in self.scheduled_jobs.items():
                if self._should_run(schedule, now):
                    asyncio.create_task(
                        self.engine.run_workflow(
                            workflow_id,
                            trigger_type=TriggerType.SCHEDULED,
                            triggered_by="scheduler"
                        )
                    )
            
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        print("[Scheduler] Stopped")
    
    def schedule_workflow(self, workflow_id: str, cron: str):
        """Schedule a workflow with cron expression."""
        self.scheduled_jobs[workflow_id] = cron
        print(f"[Scheduler] Scheduled {workflow_id} with {cron}")
    
    def _should_run(self, schedule: str, now: datetime) -> bool:
        """Check if workflow should run based on schedule."""
        # Simplified cron parsing (production would use croniter)
        if schedule == "* * * * *":  # Every minute
            return True
        elif schedule == "0 * * * *":  # Every hour
            return now.minute == 0
        elif schedule == "0 0 * * *":  # Daily
            return now.hour == 0 and now.minute == 0
        return False
class WorkflowMonitor:
    """Monitor and visualize workflow execution."""
    
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics for a workflow."""
        runs = [r for r in self.engine.runs.values() if r.workflow_id == workflow_id]
        
        if not runs:
            return {"error": "No runs found"}
        
        successful = sum(1 for r in runs if r.status == TaskStatus.SUCCESS)
        failed = sum(1 for r in runs if r.status == TaskStatus.FAILED)
        
        return {
            "total_runs": len(runs),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(runs) if runs else 0,
            "avg_runtime": sum(r.metrics.get("run_time", 0) for r in runs) / len(runs)
        }
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task-level statistics."""
        task_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})
        
        for run in self.engine.runs.values():
            workflow = self.engine.workflows.get(run.workflow_id)
            if workflow:
                for task in workflow.tasks.values():
                    task_stats[task.task_type]["total"] += 1
                    if task.status == TaskStatus.SUCCESS:
                        task_stats[task.task_type]["success"] += 1
                    elif task.status == TaskStatus.FAILED:
                        task_stats[task.task_type]["failed"] += 1
        
        return dict(task_stats)
# ============== Main Execution ==============
if __name__ == "__main__":
    print("Starting Workflow Orchestration System...")
    
    # Run demonstration
    asyncio.run(demonstrate_orchestration())
    