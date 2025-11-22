#!/usr/bin/env python3
"""
AI Agent Implementation
A production-ready autonomous agent that combines multiple tools
to accomplish complex tasks with planning, execution, and reflection.
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import re

# ============== Core Agent Components ==============


class TaskStatus(Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Tool:
    """Represents a tool the agent can use."""

    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    returns: str
    cost: float = 0.0  # Cost per invocation (for optimization)
    reliability: float = 0.95  # Success rate (for planning)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        try:
            result = await self.function(**kwargs)
            return {
                "success": True,
                "result": result,
                "tool": self.name,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name,
                "timestamp": datetime.now().isoformat(),
            }


@dataclass
class TaskStep:
    """Represents a single step in a task plan."""

    step_id: int
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[int] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class TaskPlan:
    """Complete plan for executing a task."""

    task_id: str
    goal: str
    steps: List[TaskStep]
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PLANNING
    total_cost: float = 0.0
    estimated_time: float = 0.0


@dataclass
class Memory:
    """Agent's memory system."""

    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    tool_usage_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def remember_execution(self, tool: str, success: bool, duration: float):
        """Record tool execution statistics."""
        if tool not in self.tool_usage_stats:
            self.tool_usage_stats[tool] = {
                "successes": 0,
                "failures": 0,
                "total_duration": 0,
                "avg_duration": 0,
            }

        stats = self.tool_usage_stats[tool]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        stats["total_duration"] += duration
        total_executions = stats["successes"] + stats["failures"]
        stats["avg_duration"] = stats["total_duration"] / total_executions
        stats["success_rate"] = stats["successes"] / total_executions


# ============== Tool Implementations ==============


class ToolKit:
    """Collection of available tools for the agent."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_tools()
        print(f"[ToolKit] Initialized with {len(self.tools)} tools")

    def _register_tools(self):
        """Register all available tools."""

        # Search tool
        self.tools["web_search"] = Tool(
            name="web_search",
            description="Search the web for information",
            function=self._web_search,
            parameters={"query": {"type": "string", "description": "Search query"}},
            returns="List of search results",
            cost=0.001,
            reliability=0.98,
        )

        # Database tool
        self.tools["database_query"] = Tool(
            name="database_query",
            description="Query internal database",
            function=self._database_query,
            parameters={
                "sql": {"type": "string", "description": "SQL query"},
                "table": {"type": "string", "description": "Table name"},
            },
            returns="Query results",
            cost=0.0001,
            reliability=0.99,
        )

        # Calculator tool
        self.tools["calculator"] = Tool(
            name="calculator",
            description="Perform mathematical calculations",
            function=self._calculator,
            parameters={
                "expression": {"type": "string", "description": "Math expression"}
            },
            returns="Calculation result",
            cost=0.0,
            reliability=1.0,
        )

        # File tool
        self.tools["file_operation"] = Tool(
            name="file_operation",
            description="Read or write files",
            function=self._file_operation,
            parameters={
                "operation": {"type": "string", "enum": ["read", "write"]},
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content for write"},
            },
            returns="File operation result",
            cost=0.0,
            reliability=0.95,
        )

        # API tool
        self.tools["api_call"] = Tool(
            name="api_call",
            description="Make external API calls",
            function=self._api_call,
            parameters={
                "endpoint": {"type": "string", "description": "API endpoint"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
                "data": {"type": "object", "description": "Request data"},
            },
            returns="API response",
            cost=0.002,
            reliability=0.9,
        )

        # Data analysis tool
        self.tools["analyze_data"] = Tool(
            name="analyze_data",
            description="Analyze data and generate insights",
            function=self._analyze_data,
            parameters={
                "data": {"type": "array", "description": "Data to analyze"},
                "analysis_type": {"type": "string", "description": "Type of analysis"},
            },
            returns="Analysis results",
            cost=0.001,
            reliability=0.97,
        )

    async def _web_search(self, query: str) -> List[Dict[str, str]]:
        """Mock web search implementation."""
        await asyncio.sleep(0.5)  # Simulate API delay

        # Mock search results
        results = [
            {
                "title": f"Result for {query} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"Comprehensive information about {query}...",
            },
            {
                "title": f"{query} - Latest News",
                "url": f"https://news.example.com/{query.replace(' ', '-')}",
                "snippet": f"Recent developments regarding {query}...",
            },
            {
                "title": f"Understanding {query}",
                "url": f"https://blog.example.com/{query.replace(' ', '-')}",
                "snippet": f"In-depth analysis of {query} and its implications...",
            },
        ]
        return results

    async def _database_query(self, sql: str, table: str) -> List[Dict[str, Any]]:
        """Mock database query implementation."""
        await asyncio.sleep(0.2)

        # Mock database data
        if "sales" in table.lower():
            return [
                {"id": 1, "product": "Widget A", "quantity": 100, "revenue": 5000},
                {"id": 2, "product": "Widget B", "quantity": 150, "revenue": 7500},
                {"id": 3, "product": "Widget C", "quantity": 75, "revenue": 3750},
            ]
        elif "customers" in table.lower():
            return [
                {"id": 1, "name": "Acme Corp", "tier": "Enterprise", "value": 50000},
                {"id": 2, "name": "TechCo", "tier": "SMB", "value": 15000},
            ]
        else:
            return []

    async def _calculator(self, expression: str) -> float:
        """Safe calculator implementation."""
        await asyncio.sleep(0.1)

        # Safe evaluation of math expressions
        try:
            # Remove any non-math characters for safety
            safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)
            result = eval(safe_expr)
            return float(result)
        except:
            raise ValueError(f"Invalid expression: {expression}")

    async def _file_operation(
        self, operation: str, path: str, content: str = ""
    ) -> Dict[str, Any]:
        """Mock file operation implementation."""
        await asyncio.sleep(0.3)

        # Mock file system
        mock_files = {
            "/reports/sales.txt": "Q4 Sales: $2.5M, up 15% YoY",
            "/reports/customers.txt": "Total customers: 1,234",
            "/config/settings.json": '{"theme": "dark", "language": "en"}',
        }

        if operation == "read":
            if path in mock_files:
                return {"success": True, "content": mock_files[path]}
            else:
                return {"success": False, "error": "File not found"}
        elif operation == "write":
            mock_files[path] = content
            return {"success": True, "message": f"Written to {path}"}
        else:
            return {"success": False, "error": "Unknown operation"}

    async def _api_call(
        self, endpoint: str, method: str, data: Dict = None
    ) -> Dict[str, Any]:
        """Mock API call implementation."""
        await asyncio.sleep(0.7)

        # Mock API responses
        if "weather" in endpoint:
            return {"temperature": 22, "conditions": "Partly cloudy", "humidity": 65}
        elif "stock" in endpoint:
            return {"symbol": "AAPL", "price": 175.50, "change": 2.3}
        else:
            return {"status": "success", "data": data or {}}

    async def _analyze_data(
        self, data: List[Any], analysis_type: str
    ) -> Dict[str, Any]:
        """Mock data analysis implementation."""
        await asyncio.sleep(0.4)

        if analysis_type == "statistics":
            if data and isinstance(data[0], (int, float)):
                return {
                    "mean": sum(data) / len(data),
                    "min": min(data),
                    "max": max(data),
                    "count": len(data),
                }
        elif analysis_type == "trends":
            return {
                "trend": "increasing",
                "growth_rate": 0.15,
                "projection": "Positive outlook",
            }

        return {"analysis": "Complete", "insights": ["Key finding 1", "Key finding 2"]}


# ============== Agent Planner ==============


class TaskPlanner:
    """
    Plans how to accomplish tasks using available tools.
    Uses Chain of Thought (CoT) reasoning.
    """

    def __init__(self, toolkit: ToolKit):
        self.toolkit = toolkit
        print("TaskPlanner Initialized")

    async def create_plan(self, goal: str) -> TaskPlan:
        """
        Create a plan to accomplish the given goal.
        This is a simplified planner - production would use LLM.
        """
        print(f"TaskPlanner Creating plan for: {goal}")

        # Analyze goal to determine required tools
        goal_lower = goal.lower()
        steps = []
        step_id = 0

        # Pattern matching for common task types
        if "analyze" in goal_lower and "sales" in goal_lower:
            # Sales analysis workflow
            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Query sales data from database",
                    tool_name="database_query",
                    parameters={"sql": "SELECT * FROM sales", "table": "sales"},
                    dependencies=[],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Analyze sales data",
                    tool_name="analyze_data",
                    parameters={"data": [], "analysis_type": "statistics"},
                    dependencies=[step_id - 1],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Calculate growth rate",
                    tool_name="calculator",
                    parameters={"expression": "(7500 - 5000) / 5000 * 100"},
                    dependencies=[step_id - 1],
                )
            )

        elif "research" in goal_lower:
            # Research workflow
            search_term = goal.split("research")[-1].strip()

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description=f"Search for information about {search_term}",
                    tool_name="web_search",
                    parameters={"query": search_term},
                    dependencies=[],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Save research to file",
                    tool_name="file_operation",
                    parameters={
                        "operation": "write",
                        "path": "/reports/research.txt",
                        "content": "",
                    },
                    dependencies=[step_id - 1],
                )
            )

        elif "report" in goal_lower:
            # Report generation workflow
            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Gather sales data",
                    tool_name="database_query",
                    parameters={"sql": "SELECT * FROM sales", "table": "sales"},
                    dependencies=[],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Gather customer data",
                    tool_name="database_query",
                    parameters={"sql": "SELECT * FROM customers", "table": "customers"},
                    dependencies=[],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Analyze combined data",
                    tool_name="analyze_data",
                    parameters={"data": [], "analysis_type": "trends"},
                    dependencies=[0, 1],
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    step_id=step_id,
                    description="Write report to file",
                    tool_name="file_operation",
                    parameters={
                        "operation": "write",
                        "path": "/reports/analysis.txt",
                        "content": "",
                    },
                    dependencies=[step_id - 1],
                )
            )

        else:
            # Default single-step plan
            steps.append(
                TaskStep(
                    step_id=0,
                    description="Execute task",
                    tool_name="web_search",
                    parameters={"query": goal},
                    dependencies=[],
                )
            )

        # Calculate estimated cost and time
        total_cost = sum(self.toolkit.tools[step.tool_name].cost for step in steps)
        estimated_time = len(steps) * 0.5  # Rough estimate

        plan = TaskPlan(
            task_id=f"task_{int(time.time())}",
            goal=goal,
            steps=steps,
            total_cost=total_cost,
            estimated_time=estimated_time,
        )

        print(f"TaskPlanner Created plan with {len(steps)} steps")
        return plan


# ============== Agent Executor ==============


class TaskExecutor:
    """Executes task plans with error handling and retry logic."""

    def __init__(self, toolkit: ToolKit, memory: Memory):
        self.toolkit = toolkit
        self.memory = memory
        self.execution_history = []
        print("TaskExecutor Initialized")

    async def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute a task plan step by step."""
        print(f"TaskExecutor Executing plan: {plan.task_id}")
        plan.status = TaskStatus.EXECUTING

        results = {}
        failed_steps = []

        # Execute steps in dependency order
        for step in plan.steps:
            # Check dependencies
            ready = all(
                plan.steps[dep].status == TaskStatus.COMPLETED
                for dep in step.dependencies
            )

            if not ready:
                print(
                    f"TaskExecutor Skipping step {step.step_id} - dependencies not met"
                )
                continue

            # Execute step with retry logic
            success = False
            while step.attempts < step.max_attempts and not success:
                step.attempts += 1
                step.status = TaskStatus.EXECUTING

                print(f"TaskExecutor Executing step {step.step_id}: {step.description}")

                try:
                    # Get tool
                    tool = self.toolkit.tools.get(step.tool_name)
                    if not tool:
                        raise ValueError(f"Tool not found: {step.tool_name}")

                    # Prepare parameters (may use results from previous steps)
                    params = self._prepare_parameters(step, results)

                    # Execute tool
                    start_time = time.time()
                    result = await tool.execute(**params)
                    duration = time.time() - start_time

                    if result["success"]:
                        step.status = TaskStatus.COMPLETED
                        step.result = result["result"]
                        results[step.step_id] = result["result"]
                        success = True

                        # Record in memory
                        self.memory.remember_execution(tool.name, True, duration)

                        print(
                            f"TaskExecutor Step {step.step_id} completed successfully"
                        )
                    else:
                        raise Exception(result.get("error", "Unknown error"))

                except Exception as e:
                    print(f"TaskExecutor Step {step.step_id} failed: {e}")

                    if step.attempts >= step.max_attempts:
                        step.status = TaskStatus.FAILED
                        failed_steps.append(step.step_id)
                        self.memory.remember_execution(step.tool_name, False, 0)
                    else:
                        step.status = TaskStatus.RETRYING
                        await asyncio.sleep(1)  # Wait before retry

        # Update plan status
        if failed_steps:
            plan.status = TaskStatus.FAILED
        else:
            plan.status = TaskStatus.COMPLETED

        # Record execution
        execution_record = {
            "task_id": plan.task_id,
            "goal": plan.goal,
            "status": plan.status.value,
            "results": results,
            "failed_steps": failed_steps,
            "timestamp": datetime.now().isoformat(),
        }

        self.execution_history.append(execution_record)
        self.memory.short_term.append(execution_record)

        return execution_record

    def _prepare_parameters(
        self, step: TaskStep, results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for tool execution, using results from dependencies."""
        params = step.parameters.copy()

        # If step depends on previous results, inject them
        if step.dependencies:
            # For data analysis, use results from database query
            if step.tool_name == "analyze_data" and "data" in params:
                dependency_results = []
                for dep_id in step.dependencies:
                    if dep_id in results:
                        dep_result = results[dep_id]
                        if isinstance(dep_result, list):
                            # Extract numeric values for analysis
                            for item in dep_result:
                                if isinstance(item, dict) and "revenue" in item:
                                    dependency_results.append(item["revenue"])
                                elif isinstance(item, dict) and "value" in item:
                                    dependency_results.append(item["value"])

                if dependency_results:
                    params["data"] = dependency_results

            # For file write, use analysis results
            elif (
                step.tool_name == "file_operation"
                and params.get("operation") == "write"
            ):
                content_parts = []
                for dep_id in step.dependencies:
                    if dep_id in results:
                        content_parts.append(
                            f"Step {dep_id} results:\n{json.dumps(results[dep_id], indent=2)}"
                        )

                params["content"] = "\n\n".join(content_parts)

        return params


# ============== Reflection Engine ==============


class ReflectionEngine:
    """Analyzes execution results and improves future performance."""

    def __init__(self, memory: Memory):
        self.memory = memory
        print("ReflectionEngine Initialized")

    async def reflect(self, execution_record: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on execution and generate insights."""
        insights = {"successes": [], "failures": [], "improvements": [], "patterns": []}

        # Analyze successes
        if execution_record["status"] == "completed":
            insights["successes"].append(
                {
                    "goal": execution_record["goal"],
                    "approach": "Multi-step execution successful",
                }
            )

            # Store successful pattern
            pattern = {
                "goal_type": self._classify_goal(execution_record["goal"]),
                "tools_used": self._extract_tools_used(execution_record),
                "success": True,
            }
            self.memory.successful_patterns.append(pattern)

        # Analyze failures
        if execution_record["failed_steps"]:
            for step_id in execution_record["failed_steps"]:
                insights["failures"].append(
                    {
                        "step": step_id,
                        "recommendation": "Consider alternative tools or approaches",
                    }
                )

        # Generate improvements based on tool statistics
        for tool_name, stats in self.memory.tool_usage_stats.items():
            if stats["success_rate"] < 0.8:
                insights["improvements"].append(
                    {
                        "tool": tool_name,
                        "issue": "Low success rate",
                        "success_rate": stats["success_rate"],
                        "recommendation": "Review tool implementation or add fallbacks",
                    }
                )

            if stats["avg_duration"] > 2.0:
                insights["improvements"].append(
                    {
                        "tool": tool_name,
                        "issue": "Slow execution",
                        "avg_duration": stats["avg_duration"],
                        "recommendation": "Optimize or cache results",
                    }
                )

        # Identify patterns
        if len(self.memory.successful_patterns) >= 3:
            common_patterns = self._find_common_patterns()
            insights["patterns"] = common_patterns

        print(
            f"[ReflectionEngine] Generated {len(insights['improvements'])} improvements"
        )
        return insights

    def _classify_goal(self, goal: str) -> str:
        """Classify the type of goal."""
        goal_lower = goal.lower()
        if "analyze" in goal_lower:
            return "analysis"
        elif "research" in goal_lower:
            return "research"
        elif "report" in goal_lower:
            return "reporting"
        else:
            return "general"

    def _extract_tools_used(self, execution_record: Dict[str, Any]) -> List[str]:
        """Extract list of tools used in execution."""
        # In a real implementation, this would parse the execution steps
        return ["database_query", "analyze_data", "file_operation"]

    def _find_common_patterns(self) -> List[Dict[str, Any]]:
        """Identify common successful patterns."""
        patterns = []

        # Group by goal type
        goal_types = {}
        for pattern in self.memory.successful_patterns:
            goal_type = pattern["goal_type"]
            if goal_type not in goal_types:
                goal_types[goal_type] = []
            goal_types[goal_type].append(pattern["tools_used"])

        # Find most common tool combinations
        for goal_type, tool_lists in goal_types.items():
            if tool_lists:
                patterns.append(
                    {
                        "goal_type": goal_type,
                        "common_tools": tool_lists[0],  # Simplified
                        "success_count": len(tool_lists),
                    }
                )

        return patterns


# ============== Main Agent ==============


class AIAgent:
    """
    AI Agent that combines all components.
    Capable of planning, execution, and self-improvement.
    """

    def __init__(self, name: str = "Agent-001"):
        self.name = name
        self.toolkit = ToolKit()
        self.memory = Memory()
        self.planner = TaskPlanner(self.toolkit)
        self.executor = TaskExecutor(self.toolkit, self.memory)
        self.reflection = ReflectionEngine(self.memory)

        print(f"\n{'='*60}")
        print(f"AI AGENT '{self.name}' INITIALIZED")
        print(f"{'='*60}")
        print(f"Available tools: {list(self.toolkit.tools.keys())}")
        print(f"Ready for complex task execution")
        print()

    async def execute_task(self, goal: str) -> Dict[str, Any]:
        """
        Execute a complex task end-to-end.
        Includes planning, execution, and reflection.
        """
        print(f"\n{'='*60}")
        print(f"TASK: {goal}")
        print(f"{'='*60}")

        start_time = time.time()

        # Step 1: Create plan
        print("\nPLANNING PHASE")
        print("-" * 40)
        plan = await self.planner.create_plan(goal)

        print(f"Plan ID: {plan.task_id}")
        print(f"Estimated cost: ${plan.total_cost:.4f}")
        print(f"Estimated time: {plan.estimated_time:.1f}s")
        print(f"\nSteps to execute:")
        for step in plan.steps:
            deps = f"(depends on: {step.dependencies})" if step.dependencies else ""
            print(f"{step.step_id}. {step.description}{deps}")

        # Step 2: Execute plan
        print("\nEXECUTION PHASE")
        print("-" * 40)
        execution_result = await self.executor.execute_plan(plan)

        # Step 3: Reflect on execution
        print("\nREFLECTION PHASE")
        print("-" * 40)
        insights = await self.reflection.reflect(execution_result)

        # Prepare final result
        total_time = time.time() - start_time

        final_result = {
            "agent": self.name,
            "goal": goal,
            "plan": {
                "task_id": plan.task_id,
                "steps": len(plan.steps),
                "estimated_cost": plan.total_cost,
            },
            "execution": {
                "status": execution_result["status"],
                "results": execution_result["results"],
                "failed_steps": execution_result["failed_steps"],
            },
            "insights": insights,
            "performance": {
                "total_time": total_time,
                "tool_stats": self.memory.tool_usage_stats,
            },
        }

        # Print summary
        print(f"\nEXECUTION COMPLETE")
        print(f"Status: {execution_result['status']}")
        print(f"Time taken: {total_time:.2f}s")

        if execution_result["results"]:
            print(f"\nResults obtained:")
            for step_id, result in execution_result["results"].items():
                print(f"  Step {step_id}: {str(result)[:100]}...")

        if insights["improvements"]:
            print(f"\nSuggested improvements:")
            for imp in insights["improvements"]:
                print(f"  - {imp['tool']}: {imp['recommendation']}")

        return final_result

    async def batch_execute(self, goals: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel."""
        print(f"\n{'='*60}")
        print(f"BATCH EXECUTION: {len(goals)} tasks")
        print(f"{'='*60}")

        tasks = [self.execute_task(goal) for goal in goals]
        results = await asyncio.gather(*tasks)

        # Summary statistics
        successful = sum(1 for r in results if r["execution"]["status"] == "completed")
        print(f"\nBATCH SUMMARY")
        print(f"Total tasks: {len(goals)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(goals) - successful}")

        return results


# ============== Demo & Testing ==============


async def demonstrate_agent():
    """
    Demonstrate the AI agent's capabilities with various tasks.
    """
    print("\n" + "=" * 60)
    print("AI AGENT DEMONSTRATION")
    print("=" * 60)

    # Initialize agent
    agent = AIAgent("ResearchBot-3000")

    # Test Case 1: Sales Analysis
    print("\nTEST 1: Sales Analysis")
    result1 = await agent.execute_task("Analyze sales data and calculate growth")

    # Test Case 2: Research Task
    print("\nTEST 2: Research Task")
    result2 = await agent.execute_task("Research artificial intelligence trends")

    # Test Case 3: Report Generation
    print("\nTEST 3: Report Generation")
    result3 = await agent.execute_task("Generate a comprehensive business report")

    # Test Case 4: Batch Processing
    print("\nTEST 4: Batch Processing")
    batch_goals = [
        "Calculate revenue metrics",
        "Research market trends",
        "Analyze customer data",
    ]
    batch_results = await agent.batch_execute(batch_goals)

    # Final Statistics
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nAGENT PERFORMANCE METRICS")
    print("-" * 40)

    # Tool usage statistics
    print("Tool Usage Statistics:")
    for tool_name, stats in agent.memory.tool_usage_stats.items():
        print(f"  {tool_name}:")
        print(f"    Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"    Avg duration: {stats.get('avg_duration', 0):.2f}s")

    # Memory statistics
    print(f"\nMemory Statistics:")
    print(f"  Short-term memories: {len(agent.memory.short_term)}")
    print(f"  Successful patterns: {len(agent.memory.successful_patterns)}")

    # Insights
    if agent.memory.successful_patterns:
        print(f"\nLearned Patterns:")
        for pattern in agent.memory.successful_patterns[:3]:
            print(f"  - {pattern['goal_type']}: {pattern.get('tools_used', [])[:3]}")


# ============== Main Execution ==============

if __name__ == "__main__":
    print("Starting AI Agent System...")

    # Run the demonstration
    asyncio.run(demonstrate_agent())
