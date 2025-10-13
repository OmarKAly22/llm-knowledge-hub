# examples/python/simple-agent/simple_agent.py
"""
Simple ReAct Agent Implementation
Demonstrates the basic agent loop with tools
"""

from typing import List, Dict, Any
import json

class SimpleTool:
    """Base class for tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError

class SearchTool(SimpleTool):
    """Simple search tool"""
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information. Use when you need to find facts."
        )
    
    def execute(self, query: str) -> str:
        # Simulate search
        return f"Search results for '{query}': Found relevant information..."

class CalculatorTool(SimpleTool):
    """Simple calculator tool"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Use for math operations."
        )
    
    def execute(self, expression: str) -> str:
        try:
            result = eval(expression)  # Note: Use safe_eval in production!
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

class SimpleAgent:
    """
    A simple ReAct agent that can use tools to accomplish tasks
    """
    def __init__(self, tools: List[SimpleTool], max_iterations: int = 10):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.history = []
    
    def run(self, task: str) -> str:
        """
        Execute the agent loop
        """
        print(f"\n🎯 Task: {task}\n")
        
        for iteration in range(self.max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            
            # Think: Decide what to do next
            thought = self._think(task, self.history)
            print(f"💭 Thought: {thought}")
            
            # Check if done
            if "FINAL ANSWER" in thought:
                answer = thought.split("FINAL ANSWER:")[-1].strip()
                print(f"\n✅ Final Answer: {answer}")
                return answer
            
            # Act: Choose and execute a tool
            action = self._choose_action(thought)
            if not action:
                print("⚠️ No valid action found")
                continue
            
            print(f"🔧 Action: {action['tool']}({action['input']})")
            
            # Execute tool
            observation = self._execute_tool(action)
            print(f"👁️ Observation: {observation}\n")
            
            # Record in history
            self.history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })
        
        return "Max iterations reached without finding answer"
    
    def _think(self, task: str, history: List[Dict]) -> str:
        """
        Simulate LLM reasoning
        In production, this would be an actual LLM call
        """
        # Simple rule-based thinking for demonstration
        if not history:
            if "calculate" in task.lower() or any(char.isdigit() for char in task):
                return "I need to use the calculator tool to solve this math problem"
            else:
                return "I need to search for information about this topic"
        
        # After first iteration, provide final answer
        return f"FINAL ANSWER: Based on the observation, here is the answer to '{task}'"
    
    def _choose_action(self, thought: str) -> Dict[str, Any]:
        """
        Extract action from thought
        """
        thought_lower = thought.lower()
        
        if "calculator" in thought_lower:
            return {
                "tool": "calculator",
                "input": "2 + 2"  # Simplified for demo
            }
        elif "search" in thought_lower:
            return {
                "tool": "search",
                "input": "relevant query"
            }
        
        return None
    
    def _execute_tool(self, action: Dict[str, Any]) -> str:
        """
        Execute the chosen tool
        """
        tool_name = action["tool"]
        tool_input = action["input"]
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self.tools[tool_name]
        
        try:
            if tool_name == "calculator":
                return tool.execute(expression=tool_input)
            elif tool_name == "search":
                return tool.execute(query=tool_input)
            else:
                return tool.execute(input=tool_input)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Create tools
    tools = [
        SearchTool(),
        CalculatorTool()
    ]
    
    # Create agent
    agent = SimpleAgent(tools=tools, max_iterations=5)
    
    # Run tasks
    print("=" * 60)
    result1 = agent.run("What is the capital of France?")
    
    print("\n" + "=" * 60)
    agent.history = []  # Reset history
    result2 = agent.run("Calculate 15 * 23")
    
    print("\n" + "=" * 60)
    print("\n🎉 Agent demo complete!")