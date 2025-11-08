#!/usr/bin/env python3
"""
MCP Server Implementation(Mock Data)
A production-ready Model Context Protocol server that provides
database access, API integration, and file system tools to LLMs.
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import random
import hashlib

# ============== MCP Protocol Core ==============
class MCPError(Exception):
    """Base exception for MCP errors."""
    pass

class JSONRPCError(Enum):
    """Standard JSON-RPC error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

@dataclass
class Tool:
    """Represents an MCP tool that LLMs can invoke."""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_manifest(self) -> Dict[str, Any]:
        """Convert to MCP tool manifest format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() 
                           if v.get("required", False)]
            },
            "outputSchema": self.returns,
            "examples": self.examples
        }

@dataclass
class Resource:
    """Represents an MCP resource."""
    uri: str
    name: str
    mime_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPRequest:
    """Incoming MCP request."""
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: Optional[Union[str, int]] = None

@dataclass
class MCPResponse:
    """Outgoing MCP response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

# ============== Mock Data Store ==============
class MockDataStore:
    """Simulated database with business data."""
    
    def __init__(self):
        # Mock sales data
        self.sales_data = [
            {"id": 1, "quarter": "Q1", "year": 2024, "revenue": 1500000, "units": 1200},
            {"id": 2, "quarter": "Q2", "year": 2024, "revenue": 1750000, "units": 1400},
            {"id": 3, "quarter": "Q3", "year": 2024, "revenue": 2000000, "units": 1600},
            {"id": 4, "quarter": "Q4", "year": 2024, "revenue": 2300000, "units": 1850},
            {"id": 5, "quarter": "Q1", "year": 2025, "revenue": 2100000, "units": 1700},
        ]
        
        # Mock customer data
        self.customers = [
            {"id": 1, "name": "Acme Corp", "tier": "Enterprise", "mrr": 50000},
            {"id": 2, "name": "TechStart", "tier": "Startup", "mrr": 2000},
            {"id": 3, "name": "Global Inc", "tier": "Enterprise", "mrr": 75000},
            {"id": 4, "name": "SmallBiz Co", "tier": "SMB", "mrr": 5000},
        ]
        
        # Mock API keys for external services
        self.api_keys = {
            "weather_api": "mock_key_123",
            "stock_api": "mock_key_456"
        }
        
        # Mock file system
        self.files = {
            "/reports/q4_summary.txt": "Q4 2024 was our best quarter with $2.3M in revenue.",
            "/reports/customer_analysis.txt": "Enterprise customers contribute 70% of revenue.",
            "/configs/settings.json": json.dumps({"theme": "dark", "notifications": True})
        }
    
    def query_sales(self, quarter: Optional[str] = None, year: Optional[int] = None) -> List[Dict]:
        """Query sales data with filters."""
        results = self.sales_data
        if quarter:
            results = [s for s in results if s["quarter"] == quarter]
        if year:
            results = [s for s in results if s["year"] == year]
        return results
    
    def get_customers(self, tier: Optional[str] = None) -> List[Dict]:
        """Get customer data with optional tier filter."""
        if tier:
            return [c for c in self.customers if c["tier"] == tier]
        return self.customers
    
    def read_file(self, path: str) -> Optional[str]:
        """Read a file from mock filesystem."""
        return self.files.get(path)
    
    def write_file(self, path: str, content: str) -> bool:
        """Write to mock filesystem."""
        self.files[path] = content
        return True
    
# ============== MCP Server Implementation ==============
class MCPServer:
    """
    MCP Server implementation.
    Handles tool registration, request routing, and response formatting.
    """
    
    def __init__(self, name: str = "BusinessDataMCP"):
        self.name = name
        self.version = "1.0.0"
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.data_store = MockDataStore()
        self.sessions: Dict[str, Dict] = {}  # Track client sessions
        
        # Performance metrics
        self.metrics = {
            "requests_handled": 0,
            "errors": 0,
            "avg_response_time": 0,
            "cache_hits": 0
        }
        
        # Initialize tools
        self._register_tools()
        self._register_resources()
        
        print(f"[MCP Server] Initialized: {self.name} v{self.version}")
        print(f"[MCP Server] Registered {len(self.tools)} tools")
        print(f"[MCP Server] Available tools: {', '.join(self.tools.keys())}")
    
    def _register_tools(self):
        """Register all available tools."""
        
        # Sales data tool
        self.tools["get_sales_data"] = Tool(
            name="get_sales_data",
            description="Retrieve sales data by quarter and year",
            parameters={
                "quarter": {
                    "type": "string",
                    "description": "Quarter (Q1, Q2, Q3, Q4)",
                    "enum": ["Q1", "Q2", "Q3", "Q4"]
                },
                "year": {
                    "type": "integer",
                    "description": "Year (e.g., 2024)"
                }
            },
            returns={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quarter": {"type": "string"},
                        "year": {"type": "integer"},
                        "revenue": {"type": "number"},
                        "units": {"type": "integer"}
                    }
                }
            },
            examples=[
                {
                    "input": {"quarter": "Q4", "year": 2024},
                    "output": [{"quarter": "Q4", "year": 2024, "revenue": 2300000, "units": 1850}]
                }
            ]
        )
        
        # Customer analysis tool
        self.tools["analyze_customers"] = Tool(
            name="analyze_customers",
            description="Analyze customer data by tier",
            parameters={
                "tier": {
                    "type": "string",
                    "description": "Customer tier filter",
                    "enum": ["Enterprise", "SMB", "Startup"]
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include detailed metrics",
                    "default": False
                }
            },
            returns={
                "type": "object",
                "properties": {
                    "customers": {"type": "array"},
                    "total_mrr": {"type": "number"},
                    "customer_count": {"type": "integer"}
                }
            }
        )
        
        # File operations tool
        self.tools["file_operation"] = Tool(
            name="file_operation",
            description="Read or write files",
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Operation type"
                },
                "path": {
                    "type": "string",
                    "description": "File path"
                },
                "content": {
                    "type": "string",
                    "description": "Content for write operations"
                }
            },
            returns={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        )
        
        # Calculation tool
        self.tools["calculate_growth"] = Tool(
            name="calculate_growth",
            description="Calculate growth metrics between periods",
            parameters={
                "start_value": {
                    "type": "number",
                    "description": "Starting value"
                },
                "end_value": {
                    "type": "number",
                    "description": "Ending value"
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of periods",
                    "default": 1
                }
            },
            returns={
                "type": "object",
                "properties": {
                    "absolute_change": {"type": "number"},
                    "percent_change": {"type": "number"},
                    "cagr": {"type": "number"}
                }
            }
        )
    
    def _register_resources(self):
        """Register available resources."""
        
        self.resources["sales_database"] = Resource(
            uri="mcp://data/sales",
            name="Sales Database",
            mime_type="application/json",
            description="Historical sales data"
        )
        
        self.resources["customer_database"] = Resource(
            uri="mcp://data/customers",
            name="Customer Database",
            mime_type="application/json",
            description="Customer information and tiers"
        )
        
        self.resources["reports"] = Resource(
            uri="mcp://files/reports",
            name="Report Files",
            mime_type="text/plain",
            description="Business reports and summaries"
        )
    
    async def handle_request(self, request_data: str) -> str:
        """
        Main request handler for MCP protocol.
        Processes JSON-RPC requests and returns responses.
        """
        start_time = datetime.now()
        
        try:
            # Parse JSON-RPC request
            data = json.loads(request_data)
            request = MCPRequest(
                jsonrpc=data.get("jsonrpc", "2.0"),
                method=data["method"],
                params=data.get("params", {}),
                id=data.get("id")
            )
            
            print(f"[MCP Server] Received: {request.method}")
            
            # Route request to appropriate handler
            if request.method == "initialize":
                result = await self._handle_initialize(request.params)
            elif request.method == "tools/list":
                result = await self._handle_list_tools()
            elif request.method == "tools/invoke":
                result = await self._handle_invoke_tool(request.params)
            elif request.method == "resources/list":
                result = await self._handle_list_resources()
            elif request.method == "resources/read":
                result = await self._handle_read_resource(request.params)
            else:
                raise MCPError(f"Unknown method: {request.method}")
            
            # Update metrics
            self.metrics["requests_handled"] += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["requests_handled"] - 1) + elapsed)
                / self.metrics["requests_handled"]
            )
            
            # Build response
            response = MCPResponse(
                result=result,
                id=request.id
            )
            
            print(f"[MCP Server] Response sent in {elapsed:.3f}s")
            
        except json.JSONDecodeError as e:
            response = MCPResponse(
                error={
                    "code": JSONRPCError.PARSE_ERROR.value,
                    "message": f"Parse error: {str(e)}"
                }
            )
            self.metrics["errors"] += 1
            
        except MCPError as e:
            response = MCPResponse(
                error={
                    "code": JSONRPCError.METHOD_NOT_FOUND.value,
                    "message": str(e)
                },
                id=request.id if 'request' in locals() else None
            )
            self.metrics["errors"] += 1
            
        except Exception as e:
            response = MCPResponse(
                error={
                    "code": JSONRPCError.INTERNAL_ERROR.value,
                    "message": f"Internal error: {str(e)}"
                },
                id=request.id if 'request' in locals() else None
            )
            self.metrics["errors"] += 1
            print(f"[MCP Server] Error: {e}")
        
        return json.dumps(asdict(response))
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        client_id = params.get("clientId", "unknown")
        
        # Create session
        session_id = hashlib.md5(f"{client_id}_{datetime.now()}".encode()).hexdigest()[:8]
        self.sessions[session_id] = {
            "client_id": client_id,
            "created_at": datetime.now().isoformat(),
            "requests": 0
        }
        
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "tools": True,
                "resources": True,
                "context": True,
                "streaming": False
            },
            "sessionId": session_id
        }
    
    async def _handle_list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return {
            "tools": [tool.to_manifest() for tool in self.tools.values()]
        }
    
    async def _handle_invoke_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a specific tool."""
        tool_name = params.get("name")
        tool_params = params.get("parameters", {})
        
        if tool_name not in self.tools:
            raise MCPError(f"Tool not found: {tool_name}")
        
        print(f"[MCP Server] Invoking tool: {tool_name} with params: {tool_params}")
        
        # Execute tool based on name
        if tool_name == "get_sales_data":
            result = self._execute_sales_query(tool_params)
        elif tool_name == "analyze_customers":
            result = self._execute_customer_analysis(tool_params)
        elif tool_name == "file_operation":
            result = self._execute_file_operation(tool_params)
        elif tool_name == "calculate_growth":
            result = self._execute_calculation(tool_params)
        else:
            raise MCPError(f"Tool implementation not found: {tool_name}")
        
        return {
            "result": result,
            "metadata": {
                "executionTime": random.uniform(0.1, 0.5),  # Mock execution time
                "cached": random.choice([True, False])
            }
        }
    
    async def _handle_list_resources(self) -> Dict[str, Any]:
        """List available resources."""
        return {
            "resources": [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "mimeType": r.mime_type,
                    "description": r.description
                }
                for r in self.resources.values()
            ]
        }
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific resource."""
        uri = params.get("uri")
        
        # Mock resource reading based on URI
        if uri == "mcp://data/sales":
            data = self.data_store.sales_data
        elif uri == "mcp://data/customers":
            data = self.data_store.customers
        elif uri.startswith("mcp://files/"):
            path = uri.replace("mcp://files", "")
            data = self.data_store.read_file(path)
        else:
            raise MCPError(f"Resource not found: {uri}")
        
        return {
            "content": data,
            "metadata": {
                "size": len(str(data)),
                "lastModified": datetime.now().isoformat()
            }
        }
    
    def _execute_sales_query(self, params: Dict[str, Any]) -> List[Dict]:
        """Execute sales data query."""
        quarter = params.get("quarter")
        year = params.get("year")
        return self.data_store.query_sales(quarter, year)
    
    def _execute_customer_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute customer analysis."""
        tier = params.get("tier")
        include_metrics = params.get("include_metrics", False)
        
        customers = self.data_store.get_customers(tier)
        total_mrr = sum(c["mrr"] for c in customers)
        
        result = {
            "customers": customers,
            "total_mrr": total_mrr,
            "customer_count": len(customers)
        }
        
        if include_metrics:
            result["metrics"] = {
                "average_mrr": total_mrr / len(customers) if customers else 0,
                "tier_distribution": {}
            }
            
            # Calculate tier distribution
            for customer in customers:
                tier = customer["tier"]
                result["metrics"]["tier_distribution"][tier] = \
                    result["metrics"]["tier_distribution"].get(tier, 0) + 1
        
        return result
    
    def _execute_file_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file operation."""
        operation = params.get("operation")
        path = params.get("path")
        
        if operation == "read":
            content = self.data_store.read_file(path)
            if content:
                return {
                    "success": True,
                    "data": content,
                    "message": f"File read successfully: {path}"
                }
            else:
                return {
                    "success": False,
                    "data": "",
                    "message": f"File not found: {path}"
                }
        elif operation == "write":
            content = params.get("content", "")
            success = self.data_store.write_file(path, content)
            return {
                "success": success,
                "data": content if success else "",
                "message": f"File {'written' if success else 'write failed'}: {path}"
            }
        else:
            return {
                "success": False,
                "data": "",
                "message": f"Unknown operation: {operation}"
            }
    
    def _execute_calculation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute growth calculation."""
        start = params.get("start_value", 0)
        end = params.get("end_value", 0)
        periods = params.get("periods", 1)
        
        if start == 0:
            return {
                "absolute_change": end,
                "percent_change": float('inf'),
                "cagr": 0
            }
        
        absolute_change = end - start
        percent_change = (absolute_change / start) * 100
        
        # Calculate CAGR (Compound Annual Growth Rate)
        if periods > 0 and start > 0:
            cagr = (((end / start) ** (1 / periods)) - 1) * 100
        else:
            cagr = 0
        
        return {
            "absolute_change": absolute_change,
            "percent_change": round(percent_change, 2),
            "cagr": round(cagr, 2)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        return self.metrics
# ============== Demo & Testing ==============
async def simulate_llm_interaction():
    """
    Simulate an LLM interacting with the MCP server.
    This demonstrates the full conversation flow.
    """
    print("\n" + "="*60)
    print("MCP SERVER DEMONSTRATION")
    print("="*60)
    
    # Initialize server
    server = MCPServer("BusinessDataMCP")
    
    print("\n[Simulated LLM] Starting interaction...")
    print("-"*40)
    
    # 1. Initialize connection
    print("\n1. INITIALIZATION")
    init_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "clientId": "llm-client-001"
        },
        "id": 1
    })
    
    response = await server.handle_request(init_request)
    print(f"Response: {json.loads(response)['result']}")
    
    # 2. Discover available tools
    print("\n2. TOOL DISCOVERY")
    tools_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    })
    
    response = await server.handle_request(tools_request)
    tools = json.loads(response)['result']['tools']
    print(f"Available tools: {[t['name'] for t in tools]}")
    
    # 3. Query sales data
    print("\n3. SALES DATA QUERY")
    print("[User] 'What were our Q4 2024 sales?'")
    print("[LLM] I'll check the Q4 2024 sales data for you...")
    
    sales_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/invoke",
        "params": {
            "name": "get_sales_data",
            "parameters": {
                "quarter": "Q4",
                "year": 2024
            }
        },
        "id": 3
    })
    
    response = await server.handle_request(sales_request)
    sales_data = json.loads(response)['result']['result']
    print(f"[LLM] Q4 2024 Results:")
    for item in sales_data:
        print(f"      Revenue: ${item['revenue']:,}")
        print(f"      Units sold: {item['units']:,}")
    
    # 4. Customer analysis
    print("\n4. CUSTOMER ANALYSIS")
    print("[User] 'Show me our enterprise customers'")
    print("[LLM] Let me analyze our enterprise customers...")
    
    customer_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/invoke",
        "params": {
            "name": "analyze_customers",
            "parameters": {
                "tier": "Enterprise",
                "include_metrics": True
            }
        },
        "id": 4
    })
    
    response = await server.handle_request(customer_request)
    customer_data = json.loads(response)['result']['result']
    print(f"[LLM] Enterprise Customer Analysis:")
    print(f"      Total customers: {customer_data['customer_count']}")
    print(f"      Total MRR: ${customer_data['total_mrr']:,}")
    if 'metrics' in customer_data:
        print(f"      Average MRR: ${customer_data['metrics']['average_mrr']:,}")
    
    # 5. Growth calculation
    print("\n5. GROWTH CALCULATION")
    print("[User] 'Calculate growth from Q3 to Q4'")
    print("[LLM] I'll calculate the growth metrics...")
    
    growth_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/invoke",
        "params": {
            "name": "calculate_growth",
            "parameters": {
                "start_value": 2000000,
                "end_value": 2300000,
                "periods": 1
            }
        },
        "id": 5
    })
    
    response = await server.handle_request(growth_request)
    growth_data = json.loads(response)['result']['result']
    print(f"[LLM] Q3 to Q4 Growth:")
    print(f"      Absolute change: ${growth_data['absolute_change']:,}")
    print(f"      Percent change: {growth_data['percent_change']}%")
    
    # 6. File operation
    print("\n6. FILE OPERATION")
    print("[User] 'Read the Q4 summary report'")
    print("[LLM] I'll retrieve the Q4 summary report...")
    
    file_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/invoke",
        "params": {
            "name": "file_operation",
            "parameters": {
                "operation": "read",
                "path": "/reports/q4_summary.txt"
            }
        },
        "id": 6
    })
    
    response = await server.handle_request(file_request)
    file_data = json.loads(response)['result']['result']
    if file_data['success']:
        print(f"[LLM] Q4 Summary Report:")
        print(f"      {file_data['data']}")
    
    # 7. Show metrics
    print("\n7. SERVER METRICS")
    metrics = server.get_metrics()
    print(f"Performance Statistics:")
    print(f"  Requests handled: {metrics['requests_handled']}")
    print(f"  Errors: {metrics['errors']}")
    print(f"  Avg response time: {metrics['avg_response_time']:.3f}s")

# ============== Main Execution ==============
if __name__ == "__main__":
    print("Starting MCP Server Demo...")
    
    # Run the simulation
    asyncio.run(simulate_llm_interaction())