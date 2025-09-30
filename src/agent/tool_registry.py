"""
Tool Registry for Agentic RAG
Manages available tools and their schemas for function calling.
"""

import logging
from typing import Dict, Any, List, Optional


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def register_tool(self, tool: Any) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance with name, description, execute(), and get_schema() methods
        """
        if not hasattr(tool, 'name'):
            raise ValueError("Tool must have a 'name' attribute")
        if not hasattr(tool, 'execute'):
            raise ValueError("Tool must have an 'execute' method")
        if not hasattr(tool, 'get_schema'):
            raise ValueError("Tool must have a 'get_schema' method")

        tool_name = tool.name
        self.tools[tool_name] = tool
        self.logger.info(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool from the registry."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, Any]:
        """Get all registered tools."""
        return self.tools.copy()

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """
        Get a formatted description of all available tools.
        Useful for including in agent prompts.
        """
        if not self.tools:
            return "No tools available."

        descriptions = ["Available tools:"]
        for tool_name, tool in self.tools.items():
            schema = tool.get_schema()
            desc = f"- {tool_name}: {schema.get('description', 'No description')}"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def get_tools_schema_json(self) -> List[Dict[str, Any]]:
        """
        Get JSON schemas for all tools (for function calling).

        Returns:
            List of tool schemas in OpenAI/Gemma function calling format
        """
        schemas = []
        for tool in self.tools.values():
            schema = tool.get_schema()
            schemas.append(schema)
        return schemas

    def validate_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> tuple:
        """
        Validate a tool call before execution.

        Args:
            tool_name: Name of tool to call
            parameters: Tool parameters

        Returns:
            Tuple of (is_valid: bool, error_message: str or None)
        """
        # Check if tool exists
        if tool_name not in self.tools:
            return False, f"Tool '{tool_name}' not found in registry"

        # Get tool schema
        tool = self.tools[tool_name]
        schema = tool.get_schema()
        required_params = schema.get('parameters', {})

        # Check required parameters
        for param_name, param_info in required_params.items():
            if param_info.get('required', False):
                if param_name not in parameters:
                    return False, f"Missing required parameter: {param_name}"

        return True, None


def create_tool_registry(tool_instances: Optional[List[Any]] = None) -> ToolRegistry:
    """
    Factory function to create a ToolRegistry.

    Args:
        tool_instances: Optional list of tool instances to register

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    if tool_instances:
        for tool in tool_instances:
            registry.register_tool(tool)

    return registry