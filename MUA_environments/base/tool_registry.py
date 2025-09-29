# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

class ToolRegistry:
    """Registry for managing tools within an environment.
    
    This class provides a centralized way to register, retrieve, and manage
    tools for a specific environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool registry.
        
        Args:
            config: Configuration dictionary for the tool registry
        """
        self.config = config
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_tool_class(self, tool_name: str, tool_class: Type[BaseTool], 
                          tool_config: Dict[str, Any]) -> None:
        """Register a tool class.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class to register
            tool_config: Configuration for the tool
        """
        self._tool_classes[tool_name] = tool_class
        self._tool_configs[tool_name] = tool_config
    
    def create_tool_instance(self, tool_name: str, **kwargs) -> Optional[BaseTool]:
        """Create an instance of a registered tool.
        
        Args:
            tool_name: Name of the tool to create
            **kwargs: Additional arguments for tool creation
            
        Returns:
            Tool instance if successful, None otherwise
        """
        if tool_name not in self._tool_classes:
            return None
        
        tool_class = self._tool_classes[tool_name]
        tool_config = self._tool_configs[tool_name].copy()
        tool_config.update(kwargs)
        
        try:
            # Create a temporary instance without calling __init__ to get the real schema
            temp_instance = tool_class.__new__(tool_class)
            temp_instance.config = tool_config
            temp_instance.tool_schema = None
            temp_instance._instance_dict = {}
            
            # Get the real schema from the temporary instance
            real_schema = temp_instance.get_openai_tool_schema()
            
            # Now create the real instance with the proper schema
            tool_instance = tool_class(config=tool_config, tool_schema=real_schema)
            
            return tool_instance
        except Exception as e:
            print(f"Failed to create tool instance for {tool_name}: {e}")
            return None
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        if tool_name not in self._tools:
            tool_instance = self.create_tool_instance(tool_name)
            if tool_instance:
                self._tools[tool_name] = tool_instance
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tool instances.
        
        Returns:
            List of all tool instances
        """
        tools = []
        for tool_name in self._tool_classes:
            tool = self.get_tool_by_name(tool_name)
            if tool:
                tools.append(tool)
        return tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tool_classes.keys())
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tool_classes:
            del self._tool_classes[tool_name]
            if tool_name in self._tool_configs:
                del self._tool_configs[tool_name]
            if tool_name in self._tools:
                del self._tools[tool_name]
            return True
        return False
    
    def reset_all_tools(self) -> None:
        """Reset all tool instances."""
        self._tools.clear()
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the tool registry.
        
        Returns:
            Dictionary containing registry metadata
        """
        return {
            "registered_tools": list(self._tool_classes.keys()),
            "active_instances": len(self._tools),
            "config": self.config,
        }
