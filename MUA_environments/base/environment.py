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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from .data_loader import BaseDataLoader
from .tool_registry import ToolRegistry


class BaseEnvironment(ABC):
    """Abstract base class for environment databases.
    
    This class defines the interface that all environment implementations
    must follow. It provides a unified way to manage environment-specific
    data and tools.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the environment.
        
        Args:
            config: Configuration dictionary for the environment
        """
        self.config = config
        self.environment_id = str(uuid4())
        self._data_loader: Optional[BaseDataLoader] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._loaded_data: Optional[Dict[str, Any]] = None
        
    @property
    @abstractmethod
    def environment_name(self) -> str:
        """Return the name of this environment."""
        pass
        
    @property
    @abstractmethod
    def environment_type(self) -> str:
        """Return the type of this environment (e.g., 'retail', 'airline')."""
        pass
    
    @abstractmethod
    def get_data_loader(self) -> BaseDataLoader:
        """Get the data loader for this environment."""
        pass
    
    @abstractmethod
    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for this environment."""
        pass
    
    def load_data(self) -> Dict[str, Any]:
        """Load environment-specific data.
        
        Returns:
            Dictionary containing the loaded data
        """
        if self._loaded_data is None:
            if self._data_loader is None:
                self._data_loader = self.get_data_loader()
            self._loaded_data = self._data_loader.load_data()
        return self._loaded_data
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools available in this environment.
        
        Returns:
            List of tool instances
        """
        if self._tool_registry is None:
            self._tool_registry = self.get_tool_registry()
        return self._tool_registry.get_all_tools()
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        if self._tool_registry is None:
            self._tool_registry = self.get_tool_registry()
        return self._tool_registry.get_tool_by_name(tool_name)
    
    def get_shared_data(self) -> Dict[str, Any]:
        """Get shared data for tool execution.
        
        This method should return the data that tools need to access
        during execution (e.g., database state, user information, etc.).
        
        Returns:
            Dictionary containing shared data
        """
        return self.load_data()
    
    def reset_environment(self) -> None:
        """Reset the environment to its initial state."""
        self._loaded_data = None
        if self._tool_registry:
            self._tool_registry.reset_all_tools()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about this environment.
        
        Returns:
            Dictionary containing environment metadata
        """
        return {
            "environment_id": self.environment_id,
            "environment_name": self.environment_name,
            "environment_type": self.environment_type,
            "config": self.config,
        }
