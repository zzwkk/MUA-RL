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

from .base.environment import BaseEnvironment


class EnvironmentRegistry:
    """Global registry for managing different environment types.
    
    This class provides a centralized way to register and retrieve
    different environment implementations.
    """
    
    def __init__(self):
        """Initialize the environment registry."""
        self._environments: Dict[str, Type[BaseEnvironment]] = {}
        self._environment_instances: Dict[str, BaseEnvironment] = {}
    
    def register_environment(self, environment_type: str, 
                           environment_class: Type[BaseEnvironment]) -> None:
        """Register an environment class.
        
        Args:
            environment_type: Type identifier for the environment
            environment_class: Environment class to register
        """
        self._environments[environment_type] = environment_class
    
    def create_environment(self, environment_type: str, 
                          config: Dict[str, Any], 
                          instance_id: Optional[str] = None) -> Optional[BaseEnvironment]:
        """Create an environment instance.
        
        Args:
            environment_type: Type of environment to create
            config: Configuration for the environment
            instance_id: Optional instance ID for the environment
            
        Returns:
            Environment instance if successful, None otherwise
        """
        if environment_type not in self._environments:
            return None
        
        if instance_id is None:
            instance_id = str(uuid4())
        
        try:
            environment_class = self._environments[environment_type]
            environment_instance = environment_class(config)
            self._environment_instances[instance_id] = environment_instance
            return environment_instance
        except Exception as e:
            print(f"Failed to create environment instance for {environment_type}: {e}")
            return None
    
    def get_environment(self, instance_id: str) -> Optional[BaseEnvironment]:
        """Get an environment instance by ID.
        
        Args:
            instance_id: ID of the environment instance
            
        Returns:
            Environment instance if found, None otherwise
        """
        return self._environment_instances.get(instance_id)
    
    def get_environment_by_type(self, environment_type: str) -> Optional[BaseEnvironment]:
        """Get the first environment instance of a specific type.
        
        Args:
            environment_type: Type of environment to retrieve
            
        Returns:
            Environment instance if found, None otherwise
        """
        for instance in self._environment_instances.values():
            if instance.environment_type == environment_type:
                return instance
        return None
    
    def get_all_environments(self) -> List[BaseEnvironment]:
        """Get all environment instances.
        
        Returns:
            List of all environment instances
        """
        return list(self._environment_instances.values())
    
    def get_registered_types(self) -> List[str]:
        """Get all registered environment types.
        
        Returns:
            List of registered environment type names
        """
        return list(self._environments.keys())
    
    def unregister_environment_type(self, environment_type: str) -> bool:
        """Unregister an environment type.
        
        Args:
            environment_type: Type of environment to unregister
            
        Returns:
            True if environment type was unregistered, False if not found
        """
        if environment_type in self._environments:
            del self._environments[environment_type]
            # Remove all instances of this type
            instances_to_remove = [
                instance_id for instance_id, instance in self._environment_instances.items()
                if instance.environment_type == environment_type
            ]
            for instance_id in instances_to_remove:
                del self._environment_instances[instance_id]
            return True
        return False
    
    def remove_environment_instance(self, instance_id: str) -> bool:
        """Remove a specific environment instance.
        
        Args:
            instance_id: ID of the environment instance to remove
            
        Returns:
            True if instance was removed, False if not found
        """
        if instance_id in self._environment_instances:
            del self._environment_instances[instance_id]
            return True
        return False
    
    def clear_all_instances(self) -> None:
        """Clear all environment instances."""
        self._environment_instances.clear()
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the environment registry.
        
        Returns:
            Dictionary containing registry metadata
        """
        return {
            "registered_types": list(self._environments.keys()),
            "active_instances": len(self._environment_instances),
            "instance_types": {
                instance_id: instance.environment_type 
                for instance_id, instance in self._environment_instances.items()
            }
        }


# Global registry instance
global_environment_registry = EnvironmentRegistry()
