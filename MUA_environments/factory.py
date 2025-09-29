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

from typing import Any, Dict, Optional

from .base.environment import BaseEnvironment
from .registry import global_environment_registry
from .taubench import TauBenchRetailEnvironment, TauBenchAirlineEnvironment


class EnvironmentFactory:
    """Factory class for creating environment instances.
    
    This class provides a convenient interface for creating and managing
    different types of environments.
    """
    
    def __init__(self):
        """Initialize the environment factory."""
        self._register_default_environments()
    
    def _register_default_environments(self) -> None:
        """Register default environment types."""
        global_environment_registry.register_environment("retail", TauBenchRetailEnvironment)
        global_environment_registry.register_environment("airline", TauBenchAirlineEnvironment)
    
    def create_environment(self, environment_type: str, 
                          config: Optional[Dict[str, Any]] = None,
                          instance_id: Optional[str] = None) -> Optional[BaseEnvironment]:
        """Create an environment instance.
        
        Args:
            environment_type: Type of environment to create
            config: Configuration for the environment
            instance_id: Optional instance ID for the environment
            
        Returns:
            Environment instance if successful, None otherwise
        """
        if config is None:
            config = {}
        
        return global_environment_registry.create_environment(
            environment_type, config, instance_id
        )
    
    def get_environment(self, instance_id: str) -> Optional[BaseEnvironment]:
        """Get an environment instance by ID.
        
        Args:
            instance_id: ID of the environment instance
            
        Returns:
            Environment instance if found, None otherwise
        """
        return global_environment_registry.get_environment(instance_id)
    
    def get_environment_by_type(self, environment_type: str) -> Optional[BaseEnvironment]:
        """Get the first environment instance of a specific type.
        
        Args:
            environment_type: Type of environment to retrieve
            
        Returns:
            Environment instance if found, None otherwise
        """
        return global_environment_registry.get_environment_by_type(environment_type)
    
    def list_available_types(self) -> list[str]:
        """List all available environment types.
        
        Returns:
            List of available environment type names
        """
        return global_environment_registry.get_registered_types()
    
    def list_active_instances(self) -> list[BaseEnvironment]:
        """List all active environment instances.
        
        Returns:
            List of active environment instances
        """
        return global_environment_registry.get_all_environments()
    
    def register_environment_type(self, environment_type: str, 
                                environment_class: type[BaseEnvironment]) -> None:
        """Register a new environment type.
        
        Args:
            environment_type: Type identifier for the environment
            environment_class: Environment class to register
        """
        global_environment_registry.register_environment(environment_type, environment_class)
    
    def remove_environment_instance(self, instance_id: str) -> bool:
        """Remove a specific environment instance.
        
        Args:
            instance_id: ID of the environment instance to remove
            
        Returns:
            True if instance was removed, False if not found
        """
        return global_environment_registry.remove_environment_instance(instance_id)
    
    def clear_all_instances(self) -> None:
        """Clear all environment instances."""
        global_environment_registry.clear_all_instances()


# Global factory instance
environment_factory = EnvironmentFactory()
