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
from .factory import environment_factory


class EnvironmentManager:
    """
    Manager class for handling environment instances in rollout.
    """
    
    def __init__(self):
        """Initialize the environment manager."""
        self._environment_cache: Dict[str, BaseEnvironment] = {}
        # Default terminal tools and stop word
        self._default_terminal_tools = ["transfer_to_human_agents"]
        self._default_stop_word = "###STOP###"
    
    def get_new_environment_data(self, ability: str, 
                                 config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        # Determine environment type from ability string
        environment_type = self._extract_environment_type(ability)
        if not environment_type:
            return None
        
        # Create new environment instance (bypass cache)
        if config is None:
            config = {}
        
        environment = environment_factory.create_environment(environment_type, config)
        if environment:
            return environment.get_shared_data()
        return None
    
    def get_environment_tools(self, ability: str, 
                            config: Optional[Dict[str, Any]] = None) -> list:
        """Get environment tools for a given ability.
        
        Args:
            ability: Ability string that may contain environment type
            config: Optional configuration for the environment
            
        Returns:
            List of environment tools
        """
        environment = self.get_environment_for_ability(ability, config)
        if environment:
            return environment.get_tools()
        return []
    
    def get_cached_environment(self, ability: str, 
                                  config: Optional[Dict[str, Any]] = None) -> Optional[BaseEnvironment]:
        """Get environment instance based on ability string.
        
        Args:
            ability: Ability string that may contain environment type
            config: Optional configuration for the environment
            
        Returns:
            Environment instance if found, None otherwise
        """
        # Determine environment type from ability string
        environment_type = self._extract_environment_type(ability)
        if not environment_type:
            return None
        
        # Check cache first
        cache_key = f"{environment_type}_{hash(str(config) if config else '')}"
        if cache_key in self._environment_cache:
            return self._environment_cache[cache_key]
        
        # Create new environment instance
        if config is None:
            config = {}
        
        environment = environment_factory.create_environment(environment_type, config)
        if environment:
            self._environment_cache[cache_key] = environment
        
        return environment
    
    def _extract_environment_type(self, ability: str) -> Optional[str]:
        """Extract environment type from ability string.
        
        Args:
            ability: Ability string to analyze
            
        Returns:
            Environment type if found, None otherwise
        """
        ability_lower = ability.lower()
        
        if "retail" in ability_lower:
            return "retail"
        elif "airline" in ability_lower:
            return "airline"
        else:
            return None
    
    def get_cached_environment_data(self, ability: str, 
                           config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get environment data for a given ability.
        
        Args:
            ability: Ability string that may contain environment type
            config: Optional configuration for the environment
            
        Returns:
            Environment data if found, None otherwise
        """
        environment = self.get_environment_for_ability(ability, config)
        if environment:
            return environment.get_shared_data()
        return None
    
    def clear_cache(self) -> None:
        """Clear the environment cache."""
        self._environment_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the environment cache.
        
        Returns:
            Dictionary containing cache metadata
        """
        return {
            "cached_environments": len(self._environment_cache),
            "cache_keys": list(self._environment_cache.keys()),
        }
    
    def get_terminal_tools(self, config: Optional[Dict[str, Any]] = None) -> list:
        """Get terminal tools for conversation termination.
        
        Args:
            config: Optional configuration containing terminal_tools
            
        Returns:
            List of terminal tool names
        """
        if config and "terminal_tools" in config:
            return config["terminal_tools"]
        return self._default_terminal_tools.copy()
    
    def get_stop_word(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Get stop word for conversation termination.
        
        Args:
            config: Optional configuration containing stop_word
            
        Returns:
            Stop word string
        """
        if config and "stop_word" in config:
            return config["stop_word"]
        return self._default_stop_word
    
    def set_default_terminal_tools(self, terminal_tools: list) -> None:
        """Set default terminal tools.
        
        Args:
            terminal_tools: List of terminal tool names
        """
        self._default_terminal_tools = terminal_tools.copy()
    
    def set_default_stop_word(self, stop_word: str) -> None:
        """Set default stop word.
        
        Args:
            stop_word: Stop word string
        """
        self._default_stop_word = stop_word


# Global environment manager instance
environment_manager = EnvironmentManager()
