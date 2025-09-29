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
from typing import Any, Dict


class BaseDataLoader(ABC):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data loader.
        
        Args:
            config: Configuration dictionary for the data loader
        """
        self.config = config
    
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """Load environment-specific data.
        
        Returns:
            Dictionary containing the loaded data
        """
        pass
    
    @abstractmethod
    def get_data_schema(self) -> Dict[str, Any]:
        """Get the schema of the data that will be loaded.
        
        Returns:
            Dictionary describing the data schema
        """
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the loaded data against the expected schema.
        
        Args:
            data: The data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        schema = self.get_data_schema()
        return self._validate_against_schema(data, schema)
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Internal method to validate data against schema.
        
        Args:
            data: The data to validate
            schema: The schema to validate against
            
        Returns:
            True if data matches schema, False otherwise
        """
        # Basic validation - can be extended for more complex schemas
        for key, expected_type in schema.items():
            if key not in data:
                return False
            if not isinstance(data[key], expected_type):
                return False
        return True
