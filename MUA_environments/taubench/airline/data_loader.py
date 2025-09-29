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

import json
import os
from typing import Any, Dict

from MUA_environments.base.data_loader import BaseDataLoader


class TauBenchAirlineDataLoader(BaseDataLoader):
    """Data loader for TauBench airline environment."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the airline data loader.
        
        Args:
            config: Configuration dictionary containing data paths
        """
        super().__init__(config)
        self.data_dir = config.get("taubench_airline_data_dir", "verl/tools/taubench_airline/data")
    
    def load_data(self) -> Dict[str, Any]:
        """Load airline environment data.
        
        Returns:
            Dictionary containing flights, reservations, and users data
        """
        flights_path = os.path.join(self.data_dir, "flights.json")
        reservations_path = os.path.join(self.data_dir, "reservations.json")
        users_path = os.path.join(self.data_dir, "users.json")
        
        with open(flights_path) as f:
            flight_data = json.load(f)
        with open(reservations_path) as f:
            reservation_data = json.load(f)
        with open(users_path) as f:
            user_data = json.load(f)
        
        return {
            "flights": flight_data,
            "reservations": reservation_data,
            "users": user_data,
        }
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get the schema of the airline data.
        
        Returns:
            Dictionary describing the expected data schema
        """
        return {
            "flights": list,
            "reservations": list,
            "users": list,
        }
