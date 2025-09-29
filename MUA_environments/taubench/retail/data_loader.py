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


class TauBenchRetailDataLoader(BaseDataLoader):
    """Data loader for TauBench retail environment."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the retail data loader.
        
        Args:
            config: Configuration dictionary containing data paths
        """
        super().__init__(config)
        self.data_dir = config.get("data_dir", "verl/tools/taubench_retail/data")
    
    def load_data(self) -> Dict[str, Any]:
        """Load retail environment data.
        
        Returns:
            Dictionary containing orders, products, and users data
        """
        orders_path = os.path.join(self.data_dir, "orders.json")
        products_path = os.path.join(self.data_dir, "products.json")
        users_path = os.path.join(self.data_dir, "users.json")
        
        with open(orders_path) as f:
            order_data = json.load(f)
        with open(products_path) as f:
            product_data = json.load(f)
        with open(users_path) as f:
            user_data = json.load(f)
        
        return {
            "orders": order_data,
            "products": product_data,
            "users": user_data,
        }
    
    def get_data_schema(self) -> Dict[str, Any]:
        """Get the schema of the retail data.
        
        Returns:
            Dictionary describing the expected data schema
        """
        return {
            "orders": list,
            "products": list,
            "users": list,
        }
