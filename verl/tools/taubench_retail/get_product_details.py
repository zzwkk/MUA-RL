import logging
import os
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4
import json

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetProductDetailsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "get_product_details",
                "description": "Get the inventory details of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                        },
                    },
                    "required": ["product_id"],
                },
            },
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema for OpenAI function calling"""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Initialize a new instance with its own data copy"""
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Initialize instance data
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """Execute the product details lookup using shared data from the request"""
        product_id = parameters.get("product_id", "")
        
        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        products = shared_data["products"]

        # Look up product by ID
        if product_id in products:
            response = json.dumps(products[product_id])  # Convert to string to ensure serializable
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"found": True}

        response = "Error: product not found"
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"found": False}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this lookup instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]