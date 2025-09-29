import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReturnDeliveredOrderItemsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "return_delivered_order_items",
                "description": (
                    "Return some items of a delivered order. The order status will be changed to 'return requested'. "
                    "The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. "
                    "The user will receive follow-up email for how and where to return the item."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": (
                                "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
                            ),
                        },
                        "item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
                            ),
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": (
                                "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. "
                                "These can be looked up from the user or order details."
                            ),
                        },
                    },
                    "required": ["order_id", "item_ids", "payment_method_id"],
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
        """Execute the return order items request using shared data"""
        order_id = parameters.get("order_id", "")
        item_ids = parameters.get("item_ids", [])
        payment_method_id = parameters.get("payment_method_id", "")

        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")

        orders = shared_data["orders"]

        # Check if the order exists and is delivered
        if order_id not in orders:
            response = "Error: order not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        order = orders[order_id]
        if order["status"] != "delivered":
            response = "Error: non-delivered order cannot be returned"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Check if the payment method exists and is either the original payment method or a gift card
        if payment_method_id not in shared_data["users"][order["user_id"]]["payment_methods"]:
            response = "Error: payment method not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        if (
            "gift_card" not in payment_method_id
            and payment_method_id != order["payment_history"][0]["payment_method_id"]
        ):
            response = "Error: payment method should be either the original payment method or a gift card"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Check if the items to be returned exist (there could be duplicate items in either list)
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                response = "Error: some item not found"
                # self._instance_dict[instance_id]["response"] = response
                return response, 0.0, {"success": False}

        # Update the order status
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id

        response = json.dumps(order)
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this return order instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
