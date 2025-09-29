import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ExchangeDeliveredOrderItemsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "exchange_delivered_order_items",
                "description": (
                    "Exchange items in a delivered order to new items of the same product type. "
                    "For a delivered order, return or exchange can be only done once by the agent. "
                    "The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "item_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.",
                        },
                        "new_item_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": (
                                "The item ids to be exchanged for, each such as '1008292230'. "
                                "There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product."
                            ),
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": (
                                "The payment method id to pay or receive refund for the item price difference, "
                                "such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
                            ),
                        },
                    },
                    "required": [
                        "order_id",
                        "item_ids",
                        "new_item_ids",
                        "payment_method_id",
                    ],
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
        """Execute the order exchange using shared data from the request"""
        order_id = parameters.get("order_id", "")
        item_ids = parameters.get("item_ids", [])
        new_item_ids = parameters.get("new_item_ids", [])
        payment_method_id = parameters.get("payment_method_id", "")
        
        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        products, orders, users = shared_data["products"], shared_data["orders"], shared_data["users"]

        # Check order exists and is delivered
        if order_id not in orders:
            response = "Error: order not found"
            
            return response, 0.0, {}
            
        order = orders[order_id]
        if order["status"] != "delivered":
            response = "Error: non-delivered order cannot be exchanged"
            
            return response, 0.0, {}

        # Check the items to be exchanged exist
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                response = f"Error: {item_id} not found"
                
                return response, 0.0, {}

        # Check new items exist and match old items and are available
        if len(item_ids) != len(new_item_ids):
            response = "Error: the number of items to be exchanged should match"
            
            return response, 0.0, {}

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            product_id = item["product_id"]
            if not (
                new_item_id in products[product_id]["variants"]
                and products[product_id]["variants"][new_item_id]["available"]
            ):
                response = f"Error: new item {new_item_id} not found or available"
                
                return response, 0.0, {}

            old_price = item["price"]
            new_price = products[product_id]["variants"][new_item_id]["price"]
            diff_price += new_price - old_price

        diff_price = round(diff_price, 2)

        # Check payment method exists and can cover the price difference if gift card
        if payment_method_id not in users[order["user_id"]]["payment_methods"]:
            response = "Error: payment method not found"
            
            return response, 0.0, {}

        payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
        if (
            payment_method["source"] == "gift_card"
            and payment_method["balance"] < diff_price
        ):
            response = "Error: insufficient gift card balance to pay for the price difference"
            
            return response, 0.0, {}

        # Modify the order
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff_price

        response = json.dumps(order)
        

        # Calculate metrics
        metrics = {
            "items_exchanged": len(item_ids),
            "price_difference": diff_price,
        }
        
        return response, 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this exchange instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
