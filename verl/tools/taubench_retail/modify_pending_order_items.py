import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ModifyPendingOrderItemsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "modify_pending_order_items",
                "description": (
                    "Modify items in a pending order to new items of the same product type. For a pending order, "
                    "this function can only be called once. The agent needs to explain the exchange detail and ask "
                    "for explicit user confirmation (yes/no) to proceed."
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
                            "description": "The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.",
                        },
                        "new_item_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.",
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
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
        """Execute the order item modification using shared data from the request"""
        order_id = parameters.get("order_id", "")
        item_ids = parameters.get("item_ids", [])
        new_item_ids = parameters.get("new_item_ids", [])
        payment_method_id = parameters.get("payment_method_id", "")

        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")

        products = shared_data["products"]
        orders = shared_data["orders"]
        users = shared_data["users"]

        # Check if the order exists and is pending
        if order_id not in orders:
            response = "Error: order not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        order = orders[order_id]
        if order["status"] != "pending":
            response = "Error: non-pending order cannot be modified"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Check if the items to be modified exist
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                response = f"Error: {item_id} not found"
                # self._instance_dict[instance_id]["response"] = response
                return response, 0.0, {"success": False}

        # Check new items exist, match old items, and are available
        if len(item_ids) != len(new_item_ids):
            response = "Error: the number of items to be exchanged should match"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            product_id = item["product_id"]
            if not (
                new_item_id in products[product_id]["variants"]
                and products[product_id]["variants"][new_item_id]["available"]
            ):
                response = f"Error: new item {new_item_id} not found or available"
                # self._instance_dict[instance_id]["response"] = response
                return response, 0.0, {"success": False}

            old_price = item["price"]
            new_price = products[product_id]["variants"][new_item_id]["price"]
            diff_price += new_price - old_price

        # Check if the payment method exists
        if payment_method_id not in users[order["user_id"]]["payment_methods"]:
            response = "Error: payment method not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # If the new item is more expensive, check if the gift card has enough balance
        payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
        if (
            payment_method["source"] == "gift_card"
            and payment_method["balance"] < diff_price
        ):
            response = "Error: insufficient gift card balance to pay for the new item"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Handle the payment or refund
        order["payment_history"].append(
            {
                "transaction_type": "payment" if diff_price > 0 else "refund",
                "amount": abs(diff_price),
                "payment_method_id": payment_method_id,
            }
        )
        if payment_method["source"] == "gift_card":
            payment_method["balance"] -= diff_price
            payment_method["balance"] = round(payment_method["balance"], 2)

        # Modify the order
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            item["item_id"] = new_item_id
            item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
            item["options"] = products[item["product_id"]]["variants"][new_item_id]["options"]
        order["status"] = "pending (item modified)"

        response = json.dumps(order)
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this modification instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
