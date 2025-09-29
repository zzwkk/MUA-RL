import logging
import os
import json
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ModifyPendingOrderPaymentTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "modify_pending_order_payment",
                "description": "Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "payment_method_id": {
                            "type": "string",
                            "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                        },
                    },
                    "required": [
                        "order_id",
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
        """Execute the payment method modification for a pending order"""
        order_id = parameters.get("order_id", "")
        payment_method_id = parameters.get("payment_method_id", "")
        
        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        orders = shared_data["orders"]

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

        # Check if the payment method exists
        if payment_method_id not in shared_data["users"][order["user_id"]]["payment_methods"]:
            response = "Error: payment method not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Check that the payment history should only have one payment
        if (
            len(order["payment_history"]) > 1
            or order["payment_history"][0]["transaction_type"] != "payment"
        ):
            response = "Error: there should be exactly one payment for a pending order"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Check that the payment method is different
        if order["payment_history"][0]["payment_method_id"] == payment_method_id:
            response = "Error: the new payment method should be different from the current one"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        amount = order["payment_history"][0]["amount"]
        payment_method = shared_data["users"][order["user_id"]]["payment_methods"][payment_method_id]

        # Check if the new payment method has enough balance if it is a gift card
        if (
            payment_method["source"] == "gift_card"
            and payment_method["balance"] < amount
        ):
            response = "Error: insufficient gift card balance to pay for the order"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Modify the payment method
        order["payment_history"].extend(
            [
                {
                    "transaction_type": "payment",
                    "amount": amount,
                    "payment_method_id": payment_method_id,
                },
                {
                    "transaction_type": "refund",
                    "amount": amount,
                    "payment_method_id": order["payment_history"][0]["payment_method_id"],
                },
            ]
        )

        # If payment is made by gift card, update the balance
        if payment_method["source"] == "gift_card":
            payment_method["balance"] -= amount
            payment_method["balance"] = round(payment_method["balance"], 2)

        # If refund is made to a gift card, update the balance
        if "gift_card" in order["payment_history"][0]["payment_method_id"]:
            old_payment_method = shared_data["users"][order["user_id"]]["payment_methods"][
                order["payment_history"][0]["payment_method_id"]
            ]
            old_payment_method["balance"] += amount
            old_payment_method["balance"] = round(old_payment_method["balance"], 2)

        response = json.dumps(order)
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]