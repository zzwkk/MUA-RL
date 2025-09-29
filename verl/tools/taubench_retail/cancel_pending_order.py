import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CancelPendingOrderTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "description": (
                    "Cancel a pending order. If the order is already processed or delivered, "
                    "it cannot be cancelled. The agent needs to explain the cancellation detail "
                    "and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, "
                    "the order status will be changed to 'cancelled' and the payment will be refunded. "
                    "The refund will be added to the user's gift card balance immediately if the payment "
                    "was made using a gift card, otherwise the refund would take 5-7 business days to process. "
                    "The function returns the order details after the cancellation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["no longer needed", "ordered by mistake"],
                            "description": "The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.",
                        },
                    },
                    "required": ["order_id", "reason"],
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
        
        # Load fresh data for this instance
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """Execute the order cancellation using shared data from the request"""
        order_id = parameters.get("order_id", "")
        reason = parameters.get("reason", "")
        
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")

        orders = shared_data["orders"]
        if order_id not in orders:
            response = "Error: order not found"
            
            return response, 0.0, {}

        order = orders[order_id]
        if order["status"] != "pending":
            response = "Error: non-pending order cannot be cancelled"
            
            return response, 0.0, {}

        # Validate reason
        if reason not in ["no longer needed", "ordered by mistake"]:
            response = "Error: invalid reason"
            
            return response, 0.0, {}

        # Process refunds
        refunds = []
        for payment in order["payment_history"]:
            payment_id = payment["payment_method_id"]
            refund = {
                "transaction_type": "refund",
                "amount": payment["amount"],
                "payment_method_id": payment_id,
            }
            refunds.append(refund)
            
            # Immediate refund for gift cards
            if "gift_card" in payment_id:
                user = shared_data["users"][order["user_id"]]
                payment_method = user["payment_methods"][payment_id]
                payment_method["balance"] += payment["amount"]
                payment_method["balance"] = round(payment_method["balance"], 2)

        # Update order status
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)

        response = json.dumps(order)
        

        # Calculate reward based on successful cancellation
        tool_reward = 0.0
        metric = {
            "refund_count": len(refunds),
            "total_refund_amount": sum(refund["amount"] for refund in refunds)
        }
        
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this cancel order instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
