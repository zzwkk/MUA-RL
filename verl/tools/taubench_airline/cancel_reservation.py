import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CancelReservationTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "cancel_reservation",
                "description": "Cancel the whole reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                    },
                    "required": ["reservation_id"],
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
        """Execute the reservation cancellation using shared data from the request"""
        reservation_id = parameters.get("reservation_id", "")
        
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        reservations = shared_data["reservations"]
        if reservation_id not in reservations:
            response = "Error: reservation not found"
            return response, 0.0, {}

        reservation = reservations[reservation_id]

        # reverse the payment
        refunds = []
        for payment in reservation["payment_history"]:
            refunds.append(
                {
                    "payment_id": payment["payment_id"],
                    "amount": -payment["amount"],
                }
            )
        reservation["payment_history"].extend(refunds)
        reservation["status"] = "cancelled"
        
        response = json.dumps(reservation)
        
        # Calculate reward based on successful cancellation
        tool_reward = 0.0
        metric = {
            "refund_count": len(refunds),
            "total_refund_amount": sum(abs(refund["amount"]) for refund in refunds)
        }
        
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this cancel reservation instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

