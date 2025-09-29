import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UpdateReservationBaggagesTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "update_reservation_baggages",
                "description": "Update the baggage information of a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "total_baggages": {
                            "type": "integer",
                            "description": "The updated total number of baggage items included in the reservation.",
                        },
                        "nonfree_baggages": {
                            "type": "integer",
                            "description": "The updated number of non-free baggage items included in the reservation.",
                        },
                        "payment_id": {
                            "type": "string",
                            "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.",
                        },
                    },
                    "required": [
                        "reservation_id",
                        "total_baggages",
                        "nonfree_baggages",
                        "payment_id",
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
        
        # Load fresh data for this instance
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """Execute the reservation baggage update using shared data from the request"""
        reservation_id = parameters.get("reservation_id", "")
        total_baggages = parameters.get("total_baggages", 0)
        nonfree_baggages = parameters.get("nonfree_baggages", 0)
        payment_id = parameters.get("payment_id", "")
        
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        users, reservations = shared_data["users"], shared_data["reservations"]
        if reservation_id not in reservations:
            response = "Error: reservation not found"
            return response, 0.0, {}
            
        reservation = reservations[reservation_id]

        total_price = 50 * max(0, nonfree_baggages - reservation["nonfree_baggages"])
        if payment_id not in users[reservation["user_id"]]["payment_methods"]:
            response = "Error: payment method not found"
            return response, 0.0, {}
            
        payment_method = users[reservation["user_id"]]["payment_methods"][payment_id]
        if payment_method["source"] == "certificate":
            response = "Error: certificate cannot be used to update reservation"
            return response, 0.0, {}
            
        elif (
            payment_method["source"] == "gift_card"
            and payment_method["amount"] < total_price
        ):
            response = "Error: gift card balance is not enough"
            return response, 0.0, {}

        reservation["total_baggages"] = total_baggages
        reservation["nonfree_baggages"] = nonfree_baggages
        if payment_method["source"] == "gift_card":
            payment_method["amount"] -= total_price

        if total_price != 0:
            reservation["payment_history"].append(
                {
                    "payment_id": payment_id,
                    "amount": total_price,
                }
            )

        response = json.dumps(reservation)
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this update reservation baggages instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
