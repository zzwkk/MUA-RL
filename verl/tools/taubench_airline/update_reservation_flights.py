import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4
from copy import deepcopy

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UpdateReservationFlightsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "update_reservation_flights",
                "description": "Update the flight information of a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "cabin": {
                            "type": "string",
                            "enum": [
                                "basic_economy",
                                "economy",
                                "business",
                            ],
                        },
                        "flights": {
                            "type": "array",
                            "description": "An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if the a flight segment is not changed, it should still be included in the array.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "flight_number": {
                                        "type": "string",
                                        "description": "Flight number, such as 'HAT001'.",
                                    },
                                    "date": {
                                        "type": "string",
                                        "description": "The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.",
                                    },
                                },
                                "required": ["flight_number", "date"],
                            },
                        },
                        "payment_id": {
                            "type": "string",
                            "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.",
                        },
                    },
                    "required": ["reservation_id", "cabin", "flights", "payment_id"],
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
        """Execute the reservation flight update using shared data from the request"""
        reservation_id = parameters.get("reservation_id", "")
        cabin = parameters.get("cabin", "")
        flights = parameters.get("flights", [])
        payment_id = parameters.get("payment_id", "")
        
        shared_data = kwargs.get("shared_data", None) 
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        users, reservations = shared_data["users"], shared_data["reservations"]
        if reservation_id not in reservations:
            response = "Error: reservation not found"
            return response, 0.0, {}
            
        reservation = reservations[reservation_id]

        # update flights and calculate price
        total_price = 0
        flights = deepcopy(flights)
        for flight in flights:
            # if existing flight, ignore
            if _ := [
                f
                for f in reservation["flights"]
                if f["flight_number"] == flight["flight_number"]
                and f["date"] == flight["date"]
                and cabin == reservation["cabin"]
            ]:
                total_price += _[0]["price"] * len(reservation["passengers"])
                flight["price"] = _[0]["price"]
                flight["origin"] = _[0]["origin"]
                flight["destination"] = _[0]["destination"]
                continue
            flight_number = flight["flight_number"]
            if flight_number not in shared_data["flights"]:
                response = f"Error: flight {flight_number} not found"
                return response, 0.0, {}
                
            flight_data = shared_data["flights"][flight_number]
            if flight["date"] not in flight_data["dates"]:
                response = f"Error: flight {flight_number} not found on date {flight['date']}"
                return response, 0.0, {}
                
            flight_date_data = flight_data["dates"][flight["date"]]
            if flight_date_data["status"] != "available":
                response = f"Error: flight {flight_number} not available on date {flight['date']}"
                return response, 0.0, {}
                
            if flight_date_data["available_seats"][cabin] < len(
                reservation["passengers"]
            ):
                response = f"Error: not enough seats on flight {flight_number}"
                return response, 0.0, {}
                
            flight["price"] = flight_date_data["prices"][cabin]
            flight["origin"] = flight_data["origin"]
            flight["destination"] = flight_data["destination"]
            total_price += flight["price"] * len(reservation["passengers"])

        total_price -= sum(flight["price"] for flight in reservation["flights"]) * len(
            reservation["passengers"]
        )

        # check payment
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

        # if checks pass, deduct payment and update seats
        if payment_method["source"] == "gift_card":
            payment_method["amount"] -= total_price
        reservation["flights"] = flights
        if total_price != 0:
            reservation["payment_history"].append(
                {
                    "payment_id": payment_id,
                    "amount": total_price,
                }
            )
        # do not make flight database update here, assume it takes time to be updated
        
        response = json.dumps(reservation)
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this update reservation flights instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]