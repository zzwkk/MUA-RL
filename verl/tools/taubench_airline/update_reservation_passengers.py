import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UpdateReservationPassengersTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "update_reservation_passengers",
                "description": "Update the passenger information of a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "passengers": {
                            "type": "array",
                            "description": "An array of objects containing details about each passenger.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "first_name": {
                                        "type": "string",
                                        "description": "The first name of the passenger, such as 'Noah'.",
                                    },
                                    "last_name": {
                                        "type": "string",
                                        "description": "The last name of the passenger, such as 'Brown'.",
                                    },
                                    "dob": {
                                        "type": "string",
                                        "description": "The date of birth of the passenger in the format 'YYYY-MM-DD', such as '1990-01-01'.",
                                    },
                                },
                                "required": ["first_name", "last_name", "dob"],
                            },
                        },
                    },
                    "required": ["reservation_id", "passengers"],
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
        """Execute the reservation passenger update using shared data from the request"""
        reservation_id = parameters.get("reservation_id", "")
        passengers = parameters.get("passengers", [])
        
        shared_data = kwargs.get("shared_data", None) 
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        reservations = shared_data["reservations"]
        if reservation_id not in reservations:
            response = "Error: reservation not found"
            return response, 0.0, {}
            
        reservation = reservations[reservation_id]
        if len(passengers) != len(reservation["passengers"]):
            response = "Error: number of passengers does not match"
            return response, 0.0, {}
            
        reservation["passengers"] = passengers
        
        response = json.dumps(reservation)
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this update reservation passengers instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
