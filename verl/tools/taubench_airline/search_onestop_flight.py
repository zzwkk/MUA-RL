import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchOnestopFlightTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "search_onestop_flight",
                "description": "Search direct flights between two cities on a specific date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "The origin city airport in three letters, such as 'JFK'.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination city airport in three letters, such as 'LAX'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.",
                        },
                    },
                    "required": ["origin", "destination", "date"],
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
        """Execute the onestop flight search using shared data from the request"""
        origin = parameters.get("origin", "")
        destination = parameters.get("destination", "")
        date = parameters.get("date", "")
        
        shared_data = kwargs.get("shared_data", None)  
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        flights = shared_data["flights"]
        results = []
        for flight1 in flights.values():
            if flight1["origin"] == origin:
                for flight2 in flights.values():
                    if (
                        flight2["destination"] == destination
                        and flight1["destination"] == flight2["origin"]
                    ):
                        date2 = (
                            f"2024-05-{int(date[-2:])+1}"
                            if "+1" in flight1["scheduled_arrival_time_est"]
                            else date
                        )
                        if (
                            flight1["scheduled_arrival_time_est"]
                            > flight2["scheduled_departure_time_est"]
                        ):
                            continue
                        if date in flight1["dates"] and date2 in flight2["dates"]:
                            if (
                                flight1["dates"][date]["status"] == "available"
                                and flight2["dates"][date2]["status"] == "available"
                            ):
                                result1 = {
                                    k: v for k, v in flight1.items() if k != "dates"
                                }
                                result1.update(flight1["dates"][date])
                                result1["date"] = date
                                result2 = {
                                    k: v for k, v in flight2.items() if k != "dates"
                                }
                                result2.update(flight2["dates"][date])
                                result2["date"] = date2
                                results.append([result1, result2])
        
        response = json.dumps(results)
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this search onestop flight instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
