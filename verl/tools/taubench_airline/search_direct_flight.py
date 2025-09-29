import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchDirectFlightTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "search_direct_flight",
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
                            "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.",
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
        """Execute the direct flight search using shared data from the request"""
        origin = parameters.get("origin", "")
        destination = parameters.get("destination", "")
        date = parameters.get("date", "")
        
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        flights = shared_data["flights"]
        results = []
        for flight in flights.values():
            if flight["origin"] == origin and flight["destination"] == destination:
                if (
                    date in flight["dates"]
                    and flight["dates"][date]["status"] == "available"
                ):
                    # results add flight except dates, but add flight["datas"][date]
                    results.append({k: v for k, v in flight.items() if k != "dates"})
                    results[-1].update(flight["dates"][date])
        
        response = json.dumps(results)
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this search direct flight instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
