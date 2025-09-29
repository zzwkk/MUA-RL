import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class ListAllAirportsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "list_all_airports",
                "description": "List all airports and their cities.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
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
        """Execute the airport listing using shared data from the request"""
        airports = [
            "SFO",
            "JFK",
            "LAX",
            "ORD",
            "DFW",
            "DEN",
            "SEA",
            "ATL",
            "MIA",
            "BOS",
            "PHX",
            "IAH",
            "LAS",
            "MCO",
            "EWR",
            "CLT",
            "MSP",
            "DTW",
            "PHL",
            "LGA",
        ]
        cities = [
            "San Francisco",
            "New York",
            "Los Angeles",
            "Chicago",
            "Dallas",
            "Denver",
            "Seattle",
            "Atlanta",
            "Miami",
            "Boston",
            "Phoenix",
            "Houston",
            "Las Vegas",
            "Orlando",
            "Newark",
            "Charlotte",
            "Minneapolis",
            "Detroit",
            "Philadelphia",
            "LaGuardia",
        ]
        response = json.dumps({airport: city for airport, city in zip(airports, cities)})
        tool_reward = 0.0
        metric = {}
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this list airports instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
