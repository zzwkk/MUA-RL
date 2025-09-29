import logging
import os
from typing import Any, Dict, Optional, Tuple
import json
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class GetAirlineUserDetailsTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get the details of an user, including their reservations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user id, such as 'sara_doe_496'.",
                        },
                    },
                    "required": ["user_id"],
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
        """Execute the user details retrieval using shared data from the request"""
        user_id = parameters.get("user_id", "")

        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        users = shared_data["users"]
        if user_id in users:
            response = json.dumps(users[user_id])
            tool_reward = 0.0
            metric = {}
            return response, tool_reward, metric
        return "Error: user not found", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this get user details instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
