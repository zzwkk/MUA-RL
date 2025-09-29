import logging
import os
import json
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ModifyUserAddressTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "modify_user_address",
                "description": "Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user id, such as 'sara_doe_496'.",
                        },
                        "address1": {
                            "type": "string",
                            "description": "The first line of the address, such as '123 Main St'.",
                        },
                        "address2": {
                            "type": "string",
                            "description": "The second line of the address, such as 'Apt 1' or ''.",
                        },
                        "city": {
                            "type": "string",
                            "description": "The city, such as 'San Francisco'.",
                        },
                        "state": {
                            "type": "string",
                            "description": "The state, such as 'CA'.",
                        },
                        "country": {
                            "type": "string",
                            "description": "The country, such as 'USA'.",
                        },
                        "zip": {
                            "type": "string",
                            "description": "The zip code, such as '12345'.",
                        },
                    },
                    "required": [
                        "user_id",
                        "address1",
                        "address2",
                        "city",
                        "state",
                        "country",
                        "zip",
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
        """Execute the address modification using shared data from the request"""
        user_id = parameters.get("user_id", "")
        address1 = parameters.get("address1", "")
        address2 = parameters.get("address2", "")
        city = parameters.get("city", "")
        state = parameters.get("state", "")
        country = parameters.get("country", "")
        zip_code = parameters.get("zip", "")
        
        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        users = shared_data["users"]

        # Check if user exists and modify address
        if user_id not in users:
            response = "Error: user not found"
            # self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {"success": False}

        # Update user's address
        users[user_id]["address"] = {
            "address1": address1,
            "address2": address2,
            "city": city,
            "state": state,
            "country": country,
            "zip": zip_code,
        }

        response = json.dumps(users[user_id])
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"success": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this modification instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
