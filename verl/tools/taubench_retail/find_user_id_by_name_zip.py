import logging
import os
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FindUserIdByNameZipTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "find_user_id_by_name_zip",
                "description": (
                    "Find user id by first name, last name, and zip code. If the user is not found, the function "
                    "will return an error message. By default, find user id by email, and only call this function "
                    "if the user is not found by email or cannot remember email."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "The first name of the customer, such as 'John'.",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "The last name of the customer, such as 'Doe'.",
                        },
                        "zip": {
                            "type": "string",
                            "description": "The zip code of the customer, such as '12345'.",
                        },
                    },
                    "required": ["first_name", "last_name", "zip"],
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
        """Execute the user search by name and zip using shared data from the request"""
        first_name = parameters.get("first_name", "").lower()
        last_name = parameters.get("last_name", "").lower()
        zip_code = parameters.get("zip", "")
        
        # Get shared data from kwargs
        shared_data = kwargs.get("shared_data", None)
        if shared_data is None:
            raise ValueError("Shared data not provided in kwargs")
        
        users = shared_data["users"]

        # Search for user by name and zip
        for user_id, profile in users.items():
            if (
                profile["name"]["first_name"].lower() == first_name
                and profile["name"]["last_name"].lower() == last_name
                and profile["address"]["zip"] == zip_code
            ):
                # self._instance_dict[instance_id]["response"] = user_id
                return user_id, 0.0, {"found": True}

        response = "Error: user not found"
        # self._instance_dict[instance_id]["response"] = response
        return response, 0.0, {"found": False}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this search instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]