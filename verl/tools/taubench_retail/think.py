import logging
import os
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ThinkTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "think",
                "description": (
                    "Use the tool to think about something. It will not obtain new information or change the database, "
                    "but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "A thought to think about.",
                        },
                    },
                    "required": ["thought"],
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
        """Execute the think action - simply records the thought and returns empty string"""
        thought = parameters.get("thought", "")
        
        # Record the thought but return empty string as per reference implementation
        response = ""
        # self._instance_dict[instance_id]["response"] = response
        
        # Log the thought for debugging/tracking purposes
        logger.debug(f"Thought recorded: {thought}")
        
        return response, 0.0, {"thought_recorded": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """No reward is given for thinking"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this think instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
