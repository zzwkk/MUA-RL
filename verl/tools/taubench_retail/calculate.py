import logging
import os
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from ..base_tool import BaseTool
from ..schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CalculateTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate the result of a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
                        },
                    },
                    "required": ["expression"],
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
        """Initialize a new calculator instance"""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """Execute the calculation
        
        Args:
            instance_id: The unique identifier for this calculation instance
            parameters: Dictionary containing the 'expression' parameter
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing:
            - Response string (calculation result or error message)
            - Reward value (0.0 in this case)
            - Empty metrics dictionary
        """
        expression = parameters.get("expression", "")
        
        # Validate expression characters
        if not all(char in "0123456789+-*/(). " for char in expression):
            response = "Error: invalid characters in expression"
        else:
            try:
                # Evaluate the mathematical expression safely
                result = eval(expression, {"__builtins__": None}, {})
                response = str(round(float(result), 2))
            except Exception as e:
                response = f"Error: {str(e)}"

        # self._instance_dict[instance_id]["response"] = response

        tool_reward = 0.0
        metric = {} 
        
        return response, tool_reward, metric

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward for this tool execution (always 0.0 for calculator)"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this calculator instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id] 