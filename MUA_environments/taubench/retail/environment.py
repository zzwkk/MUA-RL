# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

from MUA_environments.base.environment import BaseEnvironment
from MUA_environments.base.tool_registry import ToolRegistry
from .data_loader import TauBenchRetailDataLoader


class TauBenchRetailEnvironment(BaseEnvironment):
    """TauBench retail environment implementation."""
    
    @property
    def environment_name(self) -> str:
        """Return the name of this environment."""
        return "taubench_retail"
    
    @property
    def environment_type(self) -> str:
        """Return the type of this environment."""
        return "retail"
    
    def get_data_loader(self) -> TauBenchRetailDataLoader:
        """Get the data loader for this environment."""
        return TauBenchRetailDataLoader(self.config)
    
    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for this environment."""
        registry = ToolRegistry(self.config)
        
        # Register all retail tools
        self._register_retail_tools(registry)
        
        return registry
    
    def _register_retail_tools(self, registry: ToolRegistry) -> None:
        """Register all retail-specific tools.
        
        Args:
            registry: Tool registry to register tools with
        """
        # Import tool classes
        from verl.tools.taubench_retail.calculate import CalculateTool
        from verl.tools.taubench_retail.cancel_pending_order import CancelPendingOrderTool
        from verl.tools.taubench_retail.exchange_delivered_order_items import ExchangeDeliveredOrderItemsTool
        from verl.tools.taubench_retail.find_user_id_by_email import FindUserIdByEmailTool
        from verl.tools.taubench_retail.find_user_id_by_name_zip import FindUserIdByNameZipTool
        from verl.tools.taubench_retail.get_order_details import GetOrderDetailsTool
        from verl.tools.taubench_retail.get_product_details import GetProductDetailsTool
        from verl.tools.taubench_retail.get_user_details import GetUserDetailsTool
        from verl.tools.taubench_retail.list_all_product_types import ListAllProductTypesTool
        from verl.tools.taubench_retail.modify_pending_order_address import ModifyPendingOrderAddressTool
        from verl.tools.taubench_retail.modify_pending_order_items import ModifyPendingOrderItemsTool
        from verl.tools.taubench_retail.modify_pending_order_payment import ModifyPendingOrderPaymentTool
        from verl.tools.taubench_retail.modify_user_address import ModifyUserAddressTool
        from verl.tools.taubench_retail.return_delivered_order_items import ReturnDeliveredOrderItemsTool
        from verl.tools.taubench_retail.think import ThinkTool
        from verl.tools.taubench_retail.transfer_to_human_agents import TransferToHumanAgentsTool
        
        # Register each tool with default config
        tools_to_register = [
            ("calculate", CalculateTool, {}),
            ("cancel_pending_order", CancelPendingOrderTool, {}),
            ("exchange_delivered_order_items", ExchangeDeliveredOrderItemsTool, {}),
            ("find_user_id_by_email", FindUserIdByEmailTool, {}),
            ("find_user_id_by_name_zip", FindUserIdByNameZipTool, {}),
            ("get_order_details", GetOrderDetailsTool, {}),
            ("get_product_details", GetProductDetailsTool, {}),
            ("get_user_details", GetUserDetailsTool, {}),
            ("list_all_product_types", ListAllProductTypesTool, {}),
            ("modify_pending_order_address", ModifyPendingOrderAddressTool, {}),
            ("modify_pending_order_items", ModifyPendingOrderItemsTool, {}),
            ("modify_pending_order_payment", ModifyPendingOrderPaymentTool, {}),
            ("modify_user_address", ModifyUserAddressTool, {}),
            ("return_delivered_order_items", ReturnDeliveredOrderItemsTool, {}),
            ("think", ThinkTool, {}),
            ("transfer_to_human_agents", TransferToHumanAgentsTool, {}),
        ]
        
        for tool_name, tool_class, tool_config in tools_to_register:
            registry.register_tool_class(tool_name, tool_class, tool_config)
