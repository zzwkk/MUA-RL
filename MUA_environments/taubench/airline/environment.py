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
from .data_loader import TauBenchAirlineDataLoader


class TauBenchAirlineEnvironment(BaseEnvironment):
    """TauBench airline environment implementation."""
    
    @property
    def environment_name(self) -> str:
        """Return the name of this environment."""
        return "taubench_airline"
    
    @property
    def environment_type(self) -> str:
        """Return the type of this environment."""
        return "airline"
    
    def get_data_loader(self) -> TauBenchAirlineDataLoader:
        """Get the data loader for this environment."""
        return TauBenchAirlineDataLoader(self.config)
    
    def get_tool_registry(self) -> ToolRegistry:
        """Get the tool registry for this environment."""
        registry = ToolRegistry(self.config)
        
        # Register all airline tools
        self._register_airline_tools(registry)
        
        return registry
    
    def _register_airline_tools(self, registry: ToolRegistry) -> None:
        """Register all airline-specific tools.
        
        Args:
            registry: Tool registry to register tools with
        """
        # Import tool classes
        from verl.tools.taubench_airline.book_reservation import BookReservationTool
        from verl.tools.taubench_airline.calculate import CalculateTool
        from verl.tools.taubench_airline.cancel_reservation import CancelReservationTool
        from verl.tools.taubench_airline.get_reservation_details import GetReservationDetailsTool
        from verl.tools.taubench_airline.get_airline_user_details import GetAirlineUserDetailsTool
        from verl.tools.taubench_airline.list_all_airports import ListAllAirportsTool
        from verl.tools.taubench_airline.search_direct_flight import SearchDirectFlightTool
        from verl.tools.taubench_airline.search_onestop_flight import SearchOnestopFlightTool
        from verl.tools.taubench_airline.send_certificate import SendCertificateTool
        from verl.tools.taubench_airline.update_reservation_baggages import UpdateReservationBaggagesTool
        from verl.tools.taubench_airline.update_reservation_flights import UpdateReservationFlightsTool
        from verl.tools.taubench_airline.update_reservation_passengers import UpdateReservationPassengersTool
        from verl.tools.taubench_airline.think import ThinkTool
        from verl.tools.taubench_airline.transfer_to_human_agents import TransferToHumanAgentsTool
        
        # Register each tool with default config
        tools_to_register = [
            ("book_reservation", BookReservationTool, {}),
            ("calculate", CalculateTool, {}),
            ("cancel_reservation", CancelReservationTool, {}),
            ("get_reservation_details", GetReservationDetailsTool, {}),
            ("get_airline_user_details", GetAirlineUserDetailsTool, {}),
            ("list_all_airports", ListAllAirportsTool, {}),
            ("search_direct_flight", SearchDirectFlightTool, {}),
            ("search_onestop_flight", SearchOnestopFlightTool, {}),
            ("send_certificate", SendCertificateTool, {}),
            ("update_reservation_baggages", UpdateReservationBaggagesTool, {}),
            ("update_reservation_flights", UpdateReservationFlightsTool, {}),
            ("update_reservation_passengers", UpdateReservationPassengersTool, {}),
            ("think", ThinkTool, {}),
            ("transfer_to_human_agents", TransferToHumanAgentsTool, {}),
        ]
        
        for tool_name, tool_class, tool_config in tools_to_register:
            registry.register_tool_class(tool_name, tool_class, tool_config)
