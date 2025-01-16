from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx

class BasinType(Enum):
    RETENTION = "retention"
    DETENTION = "detention"
    NATURAL = "natural"
    ENGINEERED = "engineered"

@dataclass
class BasinProperties:
    """Properties of a drainage basin"""
    capacity: float  # cubic meters
    current_volume: float  # cubic meters
    basin_type: BasinType
    elevation: float  # meters
    surface_area: float  # square meters
    infiltration_rate: float  # mm/hour
    
class DrainageBasin:
    """Represents a single drainage basin in the FluxStorm system"""
    def __init__(
        self,
        basin_id: str,
        properties: BasinProperties
    ):
        self.basin_id = basin_id
        self.properties = properties
        self.overflow_threshold = 0.9 * properties.capacity
        
    def is_at_risk(self) -> bool:
        """Check if basin is at risk of overflow"""
        return self.properties.current_volume >= self.overflow_threshold
    
    def calculate_available_capacity(self) -> float:
        """Calculate remaining capacity in cubic meters"""
        return self.properties.capacity - self.properties.current_volume
    
    def update_volume(self, volume_change: float):
        """Update current volume considering infiltration and capacity"""
        infiltration_loss = (
            self.properties.infiltration_rate *
            self.properties.surface_area *
            1/1000  # convert mm to meters
        )
        
        new_volume = (
            self.properties.current_volume +
            volume_change -
            infiltration_loss
        )
        
        self.properties.current_volume = max(0, min(
            new_volume,
            self.properties.capacity
        ))

