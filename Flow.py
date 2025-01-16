from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
from Basin import BasinProperties, BasinType, DrainageBasin
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx

class FlowEdge:
    """Represents a directed flow connection between two basins"""
    def __init__(
        self,
        from_basin: str,
        to_basin: str,
        max_flow_rate: float,  # cubic meters per second
        length: float,  # meters
        slope: float  # ratio (m/m)
    ):
        self.from_basin = from_basin
        self.to_basin = to_basin
        self.max_flow_rate = max_flow_rate
        self.length = length
        self.slope = slope
        
    def calculate_actual_flow(self, available_volume: float, time_step: float) -> float:
        """Calculate actual flow volume based on constraints"""
        max_volume = self.max_flow_rate * time_step
        return min(available_volume, max_volume)

class FluxStorm:
    """Main class implementing the FluxStorm model"""
    def __init__(self):
        self.basins: Dict[str, DrainageBasin] = {}
        self.edges: List[FlowEdge] = []
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.transpose_adjacency: Dict[str, List[str]] = defaultdict(list)
        
    def add_basin(self, basin: DrainageBasin):
        """Add a drainage basin to the system"""
        self.basins[basin.basin_id] = basin
        
    def add_edge(self, edge: FlowEdge):
        """Add a flow edge to the system"""
        self.edges.append(edge)
        self.adjacency_list[edge.from_basin].append(edge.to_basin)
        self.transpose_adjacency[edge.to_basin].append(edge.from_basin)
        
    def _dfs_first_pass(
        self,
        node: str,
        visited: Set[str],
        finish_order: List[str]
    ):
        """First DFS pass for Kosaraju's algorithm"""
        visited.add(node)
        
        for neighbor in self.adjacency_list[node]:
            if neighbor not in visited:
                self._dfs_first_pass(neighbor, visited, finish_order)
                
        finish_order.append(node)
        
    def _dfs_second_pass(
        self,
        node: str,
        visited: Set[str],
        current_scc: Set[str]
    ):
        """Second DFS pass for Kosaraju's algorithm"""
        visited.add(node)
        current_scc.add(node)
        
        for neighbor in self.transpose_adjacency[node]:
            if neighbor not in visited:
                self._dfs_second_pass(neighbor, visited, current_scc)
                
    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Implementation of Kosaraju's algorithm"""
        visited = set()
        finish_order = []
        
        # First DFS pass
        for node in self.basins.keys():
            if node not in visited:
                self._dfs_first_pass(node, visited, finish_order)
                
        # Second DFS pass
        visited.clear()
        sccs = []
        
        # Process nodes in reverse finishing order
        for node in reversed(finish_order):
            if node not in visited:
                current_scc = set()
                self._dfs_second_pass(node, visited, current_scc)
                sccs.append(current_scc)
                
        return sccs
    
    def simulate_timestep(self, rainfall_input: Dict[str, float], time_step: float):
        """Simulate one time step of the system"""
        # First, add rainfall to each basin
        for basin_id, rainfall in rainfall_input.items():
            if basin_id in self.basins:
                self.basins[basin_id].update_volume(rainfall)
                
        # Then process flows along edges
        for edge in self.edges:
            from_basin = self.basins[edge.from_basin]
            to_basin = self.basins[edge.to_basin]
            
            # Calculate flow volume
            available = from_basin.properties.current_volume
            flow = edge.calculate_actual_flow(available, time_step)
            
            # Update basin volumes
            from_basin.update_volume(-flow)
            to_basin.update_volume(flow)
    
    def visualize_network(self, highlight_sccs: bool = False):
        """Visualize the drainage network using networkx"""
        G = nx.DiGraph()
        
        # Add nodes
        for basin_id, basin in self.basins.items():
            G.add_node(basin_id, **{
                'capacity': basin.properties.capacity,
                'current_volume': basin.properties.current_volume,
                'type': basin.properties.basin_type.value
            })
            
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.from_basin, edge.to_basin, **{
                'max_flow': edge.max_flow_rate,
                'length': edge.length,
                'slope': edge.slope
            })
            
        # Create layout
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(12, 8))
        
        if highlight_sccs:
            sccs = self.find_strongly_connected_components()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(sccs)))
            
            # Draw each SCC with a different color
            for scc, color in zip(sccs, colors):
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=list(scc),
                    node_color=[color],
                    node_size=500
                )
        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color='lightblue',
                node_size=500
            )
            
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos)
        
        plt.title("FluxStorm Drainage Network")
        plt.axis('off')
        plt.show()

# Example usage
def create_sample_network() -> FluxStorm:
    """Create a sample drainage network for testing"""
    network = FluxStorm()
    
    # Create some basins
    basins_data = [
        ("B1", BasinProperties(1000, 0, BasinType.RETENTION, 10, 100, 5)),
        ("B2", BasinProperties(800, 0, BasinType.DETENTION, 8, 80, 3)),
        ("B3", BasinProperties(1200, 0, BasinType.NATURAL, 12, 150, 8)),
        ("B4", BasinProperties(900, 0, BasinType.ENGINEERED, 9, 90, 4))
    ]
    
    for basin_id, props in basins_data:
        network.add_basin(DrainageBasin(basin_id, props))
    
    # Create some edges
    edges_data = [
        ("B1", "B2", 10, 100, 0.02),
        ("B2", "B3", 8, 150, 0.015),
        ("B3", "B4", 12, 200, 0.025),
        ("B4", "B2", 6, 180, 0.01)  # Creates a cycle
    ]
    
    for from_id, to_id, flow, length, slope in edges_data:
        network.add_edge(FlowEdge(from_id, to_id, flow, length, slope))
    
    return network

if __name__ == "__main__":
    # Create and visualize a sample network
    network = create_sample_network()
    
    # Find SCCs
    sccs = network.find_strongly_connected_components()
    print("Strongly Connected Components:", sccs)
    
    # Visualize the network with SCCs highlighted
    network.visualize_network(highlight_sccs=True)
    
    # Simulate some rainfall
    rainfall = {
        "B1": 100,  # cubic meters of rain
        "B2": 80,
        "B3": 120,
        "B4": 90
    }
    
    # Run simulation for 10 time steps
    for _ in range(10):
        network.simulate_timestep(rainfall, time_step=1.0)
        
    # Check basin states after simulation
    for basin_id, basin in network.basins.items():
        print(f"\nBasin {basin_id}:")
        print(f"Current volume: {basin.properties.current_volume:.2f} m³")
        print(f"Available capacity: {basin.calculate_available_capacity():.2f} m³")
        print(f"At risk: {basin.is_at_risk()}")