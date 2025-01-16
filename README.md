# FluxStorm: Urban Drainage Network Analysis System

## Overview
FluxStorm is an educational implementation of a stormwater drainage analysis system that uses graph theory, particularly Kosaraju's algorithm, to analyze and optimize urban drainage networks. This project serves as a learning resource for understanding:
- Graph theory algorithms in practical applications
- Urban drainage system modeling
- Network analysis techniques
- Python object-oriented programming

## Table of Contents
- [Key Features](#key-features)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Analysis Methods](#analysis-methods)

## Key Features
- **Graph-Based Modeling**: Represents drainage networks as directed graphs
- **Kosaraju's Algorithm**: Identifies strongly connected components (SCCs) in drainage networks
- **Flow Simulation**: Models water flow through interconnected drainage basins
- **Advanced Analysis**: Includes vulnerability scoring, bottleneck detection, and risk assessment
- **Visualization**: Network visualization using NetworkX and Matplotlib
- **Type Hints**: Comprehensive Python type annotations for learning purposes

### Dependencies
- Python 3.8+
- NetworkX
- NumPy
- Matplotlib
- SciPy

## Core Components

### 1. Basin Management
```python
from fluxstorm import DrainageBasin, BasinProperties, BasinType

# Create a drainage basin
basin = DrainageBasin(
    basin_id="B1",
    properties=BasinProperties(
        capacity=1000,
        current_volume=0,
        basin_type=BasinType.RETENTION,
        elevation=10,
        surface_area=100,
        infiltration_rate=5
    )
)
```

### 2. Network Construction
```python
from fluxstorm import FluxStorm, FlowEdge

# Initialize network
network = FluxStorm()

# Add basins and edges
network.add_basin(basin)
network.add_edge(FlowEdge(
    from_basin="B1",
    to_basin="B2",
    max_flow_rate=10,
    length=100,
    slope=0.02
))
```

### 3. Analysis System
```python
from fluxstorm import FluxStormAnalyzer

# Create analyzer
analyzer = FluxStormAnalyzer(network)

# Run comprehensive analysis
results = analyzer.run_complete_analysis()
```

## Usage Examples

### Basic Network Setup
```python
def create_sample_network():
    network = FluxStorm()
    
    # Add drainage basins
    basins_data = [
        ("B1", BasinProperties(1000, 0, BasinType.RETENTION, 10, 100, 5)),
        ("B2", BasinProperties(800, 0, BasinType.DETENTION, 8, 80, 3))
    ]
    
    for basin_id, props in basins_data:
        network.add_basin(DrainageBasin(basin_id, props))
    
    # Add flow connections
    network.add_edge(FlowEdge("B1", "B2", 10, 100, 0.02))
    
    return network
```

### Running Analysis
```python
# Create and analyze network
network = create_sample_network()
analyzer = FluxStormAnalyzer(network)

# Find critical basins
critical_basins = analyzer.identify_critical_basins(threshold=0.8)

# Calculate vulnerability scores
vulnerability_scores = analyzer.calculate_vulnerability_scores()

# Visualize results
analyzer.visualize_analysis(analyzer.run_complete_analysis())
```

## Analysis Methods

### 1. Kosaraju's Algorithm Implementation
The system uses Kosaraju's algorithm to find strongly connected components in the drainage network:

```python
def find_strongly_connected_components(self):
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
    
    for node in reversed(finish_order):
        if node not in visited:
            current_scc = set()
            self._dfs_second_pass(node, visited, current_scc)
            sccs.append(current_scc)
    
    return sccs
```

### 2. Network Analysis Features
- Critical Basin Identification
- Vulnerability Scoring
- Cycle Risk Ranking
- Flow Bottleneck Detection
- Cascade Risk Analysis