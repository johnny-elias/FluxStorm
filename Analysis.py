import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import linregress

from Flow import FluxStorm, create_sample_network

@dataclass
class NetworkAnalysis:
    """Container for network analysis results"""
    critical_basins: List[str]
    vulnerability_scores: Dict[str, float]
    cycle_rankings: List[Tuple[Set[str], float]]
    flow_bottlenecks: List[Tuple[str, str, float]]
    cascade_risks: Dict[str, float]

class FluxStormAnalyzer:
    """Advanced analysis methods for FluxStorm networks"""
    
    def __init__(self, network: 'FluxStorm'):
        self.network = network
        self.nx_graph = self._create_networkx_graph()

    def _create_networkx_graph(self) -> nx.DiGraph:
        """Convert FluxStorm network to NetworkX graph for analysis"""
        G = nx.DiGraph()
        
        # Add nodes with properties
        for basin_id, basin in self.network.basins.items():
            G.add_node(basin_id, **{
                'capacity': basin.properties.capacity,
                'elevation': basin.properties.elevation,
                'type': basin.properties.basin_type
            })
        
        # Add edges with properties
        for edge in self.network.edges:
            G.add_edge(edge.from_basin, edge.to_basin, **{
                'max_flow': edge.max_flow_rate,
                'length': edge.length,
                'slope': edge.slope
            })
            
        return G

    def identify_critical_basins(self, threshold: float = 0.8) -> List[str]:
        """
        Identify critical basins based on:
        - Centrality in the network
        - Capacity utilization
        - Number of dependent downstream basins
        """
        # Calculate various centrality measures
        betweenness = nx.betweenness_centrality(self.nx_graph)
        in_degree = dict(self.nx_graph.in_degree())
        out_degree = dict(self.nx_graph.out_degree())
        
        critical_basins = []
        for basin_id, basin in self.network.basins.items():
            # Calculate criticality score
            score = (
                betweenness[basin_id] * 0.4 +  # Network centrality
                (in_degree[basin_id] + out_degree[basin_id]) / 
                    (2 * len(self.network.basins)) * 0.3 +  # Connectivity
                (basin.properties.current_volume / 
                 basin.properties.capacity) * 0.3  # Capacity utilization
            )
            
            if score > threshold:
                critical_basins.append(basin_id)
                
        return sorted(critical_basins, 
                     key=lambda x: betweenness[x], 
                     reverse=True)

    def calculate_vulnerability_scores(self) -> Dict[str, float]:
        """
        Calculate vulnerability scores for each basin based on:
        - Elevation relative to neighbors
        - Available storage capacity
        - Incoming flow potential
        - Position in strongly connected components
        """
        scores = {}
        sccs = self.network.find_strongly_connected_components()
        
        # Find which SCC each basin belongs to
        basin_to_scc = {}
        for i, scc in enumerate(sccs):
            for basin in scc:
                basin_to_scc[basin] = i
        
        for basin_id, basin in self.network.basins.items():
            # Get incoming edges
            incoming = [edge for edge in self.network.edges 
                       if edge.to_basin == basin_id]
            
            # Calculate total potential incoming flow
            max_incoming_flow = sum(edge.max_flow_rate for edge in incoming)
            
            # Calculate elevation difference with neighbors
            elevation_diffs = []
            for edge in incoming:
                from_basin = self.network.basins[edge.from_basin]
                diff = from_basin.properties.elevation - basin.properties.elevation
                elevation_diffs.append(diff)
            
            avg_elevation_diff = (np.mean(elevation_diffs) 
                                if elevation_diffs else 0)
            
            # Calculate vulnerability score components
            capacity_factor = (1 - basin.calculate_available_capacity() / 
                             basin.properties.capacity)
            flow_factor = max_incoming_flow / basin.properties.capacity
            elevation_factor = max(0, avg_elevation_diff / 10)  # Normalize to 0-1
            scc_factor = 1 if basin_id in basin_to_scc else 0
            
            # Combine factors into final score
            scores[basin_id] = (
                0.3 * capacity_factor +
                0.3 * flow_factor +
                0.2 * elevation_factor +
                0.2 * scc_factor
            )
            
        return scores

    def rank_cycles_by_risk(self) -> List[Tuple[Set[str], float]]:
        """
        Rank strongly connected components (cycles) by risk level based on:
        - Total volume of water that can be trapped
        - Flow rates within the cycle
        - Elevation differences
        """
        sccs = self.network.find_strongly_connected_components()
        cycle_risks = []
        
        for scc in sccs:
            if len(scc) < 2:  # Skip single-basin SCCs
                continue
                
            # Calculate cycle properties
            total_capacity = sum(self.network.basins[b].properties.capacity 
                               for b in scc)
            
            # Find edges within this SCC
            internal_edges = [edge for edge in self.network.edges 
                            if edge.from_basin in scc and edge.to_basin in scc]
            
            # Calculate flow metrics
            avg_flow_rate = (np.mean([edge.max_flow_rate 
                                    for edge in internal_edges])
                           if internal_edges else 0)
            
            # Calculate elevation differences
            elevation_diffs = []
            for edge in internal_edges:
                from_basin = self.network.basins[edge.from_basin]
                to_basin = self.network.basins[edge.to_basin]
                diff = abs(from_basin.properties.elevation - 
                          to_basin.properties.elevation)
                elevation_diffs.append(diff)
            
            avg_elevation_diff = (np.mean(elevation_diffs) 
                                if elevation_diffs else 0)
            
            # Calculate risk score
            risk_score = (
                0.4 * (total_capacity / max(b.properties.capacity 
                       for b in self.network.basins.values())) +
                0.3 * (avg_flow_rate / max(edge.max_flow_rate 
                       for edge in self.network.edges)) +
                0.3 * (avg_elevation_diff / 10)  # Normalize elevation factor
            )
            
            cycle_risks.append((scc, risk_score))
            
        return sorted(cycle_risks, key=lambda x: x[1], reverse=True)

    def find_flow_bottlenecks(self) -> List[Tuple[str, str, float]]:
        """
        Identify potential bottlenecks in the network based on:
        - Flow capacity vs upstream contribution
        - Edge betweenness centrality
        - Downstream capacity constraints
        """
        bottlenecks = []
        edge_betweenness = nx.edge_betweenness_centrality(self.nx_graph)
        
        for edge in self.network.edges:
            # Calculate upstream contributing volume
            upstream_basins = nx.ancestors(self.nx_graph, edge.from_basin)
            upstream_basins.add(edge.from_basin)
            upstream_capacity = sum(self.network.basins[b].properties.capacity 
                                  for b in upstream_basins)
            
            # Calculate downstream available capacity
            downstream_basins = nx.descendants(self.nx_graph, edge.to_basin)
            downstream_basins.add(edge.to_basin)
            downstream_capacity = sum(
                self.network.basins[b].calculate_available_capacity() 
                for b in downstream_basins
            )
            
            # Calculate bottleneck score
            bottleneck_score = (
                0.4 * (upstream_capacity / (edge.max_flow_rate * 3600)) +  # hrs
                0.3 * edge_betweenness[(edge.from_basin, edge.to_basin)] +
                0.3 * (1 - downstream_capacity / upstream_capacity)
            )
            
            if bottleneck_score > 0.7:  # Threshold for significant bottlenecks
                bottlenecks.append((edge.from_basin, edge.to_basin, 
                                  bottleneck_score))
                
        return sorted(bottlenecks, key=lambda x: x[2], reverse=True)

    def analyze_cascade_risks(self) -> Dict[str, float]:
        """
        Analyze risk of cascade failures starting from each basin based on:
        - Network topology
        - Basin capacities
        - Flow rates
        """
        cascade_risks = {}
        
        for start_basin in self.network.basins:
            # Get all downstream basins in order
            downstream_paths = list(nx.dfs_edges(self.nx_graph, start_basin))
            if not downstream_paths:
                cascade_risks[start_basin] = 0
                continue
                
            # Calculate cumulative risk along each path
            path_risks = []
            for from_basin, to_basin in downstream_paths:
                edge = next(e for e in self.network.edges 
                          if e.from_basin == from_basin 
                          and e.to_basin == to_basin)
                
                # Calculate risk factors
                capacity_ratio = (
                    self.network.basins[to_basin].properties.current_volume /
                    self.network.basins[to_basin].properties.capacity
                )
                
                flow_ratio = edge.max_flow_rate / (
                    self.network.basins[to_basin].properties.capacity / 3600
                )
                
                elevation_diff = (
                    self.network.basins[from_basin].properties.elevation -
                    self.network.basins[to_basin].properties.elevation
                )
                
                # Combine into path segment risk
                segment_risk = (
                    0.4 * capacity_ratio +
                    0.3 * flow_ratio +
                    0.3 * (elevation_diff / 10)  # Normalize elevation factor
                )
                path_risks.append(segment_risk)
            
            # Calculate overall cascade risk for this starting basin
            cascade_risks[start_basin] = (
                np.mean(path_risks) * (1 + 0.1 * len(path_risks))  # Length penalty
            )
            
        return cascade_risks

    def run_complete_analysis(self) -> NetworkAnalysis:
        """Run all analysis methods and return comprehensive results"""
        return NetworkAnalysis(
            critical_basins=self.identify_critical_basins(),
            vulnerability_scores=self.calculate_vulnerability_scores(),
            cycle_rankings=self.rank_cycles_by_risk(),
            flow_bottlenecks=self.find_flow_bottlenecks(),
            cascade_risks=self.analyze_cascade_risks()
        )

    def visualize_analysis(self, analysis: NetworkAnalysis):
        """Create visualization of analysis results"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = plt.GridSpec(2, 2)
        
        # 1. Vulnerability Map
        ax1 = plt.subplot(gs[0, 0])
        pos = nx.spring_layout(self.nx_graph)
        
        # Draw nodes with vulnerability-based colors
        node_colors = [analysis.vulnerability_scores[node] 
                      for node in self.nx_graph.nodes()]
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                             node_color=node_colors, 
                             cmap=plt.cm.YlOrRd,
                             node_size=500)
        nx.draw_networkx_edges(self.nx_graph, pos)
        nx.draw_networkx_labels(self.nx_graph, pos)
        ax1.set_title("Vulnerability Map")
        
        # 2. Critical Basins Bar Chart
        ax2 = plt.subplot(gs[0, 1])
        critical_scores = [analysis.vulnerability_scores[basin] 
                         for basin in analysis.critical_basins]
        ax2.bar(analysis.critical_basins, critical_scores)
        ax2.set_title("Critical Basin Scores")
        plt.xticks(rotation=45)
        
        # 3. Cycle Risk Distribution
        ax3 = plt.subplot(gs[1, 0])
        cycle_sizes = [len(cycle) for cycle, _ in analysis.cycle_rankings]
        cycle_risks = [risk for _, risk in analysis.cycle_rankings]
        ax3.scatter(cycle_sizes, cycle_risks)
        ax3.set_xlabel("Cycle Size")
        ax3.set_ylabel("Risk Score")
        ax3.set_title("Cycle Risk Distribution")
        
        # 4. Cascade Risk Distribution
        ax4 = plt.subplot(gs[1, 1])
        basins = list(analysis.cascade_risks.keys())
        risks = list(analysis.cascade_risks.values())
        ax4.bar(basins, risks)
        ax4.set_title("Cascade Risk by Basin")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a sample network (using the previous create_sample_network function)
    network = create_sample_network()
    
    # Create analyzer
    analyzer = FluxStormAnalyzer(network)
    
    # Run complete analysis
    analysis_results = analyzer.run_complete_analysis()
    
    # Print results
    print("\nCritical Basins:", analysis_results.critical_basins)
    print("\nVulnerability Scores:", analysis_results.vulnerability_scores)
    print("\nCycle Rankings:", analysis_results.cycle_rankings)
    print("\nFlow Bottlenecks:", analysis_results.flow_bottlenecks)
    print("\nCascade Risks:", analysis_results.cascade_risks)
    
    # Visualize results
    analyzer.visualize_analysis(analysis_results)