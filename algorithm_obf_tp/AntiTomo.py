import networkx as nx
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import random
from generate_tree import generate_candidate_forest


class AntiTomoTopologyObfuscation:
    def __init__(self, original_topology, delays, lambda_simi=0.5, lambda_cost=0.5, delta_max=100):
        """
        Initialize the AntiTomo topology obfuscation process
        
        Parameters:
        - original_topology: NetworkX graph representing the original network
        - lambda_simi: Weight for similarity in objective function
        - lambda_cost: Weight for cost in objective function
        - delta_max: Maximum waiting time for attacker
        """
        self.original_topology = original_topology
        self.lambda_simi = lambda_simi
        self.lambda_cost = lambda_cost
        self.delta_max = delta_max
        self.path_delays = delays
        
        # Extract key topology characteristics
        self.num_nodes = original_topology.number_of_nodes()
        self.num_links = original_topology.number_of_edges()
        self.destinations = original_topology.graph.get('destinations', [])
    
    def calculate_topology_similarity(self, original_tree, obfuscated_tree):
        """
        Calculate topology similarity using normalized editing tree distance
        
        Implements equation (4) from the paper
        """
        def tree_editing_distance(t1, t2):
            # Simplified tree editing distance calculation
            # In practice, this would use the Zhang-Shasha algorithm
            return len(list(set(t1.edges) ^ set(t2.edges)))
        
        zero_tree = nx.empty_graph()
        ted_original_zero = tree_editing_distance(original_tree, zero_tree)
        ted_obfuscated_zero = tree_editing_distance(obfuscated_tree, zero_tree)
        ted_original_obfuscated = tree_editing_distance(original_tree, obfuscated_tree)
        
        similarity = 1 - (ted_original_obfuscated / 
                          (ted_original_zero + ted_obfuscated_zero))
        return similarity
        
    def solve_correlation_delays(self, candidate_tree):
        """
        Make the difference between candidate_tree's different paths' delays and the self.path_delays
        """
        # Create a new model
        model = gp.Model("CorrelationDelays")
        model.setParam('OutputFlag', 0)

        # Create a helper function to get the canonical edge representation
        def get_edge(u, v):
            # Always return edge with smaller node first
            return tuple(sorted([u, v]))

        # Decision Variables link_delay_vars from the candidate_tree
        # Store edges in canonical form (smaller node index first)
        canonical_edges = {get_edge(u, v) for u, v in candidate_tree.edges}
        links_delays_vars = model.addVars(canonical_edges, vtype=GRB.CONTINUOUS, name="link_delays", lb=1)
        
        redusials = []

        # Calculate the delay differences via topological information
        for i, j in self.path_delays.keys():
            # Find the shared path between i and j
            path_i = nx.shortest_path(candidate_tree, source=0, target=i)
            path_j = nx.shortest_path(candidate_tree, source=0, target=j)
            
            # Convert path edges to canonical form
            shared_path = {get_edge(u, v) for u, v in zip(path_i, path_i[1:])} & \
                        {get_edge(u, v) for u, v in zip(path_j, path_j[1:])}
            
            # Calculate the delay difference as the sum of shared link delays
            shared_delay = gp.quicksum(links_delays_vars[edge] for edge in shared_path)
            
            # Add minimum delay constraint
            # model.addConstr(shared_delay >= 1, name=f"min_delay_{i}_{j}")
            
            # Minimize the absolute difference between self.path_delays[(i, j)] and shared_delay
            residual = model.addVar(lb=0, name=f"Residual_{i}_{j}")
            model.addConstr(residual >= shared_delay - self.path_delays[(i, j)])
            model.addConstr(residual >= self.path_delays[(i, j)] - shared_delay)
            redusials.append(residual)

        # Objective Function: Minimize total deviation from observed shared delays
        model.setObjective(gp.quicksum(redusials), GRB.MINIMIZE)

        # Solve the model
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            correlation_path_delays = {}
            for i, j in self.path_delays.keys():
                path_i = nx.shortest_path(candidate_tree, source=0, target=i)
                path_j = nx.shortest_path(candidate_tree, source=0, target=j)
                # Convert path edges to canonical form
                shared_path = {get_edge(u, v) for u, v in zip(path_i, path_i[1:])} & \
                            {get_edge(u, v) for u, v in zip(path_j, path_j[1:])}
                shared_delay = sum(links_delays_vars[edge].X for edge in shared_path)
                correlation_path_delays[(i, j)] = shared_delay
            return correlation_path_delays
        else:
            print("No solution found")
            return None
                
    
    def calculate_cost(self, original_delays, obfuscated_delays):
        """
        Calculate the cost of obfuscation
        
        Parameters:
        - original_delays: dictionary with (i,j) keys and delay values
        - obfuscated_delays: dictionary with (i,j) keys and delay values
        
        Returns:
        - total cost as sum of relative differences
        """
        total_cost = 0
        for key in original_delays.keys():
            orig = original_delays[key]
            obf = obfuscated_delays[key]
            if orig > 0:  # Avoid division by zero
                total_cost += (obf / orig - 1)
        return total_cost

    def generate_obfuscated_topology(self, candidate_forest):
        """
        Generate obfuscated topology from candidate forest
        """
        best_object_value = float('inf')
        best_obfuscated_tree = None
        
        for candidate_tree in candidate_forest:
            # Solve correlation delays
            correlation_delays = self.solve_correlation_delays(candidate_tree)
            
            if correlation_delays is not None:
                # Calculate similarity
                similarity = self.calculate_topology_similarity(
                    self.original_topology, candidate_tree)
                
                # Calculate cost using the delay values directly
                cost = self.calculate_cost(
                    self.path_delays, correlation_delays)
                
                # Calculate objective function value
                object_value = (self.lambda_simi * similarity + 
                                self.lambda_cost * cost)
                
                # Update best tree if better
                if object_value < best_object_value:
                    best_object_value = object_value
                    best_obfuscated_tree = candidate_tree
        
        return best_obfuscated_tree, best_object_value




# Example usage
def main():
    # Create an example original topology
    G = nx.complete_graph(10)
    
    # Set destinations (nodes with degree 1)
    destinations = [node for node, degree in dict(G.degree()).items() if degree == 1]
    G.graph['destinations'] = destinations
    
    # true paths delays
    true_pair_delays = {
        (1, 2): 5,  # Observed shared delay between paths 1 and 2
        (1, 3): 5,
        (3, 4): 10,
        (1, 4): 5,
        (2, 3): 10,
        (2, 4): 10,
        (1, 5): 5,
        # Add more observed shared delays as needed
    }

    # Generate candidate forest (using previous implementation)
    candidate_forest = generate_candidate_forest(
        w=5, 
        n_lower=10, 
        n_upper=12, 
        num_destinations=len(destinations)
    )
    
    # Initialize AntiTomo obfuscation
    antitomo = AntiTomoTopologyObfuscation(G, true_pair_delays)
    
    # Generate obfuscated topology
    obfuscated_topology, object_value = antitomo.generate_obfuscated_topology(candidate_forest)
    
    # Print results
    print("Obfuscation Complete:")
    print(f"Objective Value: {object_value}")
    print(f"Original Topology Nodes: {G.number_of_nodes()}")
    print(f"Obfuscated Topology Nodes: {obfuscated_topology.number_of_nodes()}")
