import gurobipy as gp
from gurobipy import GRB
from algorithm_tmp import AlogrithmTmp


class MLE(AlogrithmTmp):
    def __init__(self, input: dict):
        self.name = "MLE"
        self.description = "MLE algorithm for network topology and delay estimation, the input is a dictionary with keys 'n' and 'Z'. 'n' is the number of links/nodes, and 'Z' is a dictionary with keys as pairs of links/nodes and values as observed shared delays between them."
        self.input = input
        self.output = {'link_delays': None, 'topology_matrix': None}
        self.n = self.input['n']
        self.Z = self.input['Z']     
        self.F = self.input['F']   

    def run(self):


        # Create a new model
        model = gp.Model("NetworkTopologyAndDelays")
        model.setParam('OutputFlag', 0)

        # Decision Variables
        # Link delays (continuous variables, non-negative)
        X = model.addVars(self.n, vtype=GRB.CONTINUOUS, name="X", lb=0)

        # Topology matrix Y (binary variables)
        Y = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="Y")

        # Deviation variables for absolute differences
        deviation = {}
        for (i, j) in self.Z.keys():
            deviation[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"dev_{i}_{j}", lb=0)

        # W variables for the product of Y[i,k] and Y[j,k]
        W = {}
        for (i, j) in self.Z.keys():
            i_idx = i - 1
            j_idx = j - 1
            for k in range(n):
                W[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"W_{i}_{j}_{k}")

        # Objective Function: Minimize total deviation from observed shared delays
        # model.setObjective(gp.quicksum(deviation.values()), GRB.MINIMIZE)

        # Constraints
        # Deviation constraints and shared delay calculation
        for (i, j), observed_delay in self.Z.items():
            i_idx = i - 1
            j_idx = j - 1

            # Constraints to define W[(i,j,k)] = Y[i_idx,k] * Y[j_idx,k]
            for k in range(n):
                model.addConstr(W[(i, j, k)] <= Y[i_idx, k], name=f"W_constr1_{i}_{j}_{k}")
                model.addConstr(W[(i, j, k)] <= Y[j_idx, k], name=f"W_constr2_{i}_{j}_{k}")
                model.addConstr(W[(i, j, k)] >= Y[i_idx, k] + Y[j_idx, k] - 1, name=f"W_constr3_{i}_{j}_{k}")

            # Shared delay calculation using W variables
            shared_delay = gp.quicksum(
                X[k] * W[(i, j, k)] for k in range(n)
            )

            # Deviation constraints (linearizing absolute value)
            model.addConstr(shared_delay - observed_delay <= deviation[(i, j)], name=f"dev_c1_{i}_{j}")
            model.addConstr(observed_delay - shared_delay <= deviation[(i, j)], name=f"dev_c2_{i}_{j}")

        # Topology constraints: Ensure that Y[i][i] = 1 for all i
        # Constraints
        # 1. Topology matrix constraints
        for i in range(n):
            model.addConstr(Y[i, i] == 1, name=f"Diagonal_{i}")  # Path i must include link i
            model.addConstr(Y[i, 0] == 1, name=f"Source_{i}") # Path i must include source

        for j in range(1, n):
            model.addConstr(Y[0, j] == 0, name=f"Sink_{j}") # Sink cannot be included in any path
        
        # make the Y  L1 norm as small as possible
        # model.setObjective(gp.quicksum(Y[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

        path_deviation = model.addVars(n, vtype=GRB.CONTINUOUS, name="path_dev", lb=0)



        # New Constraints: Path total delays
        for i, s_i in self.F.items():
            i_idx = i - 1

            # Total delay along path i
            path_delay = gp.quicksum(
                X[k] * Y[i_idx, k] for k in range(n)
            )

            # Constraint: Sum of delays along path i equals s_i
            # With these constraints:
            model.addConstr(path_delay - s_i <= path_deviation[i_idx])
            model.addConstr(s_i - path_delay <= path_deviation[i_idx])

        epsilon = 0.01
        for i in range(self.n):
            model.addConstr(X[i] >= epsilon)

        # For each pair of paths (rows in Y)
        # For each pair of paths (rows in Y)
        for i in range(self.n):
            for j in range(i + 1, self.n):  # compare with all subsequent rows
                # Add constraint that at least one element must be different between rows i and j
                # If rows are same, sum of XOR would be 0. If different, sum > 0
                model.addConstr(
                    gp.quicksum(Y[i, k] + Y[j, k] - 2*Y[i, k]*Y[j, k] for k in range(self.n)) >= 1,
                    name=f"diff_rows_{i}_{j}"
                )
                

        # Add a weight parameter to balance between shared delays and path delays
        alpha = 0.5  # adjust this weight as needed
        model.setObjective(
            alpha * gp.quicksum(deviation.values()) + 
            (1-alpha) * gp.quicksum(path_deviation.values()), 
            GRB.MINIMIZE
        )

        # Optimize the model
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Optimal solution found.")

            # Output the results
            print("\nLink Delays (X):")
            self.output['link_delays'] = [X[i].X for i in range(n)]

            print("\nTopology Matrix (Y):")
            self.output['topology_matrix'] = [[Y[i, j].X for j in range(n)] for i in range(n)]

            print("\nObjective Function Value:", model.ObjVal)

