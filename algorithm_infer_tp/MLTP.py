import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

class TopologyIdentification:
    def __init__(self, num_receivers, penalty_lambda):
        """
        Initialize the topology identification parameters.

        :param num_receivers: Number of receivers in the network.
        :param penalty_lambda: Penalty parameter to control tree complexity.
        """
        self.num_receivers = num_receivers
        self.penalty_lambda = penalty_lambda
        self.trees = []
        self.log_likelihoods = {}

    def initialize_tree(self, delay_differences, sons):
        """
        Create an initial star topology and initialize edge delays based on observed delay differences.
        
        :param delay_differences: Observed delay differences (delta_ij).
        :return: Initialized tree with delays.
        """
        tree = nx.DiGraph()

        # 为 tree 添加儿子节点列表的属性
        tree.graph["sons"] = sons

        no_sons = [i for i in range(1, self.num_receivers + 1) if i not in sons]
        tree.graph["no_sons"] = no_sons
        tree.graph["virtual_nodes"] = []

        # Create a star topology
        for i in range(1, self.num_receivers + 1):
            if i in tree.graph["sons"]:
                tree.add_edge(0, i)

        # Initialize delays based on observed delay differences
        delays = {}
        for (i, j), delta in delay_differences.items():
            if i not in delays:
                delays[i] = []
            if j not in delays:
                delays[j] = []
            delays[i].append(delta)
            delays[j].append(delta)

        for i in range(1, self.num_receivers + 1):
            # Set the delay as the average delay difference involving receiver i
            if i in tree.graph["sons"]:
                tree[0][i]["delay"] = delay_differences[(0, i)]
        print("Initial Tree with Delays:", tree.edges(data=True))

        print("Initial Tree with Delays:", tree.edges(data=True))
        return tree


    def likelihood(self, tree, delay_differences, variances):
        """
        Compute the likelihood of a given tree.

        :param tree: A NetworkX graph representing the tree topology.
        :param delay_differences: Observed delay differences (mean).
        :param variances: Variances of the delay differences.
        :return: Log-likelihood value.
        """
        total_likelihood = 0
        draw_tree_and_calculate_likelihood(tree, delay_differences, variances, total_likelihood, step="1001")
        # print("---Tree with Delays----:", tree.edges(data=True))


        for (i, j), delay_diff in delay_differences.items():
            if i in tree.nodes and j in tree.nodes: # Check if i and j are in the tree

                path_i = nx.shortest_path(tree, source=0, target=i)
                path_j = nx.shortest_path(tree, source=0, target=j)
                # print("i and j = ", i, j , "path_i", path_i, "path_j", path_j)
                shared_path = []
                # Find the shared path between i and j
                links_in_path_i, links_in_path_j = [], []
                for i in range(len(path_i) - 1):
                    links_in_path_i.append((path_i[i], path_i[i + 1]))
                for j in range(len(path_j) - 1):
                    links_in_path_j.append((path_j[j], path_j[j + 1]))

                # print("links_in_path_i", links_in_path_i, "links_in_path_j", links_in_path_j)

                for link in links_in_path_i:
                    if link in links_in_path_j:
                        shared_path.append(link)
                
                # print("shared_path", shared_path)   

                # Theoretical delay difference (sum of link delays along the shared path)
                # if tree[u][v] don't have "delay" key, use the tree[v][u] instead
                # print("u, v", u, v)
                # gamma = sum(tree[u][v]["delay"] for u, v in zip(shared_path, list(shared_path)[1:]))
                gamma = 0

                # print("beginning!!!")
                for link in shared_path:
                    u, v = link
                    # print("i, j", i, j)
                    print("   u, v", u, v)
                    # if "delay" in tree[u][v]:
                    gamma += tree[u][v]["delay"]
                    # else:
                    #     gamma += tree[v][u]["delay"]

                # Log-likelihood for Gaussian distributed delays
                # print("over here!!!")
                variance = variances.get((i, j), 1)
                total_likelihood += -0.5 * ((delay_diff - gamma) ** 2 / variance + np.log(2 * np.pi * variance))

        return total_likelihood

    def penalized_likelihood(self, tree, delay_differences, variances):
        """
        Compute the penalized log-likelihood for a tree.

        :param tree: A NetworkX graph representing the tree topology.
        :param delay_differences: Observed delay differences (mean).
        :param variances: Variances of the delay differences.
        :return: Penalized log-likelihood value.
        """
        log_likelihood = self.likelihood(tree, delay_differences, variances)
        num_links = len(tree.edges)

        nodes_in_delays = delay_differences.keys()
        all_nodes = set([i for i, j in nodes_in_delays] + [j for i, j in nodes_in_delays])
        nodes_in_tree = set(tree.nodes)
        if all_nodes != nodes_in_tree:
            return -np.inf

        return log_likelihood - self.penalty_lambda * num_links

    def birth_step(self, tree):
        """Perform a birth step by introducing a new branching node."""
        new_tree = tree.copy()
        print("-------------------Birth Step------BEGIN------------------")
        # Find nodes with more than two children
        branching_nodes = [node for node in new_tree.nodes if len(list(new_tree.successors(node))) >= 2]
        print("Branching Nodes:", branching_nodes)
        if not branching_nodes:
            print("Birth step: These is no valid branching nodes to modify")
            return new_tree  # No valid branching nodes to modify

        # Select a random branching node and two of its children
        parent = random.choice(branching_nodes)
        children = list(new_tree.successors(parent))
        child1, child2 = random.sample(children, 2)

        print(f"Birth step: Parent = {parent}, Children = {children}, New Children = {child1}, {child2}")


        delay_1, delay_2 = new_tree[parent][child1]["delay"], new_tree[parent][child2]["delay"]
        # Create a new node and update the tree structure

        # 从 tree 的 no_sons 中选择一个节点作为新节点
        #  如果 no_sons 为空，则返回一个新节点
        if random.random() < 0.5:
            new_node = max(new_tree.nodes) + 1
            new_tree.graph["virtual_nodes"].append(new_node)
        else:
            new_node = random.choice(new_tree.graph["no_sons"])
            # 更新 no_sons 和 sons
            if new_node in new_tree.nodes:
                print("new_node in new_tree.nodes", new_node)
                print("-------------------Birth Step------END------------------")
                return new_tree


        new_tree.add_node(new_node)
        new_tree.add_edge(parent, new_node)
        new_tree.add_edge(new_node, child1)
        new_tree.add_edge(new_node, child2)

        # Remove old edges
        new_tree.remove_edge(parent, child1)
        new_tree.remove_edge(parent, child2)

        print("child_1, child_2", child1, child2, delay_1, delay_2)
        # Transformation for delays
        r = np.random.uniform(0, 1)
        mu_e_star = r * min(
            delay_1, delay_2
        )

        # Update delays for the new structure
        new_tree[parent][new_node]["delay"] = mu_e_star
        new_tree[new_node][child1]["delay"] = delay_1 - mu_e_star
        new_tree[new_node][child2]["delay"] = delay_2 - mu_e_star

        print(
            f"Birth step: Parent = {parent}, New Node = {new_node}, "
            f"New Delays = {mu_e_star}, {new_tree[new_node][child1]['delay']}, {new_tree[new_node][child2]['delay']}"
        )

        # print("New Tree with Delays:", new_tree.edges(data=True))
        print("-------------------Birth Step------END------------------")
        return new_tree


    def death_step(self, tree):
        """Remove a branching node and merge its children."""

        print("-------------------Death Step------BEGIN------------------")
        new_tree = tree.copy()
        branching_nodes = [node for node in new_tree.nodes if len(list(new_tree.successors(node))) == 2]

        if not branching_nodes:
            return new_tree  # No branching nodes to remove

        # Select a random branching node
        node_to_remove = random.choice(branching_nodes)
        if node_to_remove not in tree.graph["virtual_nodes"]:
            new_tree.graph["no_sons"].append(node_to_remove)

        parent_candidates = list(new_tree.predecessors(node_to_remove))

        if not parent_candidates:
            return new_tree  # The node has no parent (should not happen)

        parent = parent_candidates[0]
        child1, child2 = list(new_tree.successors(node_to_remove))


        # Combine delays
        mu_e_star = new_tree[parent][node_to_remove]["delay"]

        # update new tree
        new_tree.add_edge(parent, child1)
        new_tree.add_edge(parent, child2)

        new_tree[parent][child1]["delay"] = new_tree[node_to_remove][child1]["delay"] + mu_e_star
        new_tree[parent][child2]["delay"] = new_tree[node_to_remove][child2]["delay"] + mu_e_star

        # Update the tree structure
        
        new_tree.remove_edge(parent, node_to_remove)
        new_tree.remove_edge(node_to_remove, child1)
        new_tree.remove_edge(node_to_remove, child2)
        new_tree.remove_node(node_to_remove)


        print(f"Death step: Parent = {parent}, Node to Remove = {node_to_remove}, Children = {child1}, {child2}")
        print("New Tree with Delays:", new_tree.edges(data=True))

        print(f"Death step: Removed Node = {node_to_remove}, Combined Delays = {mu_e_star}")
        print("-------------------Death-Step------END------------------")
        return new_tree



    def mcmc_search(self, initial_tree, delay_differences, variances, iterations=5):
        """
        Perform MCMC search to find the tree with maximum penalized likelihood.

        :param initial_tree: Starting tree for the search.
        :param delay_differences: Observed delay differences.
        :param variances: Variances of the delay differences.
        :param iterations: Number of MCMC iterations.
        :return: Tree with maximum penalized likelihood.
        """
        current_tree = initial_tree
        best_tree = initial_tree
        best_score = self.penalized_likelihood(initial_tree, delay_differences, variances)

        print("Initial Score:", best_score)


        # Visualize the initial tree
        draw_tree_and_calculate_likelihood(initial_tree, delay_differences, variances, best_score, step=0)

        for step in range(1, iterations + 1):
            new_tree = None
            if random.random() < 0.5:
                new_tree = self.birth_step(current_tree)
            else:
                new_tree = self.death_step(current_tree)

            current_score = self.penalized_likelihood(current_tree, delay_differences, variances)
            new_score = self.penalized_likelihood(new_tree, delay_differences, variances)

            draw_tree_and_calculate_likelihood(new_tree, delay_differences, variances, new_score, step=step)

            # Decide whether to accept the new tree
            acceptance_ratio = np.exp(new_score - current_score)
            if random.random() < acceptance_ratio:
                current_tree = new_tree
                print("-------------------&&&&&&&&&&&&&&------------------------")
                print(f"Step {step}: Accepted new tree with score {new_score}")
                print("-------------------&&&&&&&&&&&&&&------------------------")

            # Update the best tree if the new score is better
            if new_score > best_score:
                best_tree = new_tree
                best_score = new_score

            # print(f"Step {step}: Current Score = {current_score}, New Score = {new_score}, Best Score = {best_score}")

        return best_tree


def draw_tree_and_calculate_likelihood(tree, delay_differences, variances, log_likelihood, step=None):
    """
    Draw the tree and calculate its log-likelihood based on delay differences.

    :param tree: NetworkX DiGraph representing the tree topology.
    :param delay_differences: Dictionary of observed delay differences (delta_ij).
    :param variances: Dictionary of variances for delay differences (sigma_ij^2).
    :param step: Optional step number for saving the visualization.
    :return: Log-likelihood value for the given tree.
    """


    # Ensure the directory exists
    output_dir = "./pic"
    os.makedirs(output_dir, exist_ok=True)
    # 1. Draw the tree
    plt.figure(figsize=(8, 6))
    pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
    nx.draw(tree, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrowsize=20, arrowstyle="->", width=2)
    plt.title(f"Inferred Topology, step-{step}", fontsize=14)


    # # 在图上加入log_likelihood
    plt.text(0, 0, f"Log-Likelihood: {log_likelihood:.2f}", fontsize=12, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.5))

    # 加入边的delay标签
    for u, v, data in tree.edges(data=True):
        plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, f"{data['delay']:.2f}", fontsize=10, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.5))
    

    # Save the figure
    if step is not None:
        plt.savefig(f"./pic/step_{step}.png", bbox_inches="tight")
        plt.clf()
        plt.close('all')  #避免内存泄漏
        
    
   #  return log_likelihood


def calculate_accuracy(ground_truth, inferred_tree):
    """
    Calculate precision, recall, and F1-score for inferred topology.
    
    :param ground_truth: NetworkX graph representing the ground truth topology.
    :param inferred_tree: NetworkX graph representing the inferred topology.
    :return: A dictionary with precision, recall, and F1-score.
    """
    # Extract edge sets
    ground_truth_edges = set(ground_truth.edges)
    inferred_edges = set(inferred_tree.edges)

    # Calculate true positives, false positives, false negatives
    true_positives = len(ground_truth_edges & inferred_edges)
    false_positives = len(inferred_edges - ground_truth_edges)
    false_negatives = len(ground_truth_edges - inferred_edges)

    # Precision, recall, F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def generate_branching_synthetic_data(num_receivers, mean_link_delay=10, link_delay_std=2, noise_std=0.5):
    """
    Generate synthetic data for branching topology with realistic delay differences.

    :param num_receivers: Number of receivers in the topology.
    :param mean_link_delay: Average delay for each link.
    :param link_delay_std: Standard deviation for link delays.
    :param noise_std: Standard deviation for added noise.
    :return: ground_truth (tree), delay_differences (dict), variances (dict)
    """
    import itertools

    # Generate a random binary tree as ground truth
    ground_truth = nx.DiGraph()
    nodes = list(range(num_receivers + 1))  # Include the root (node 0)
    parents = [0]  # Root node
    for i in range(1, len(nodes)):
        parent = random.choice(parents)
        ground_truth.add_edge(parent, i)
        parents.append(i)  # Add new node as a potential parent

    # Assign random delays to each link
    for u, v in ground_truth.edges:
        ground_truth[u][v]["delay"] = max(1, np.random.normal(mean_link_delay, link_delay_std))

    # sons nodes
    son_nodes = []
    # find the nodes have no children
    for node in ground_truth.nodes:
        if len(list(ground_truth.successors(node))) == 0:
            son_nodes.append(node)
    

    print("Ground Truth Edges delay:", ground_truth.edges(data=True))   
    # Compute delay differences based on shared path delays
    delay_differences = {}
    variances = {}
    for i, j in itertools.combinations(range(1, num_receivers + 1), 2):
        # Find the shared path between i and j
        path_i = nx.shortest_path(ground_truth, source=0, target=i)
        path_j = nx.shortest_path(ground_truth, source=0, target=j)
        shared_path = set(zip(path_i, path_i[1:])) & set(zip(path_j, path_j[1:]))
        
        # Calculate the delay difference as the sum of shared link delays
        shared_delay = sum(ground_truth[u][v]["delay"] for u, v in shared_path)
        # make shared_delay > 0
        shared_delay = max(1, shared_delay)
        
        # Add noise to simulate measurement error
        noisy_delay = shared_delay + np.random.normal(0, noise_std)
        delay_differences[(i, j)] = abs(noisy_delay)
        variances[(i, j)] = noise_std**2
    
    # add 0-x delay difference
    for i in range(1, num_receivers + 1):
        path_i = nx.shortest_path(ground_truth, source=0, target=i)
        shared_delay = sum(ground_truth[u][v]["delay"] for u, v in zip(path_i, path_i[1:]))
        shared_delay = max(1, shared_delay)
        noisy_delay = shared_delay + np.random.normal(0, noise_std)
        delay_differences[(0, i)] = abs(noisy_delay)
        variances[(0, i)] = noise_std**2



    return ground_truth, delay_differences, variances, son_nodes

