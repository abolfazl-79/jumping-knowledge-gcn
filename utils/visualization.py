
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.data import load_data, split_nodes, create_graph , label_dict
import collections
import seaborn as sns
from typing import Optional, Tuple, Dict, Union




def plot_node_degree_distribution(Gnx: nx.Graph, save_path: str = None, log_scale: bool = True,
                      figsize: tuple = (10, 6)) -> pd.DataFrame:

    """
    Plot the node degree distributions of a graph and return descriptive statistics.


    Args:
        Gnx (nx.Graph): NetworkX graph to analyze.
        save_path (str, optional): Path to save the plot. If None, display the plot.
        log_scale (bool): Use logarithmic scale for y-axis (default: True).
        figsize (tuple): Figure size as (width, height) in inches (default: (10, 6)).

    Returns:
        pd.DataFrame: Descriptive statistics of node degrees.
    Raises:
        ValueError: If the graph is empty or invalid.
    """

    if not Gnx or len(Gnx.nodes) == 0:
        raise ValueError("Input graph is empty or invalid")
    
    # Extract node degrees
    degrees = [val for (node, val) in Gnx.degree]

    # Compute descriptive statisitcs
    degree_stats = pd.DataFrame(pd.Series(degrees).describe()).transpose().round(2)
    print("Nide Degree statistics:\n", degree_stats.to_string(index=False))

    #Plot histogram
    plt.figure(figsize=(10,6))
    plt.hist(degrees, bins=50)
    plt.title("Node Degree Distribution")
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency (Log Scale)" if log_scale else "Frequency")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return degree_stats



def visualize_degree_centrality(Gnx: nx.Graph, save_path: Optional[str] = None, 
                                threshold_rank: int = 10, node_size_scale: float = 1000,
                                figsize: Tuple[int, int] = (12, 12),
                                seed: int = 42) -> dict:
    """
    Visualize a graph with nodes sized and colored by degree centrality, highlighting top central nodes.

    Args:
        graph (nx.Graph): NetworkX graph to visualize.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
        threshold_rank (int): Rank of centrality score to use as threshold (default: 10).
        node_size_scale (float): Scaling factor for node sizes (default: 1000).
        figsize (tuple): Figure size as (width, height) in inches (default: (12, 12)).
        seed (int): Random seed for layout reproducibility (default: 42).

    Returns:
        dict: Degree centrality scores for each node.
    Raises:
        ValueError: If the graph is empty or threshold_rank is invalid.
    """

    if not Gnx or len(Gnx.nodes) == 0:
        raise ValueError("Input graph is empty or invalid")
    if threshold_rank < 1 or threshold_rank >= len(Gnx.nodes):
        raise ValueError(f"threshold_rank must be between 1 and {len(Gnx.nodes)-1}")
    

    # Compute degree centrality
    centrality = nx.degree_centrality(Gnx)
    cent_array = np.array(list(centrality.values()))

    # Determine threshold for highlighting top nodes
    threshold = sorted(cent_array, reverse=True)[threshold_rank]

    # Binary coloring: 1 for top nodes, 0.1 for others (affects alpha)
    cent_bin = np.where(cent_array >= threshold, 1, 0.1)

    # Node sizes scaled by centrality
    node_size = list(map(lambda x: x * 1000, centrality.values()))

    # Compute layout
    pos = nx.spring_layout(Gnx, seed=seed)
    
    # Plot
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        Gnx, pos, node_size=node_size, cmap=plt.cm.plasma, 
        node_color=cent_bin, nodelist=list(centrality.keys()), alpha=cent_bin
    )
    nx.draw_networkx_edges(Gnx, pos, width=0.25, alpha=0.3)
    plt.title("Graph Visualization by Degree Centrality")
    plt.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return centrality



def visualize_graph_by_labels(graph: nx.Graph, labels: Union[pd.Series, np.ndarray, list], 
                             label_dict: Dict[int, str], save_path: Optional[str] = None, 
                             node_size: int = 50, edge_width: float = 0.25, 
                             figsize: Tuple[int, int] = (20, 15), seed: int = 42) -> None:
    
    """
    Visualize a graph with nodes colored by their labels.

    Args:
        graph (nx.Graph): NetworkX graph to visualize.
        labels (pd.Series, np.ndarray, or list): Node labels (indices corresponding to label_dict keys).
        label_dict (dict): Mapping of label indices to label names (e.g., {0: "Theory", ...}).
        save_path (str, optional): Path to save the plot. If None, displays the plot.
        node_size (int): Size of nodes in the plot (default: 50).
        edge_width (float): Width of edges (default: 0.25).
        figsize (tuple): Figure size as (width, height) in inches (default: (20, 15)).
        seed (int): Random seed for layout reproducibility (default: 42).

    Raises:
        ValueError: If inputs are invalid or incompatible.
    """

    if not graph or len(graph.nodes) == 0:
        raise ValueError("Input graph is empty or invalid")
    
    # print(labels)
    if len(labels) != len(graph.nodes):
        raise ValueError(f"Labels length ({len(labels)}) must match number of nodes ({len(graph.nodes)})")
    if not label_dict:
        raise ValueError("label_dict cannot be empty")
    
    # Convert labels to list if Series or array
    labels = labels.values if isinstance(labels, pd.Series) else list(labels)

    # Validate labels against label_dict
    unique_labels = set(labels)
    if not unique_labels.issubset(label_dict.values()):
        raise ValueError(f"Labels {unique_labels} must be keys in label_dict {set(label_dict.values())}")
    
    # Initialize an empty list to store the color for each node
    node_color = []
    # Create a list of 7 empty lists, one for each label class (Cora has 7 classes)
    nodelist = [[], [], [], [], [], [], []]
    # Define colors for each label class using hex codes
    colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
    # Iterate over node indices and their corresponding label indices
    for i, c in enumerate(labels):
        # Append the color corresponding to the node's label index
        node_color.append(colorlist[c])
        # Add the node index to the list for its label class
        nodelist[c].append(i)
    # Compute node positions using spring layout with a fixed seed for reproducibility
    pos = nx.spring_layout(Gnx, seed=42)
    plt.figure(figsize)
    label_list = list(label_dict.keys())
    # Iterate over node groups and their corresponding label indices
    for i, group in enumerate(zip(nodelist, label_list)):
        # Extract the nodes and label index for the current group
        nodes, label = group[0], group[1]
        # Draw nodes for this label with specified color, size, and legend label
        nx.draw_networkx_nodes(Gnx, pos, nodelist=nodes, node_size=5, node_color=colorlist[i], label=label)
    nx.draw_networkx_edges(Gnx, pos, width=0.25)
    plt.show()
    
    

def class_distribution(labels: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of classes in the dataset.

    Args:
        labels (pd.DataFrame): Series of label indices (0 to 6 for Cora).
        save_path (str, optional): Path to save the plot. If None, displays the plot.

    Raises:
        ValueError: If labels are empty or invalid.
    """
    if labels.empty:
        raise ValueError("Labels cannot be empty")
    
    # Count occurrences of each label
    counter = collections.Counter(labels)
    counter = dict(counter)

    # Extract counts in sorted order
    count = [x[1] for x in sorted(counter.items())]

    # Ensure 7 classes (Cora-specific)
    if len(count) != 7:
        raise ValueError(f"Expected 7 classes, found {len(count)}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(7), count)
    plt.xlabel("class", size=20)
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

        
def class_connection_matrix(Gnx: nx.Graph, node_data: pd.DataFrame, label_dict: Dict[str, int], 
                            save_path: Optional[str] = None) -> None:
    """
    Plot a heatmap of connections between classes.

    Args:
        Gnx (nx.Graph): NetworkX graph.
        node_data (pd.DataFrame): DataFrame with 'subject' column for node labels.
        label_dict (dict): Mapping of label names to indices (e.g., {"Theory": 0}).
        save_path (str, optional): Path to save the plot. If None, displays the plot.

    Raises:
        ValueError: If graph or node_data is empty, or label_dict is invalid.
    """
    if not Gnx or len(Gnx.nodes) == 0:
        raise ValueError("Input graph is empty or invalid")
    if node_data.empty:
        raise ValueError("node_data cannot be empty")
    if not label_dict:
        raise ValueError("label_dict cannot be empty")
    
    # Initialize 7x7 matrix for Cora's 7 classes
    label_connection_counts = np.zeros(shape=(7,7))

    # Count connections between classes
    for u, v, d in Gnx.edges(data=True):
        u_label = node_data.loc[u, 'subject']
        v_label = node_data.loc[v, 'subject']
        label_connection_counts[u_label][v_label] += 1
        label_connection_counts[v_label][u_label] += 1

    # Normalize by row sums
    s_r = np.sum(label_connection_counts, axis=1)
    scaled_matrix = np.zeros(shape=(7,7))
    for i in range(7):
        scaled_row = label_connection_counts[i, :] / s_r[i] if s_r[i] > 0 else 0
        scaled_matrix[i] = scaled_row

    # Plot heatmap
    plt.figure(figsize=(9, 7))
    with plt.rc_context({"font.size": 13}):  # Localize font size change
        hm = sns.heatmap(scaled_matrix, annot=True, cmap='hot_r', cbar=True, square=True)
        plt.xlabel("Class", size=20)
        plt.ylabel("Class", size=20)
        plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def splitted_classes_distribution(data: pd.DataFrame, label_dict: Dict[str, int], 
                                 save_path: Optional[str] = None) -> None:

    """
    Plot class distributions for train, validation, and test splits.

    Args:
        data (pd.DataFrame): DataFrame with 'subject' and 'node_status' columns.
        label_dict (dict): Mapping of label names to indices (e.g., {"Theory": 0}).
        save_path (str, optional): Path to save the plot. If None, displays the plot.

    Raises:
        ValueError: If data is empty or splits are invalid.
    """
    if data.empty:
        raise ValueError("Data cannot be empty")
    
    # Create subplots for train, val, test
    fig, axes = plt.subplots(ncols=3, figsize=(21, 6))

    # Get indices for each split
    train_index = data[data['node_status'] == 'train'].index
    val_index = data[data['node_status'] == 'validation'].index
    test_index = data[data['node_status'] == 'test'].index

    if len(train_index) == 0 or len(val_index) == 0 or len(test_index) == 0:
        raise ValueError("All splits (train, validation, test) must be non-empty")
    
    
    # Reverse label_dict for mapping names to indices
    inv_label_dict = {v: k for k, v in label_dict.items()}
    
    # Train split
    train_labels = (node_data.loc[train_index, 'subject']).map(inv_label_dict)
    train_label_counter = dict(collections.Counter(train_labels)).items()
    train_label_count = [x[1] for x in sorted(train_label_counter)]

    # Validation split (fixed bug: was using train_index)
    val_labels = node_data.loc[train_index, 'subject'].map(inv_label_dict)
    val_label_counter = dict(collections.Counter(val_labels)).items()
    val_label_count = [x[1] for x in sorted(val_label_counter)]

    # Test split
    test_labels = node_data.loc[test_index, 'subject'].map(inv_label_dict)
    test_label_counter = dict(collections.Counter(test_labels)).items()
    test_label_count = [x[1] for x in sorted(test_label_counter)]


    # Plot train distribution
    axes[0].bar(range(7), train_label_count)
    axes[0].set_xlabel("class", size=20)
    axes[0].set_title("Training")

    # Plot validation distribution
    axes[1].bar(range(7), val_label_count)
    axes[1].set_xlabel("class", size=20)
    axes[1].set_title("Validation")

    # Plot test distribution
    axes[2].bar(range(7), test_label_count)
    axes[2].set_xlabel("class", size=20)
    axes[2].set_title("Test")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    
    edgelist, node_data = load_data('cora\cora')
    labels = node_data['subject']
    splitted_nodes = split_nodes(node_data)
    Gnx = create_graph(edgelist)
    plot_node_degree_distribution(Gnx)
    visualize_degree_centrality(Gnx)
    visualize_graph_by_labels(Gnx, labels, label_dict)
    class_distribution(labels)
    class_connection_matrix(Gnx, node_data, label_dict)
    splitted_classes_distribution(node_data, label_dict)
    