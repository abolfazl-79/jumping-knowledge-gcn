import pandas as pd
import os
from pathlib import Path
import networkx as nx
import numpy as np
import random
import torch


label_dict = {
        "Theory": 0,
        "Reinforcement_Learning": 1,
        "Genetic_Algorithms": 2,
        "Neural_Networks": 3,
        "Probabilistic_Methods": 4,
        "Case_Based": 5,
        "Rule_Learning": 6}


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Cora dataset's edge list and node features.

    Args:
        data_dir (str): Directory containing 'cora.cites' and 'cora.content' files.

    Returns:
        tuple: (edgelist, node_data) where edgelist is the citation edges and node_data
               contains node features and labels.

    Raises:
        FileNotFoundError: If the data files are not found in the specified directory.
    """

    # Use Path for cross-platform compatibility
    data_path = Path(data_dir)  
    cites_path = data_path / "cora.cites"
    content_path = data_path / "cora.content"
    
    if not cites_path.exists() or not content_path.exists():
        raise FileExistsError(f"Required files are not found in {data_dir}")
    

    # Load edgelist
    edgelist = pd.read_csv(cites_path, sep='\t', header=None, names=["target", "source"])
    edgelist["label"] = "cites"
    
    # Load nodes features and labels
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = ["node_id"] + feature_names + ["subject"]
    node_data = pd.read_csv(content_path, sep='\t', header=None, names=column_names)

    # Reset node IDs to consecutive integers (0 to n-1) and update edgelist
    id_map = {old_id: new_id for new_id, old_id in enumerate(node_data["node_id"])}

    # node_data.set_index("node_id", inplace=True)
    node_data["node_id"] = node_data["node_id"].map(id_map)
    edgelist["source"] = edgelist["source"].map(id_map)
    edgelist["target"] = edgelist["target"].map(id_map)

    # Converting nodes subjects to their integer values
    node_data['subject'] = node_data['subject'].map(label_dict)

    return (edgelist, node_data)




def create_graph(edgelist: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph from the edge list.

    Args:
        edgelist (pd.DataFrame): DataFrame with 'source', 'target', and 'label' columns.

    Returns:
        nx.Graph: Undirected graph with edges labeled 'cites' and nodes labeled 'paper'.
    """
    Gnx = nx.from_pandas_edgelist(edgelist, source="source", target="target", edge_attr="label")
    nx.set_node_attributes(Gnx, "paper", "label")
    return Gnx
    

def get_feature_matrix(node_data: pd.DataFrame) -> torch.Tensor:
    """
    Extract feature matrix from node data and move it to GPU.

    Args:
        node_data (pd.DataFrame): DataFrame with feature columns and 'subject'.

    Returns:
        torch.Tensor: Feature matrix on CUDA device.
    """
    feature_matrix = node_data.drop(columns=['node_status', 'subject']).values
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float).to('cuda')
    return feature_matrix
    

def split_nodes(node_data: pd.DataFrame,
                train_rate: float = 0.6,
                val_rate: float = 0.2,
                test_rate: float = 0.2,
                seed: int = 42) -> pd.DataFrame:
    
    """
    Split nodes into train, validation, and test sets randomly.

    Args:
        node_data (pd.DataFrame): DataFrame with node features and labels.
        train_rate (float): Proportion of nodes for training (default: 0.6).
        val_rate (float): Proportion of nodes for validation (default: 0.2).
        test_rate (float): Proportion of nodes for testing (default: 0.2).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Updated DataFrame with 'node_status' column indicating split.

    Raises:
        ValueError: If train_rate + val_rate + test_rate does not equal 1.
    """

    if not np.isclose(train_rate + val_rate + test_rate, 1.0):
        raise ValueError("Train, validation, and test rates must sum to 1")
    
    np.random.seed(seed)
    num_nodes = node_data.shape[0]
    labels = np.random.choice(
        ["train", "validation", "test"],
        size=num_nodes,
        p=[train_rate, val_rate, test_rate]
    )
    node_data["node_status"] = labels
    return node_data



def get_splitted_labels(node_data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split labels into train, validation, and test tensors based on node status.

    Args:
        node_data (pd.DataFrame): DataFrame with 'subject' and 'node_status' columns.

    Returns:
        tuple: (y_train, y_val, y_test) tensors on CUDA device.
    """
    # y_train = node_data.where(node_data['node_status'] == 'train').dropna()['subject'].map(label_dict).values
    # y_train = torch.tensor(y_train).to('cuda')
    # Vectorized label mapping and splitting
    y_train = node_data.where(node_data['node_status'] == 'train').dropna()['subject'].values
    y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
    y_val = node_data.where(node_data['node_status'] == 'validation').dropna()['subject'].values
    y_val = torch.tensor(y_val, dtype=torch.long).to('cuda')
    y_test = node_data.where(node_data['node_status'] == 'test').dropna()['subject'].values
    y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')

    return (y_train, y_val, y_test)


if __name__ == '__main__':
    data_dir = "cora/cora"  # Adjust path as needed
    try:
        
        edgelist, node_data = load_data(data_dir)
        print("Edge list sample:")
        print(edgelist.head())
        print("\nNode data sample:")
        print(node_data.head())

        # Split nodes and labels
        node_data = split_nodes(node_data)
        y_train, y_val, y_test = get_splitted_labels(node_data)
        print("\nTrain labels sample:", y_train[:5])
        print("Validation labels sample:", y_val[:5])
        print("Test labels sample:", y_test[:5])

        # Create graph and feature matrix
        graph = create_graph(edgelist)
        features = get_feature_matrix(node_data)
        adj_matrix = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float).to("cuda")
        print("\nGraph nodes:", len(graph.nodes))
        print("Feature matrix shape:", features.shape)
        print("Adjacency matrix shape:", adj_matrix.shape)

    except Exception as e:
        print(f"Error during execution: {e}")





