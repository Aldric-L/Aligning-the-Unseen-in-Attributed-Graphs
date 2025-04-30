import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import StandardScaler
import random
import time
import os

from relearn.model import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def load_cora_dataset():
    """Load the Cora dataset and convert to NetworkX graph"""
    # Load the dataset
    dataset = Planetoid(root='./data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    # Create a NetworkX graph from the data
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)
    
    # Add edges
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)
    
    # Create node features (latent positions)
    latent_positions = data.x.cpu().numpy()
    
    # Normalize features for better convergence
    scaler = StandardScaler()
    latent_positions = scaler.fit_transform(latent_positions)
    
    return G, latent_positions, data.y.cpu().numpy()

def get_subset_graph(G, latent_positions, labels, num_nodes=500, class_balanced=True):
    """Get a subset of the graph for faster testing"""
    if class_balanced and labels is not None:
        # Get nodes from each class
        unique_classes = np.unique(labels)
        nodes_per_class = num_nodes // len(unique_classes)
        selected_nodes = []
        
        for cls in unique_classes:
            class_nodes = np.where(labels == cls)[0]
            if len(class_nodes) > nodes_per_class:
                selected_nodes.extend(np.random.choice(class_nodes, nodes_per_class, replace=False))
            else:
                selected_nodes.extend(class_nodes)
        
        # If we still need more nodes, add them randomly
        if len(selected_nodes) < num_nodes:
            remaining = num_nodes - len(selected_nodes)
            all_nodes = set(range(len(labels)))
            remaining_nodes = list(all_nodes - set(selected_nodes))
            selected_nodes.extend(np.random.choice(remaining_nodes, min(remaining, len(remaining_nodes)), replace=False))
            
        selected_nodes = sorted(selected_nodes)
    else:
        # Random subset of nodes
        selected_nodes = sorted(np.random.choice(G.number_of_nodes(), num_nodes, replace=False))
    
    # Create subgraph
    subgraph = G.subgraph(selected_nodes).copy()
    
    # Remap node IDs to be consecutive integers starting from 0
    mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}
    subgraph = nx.relabel_nodes(subgraph, mapping)
    
    # Get corresponding latent positions
    sub_latent_positions = latent_positions[selected_nodes]
    
    # Get corresponding labels if available
    sub_labels = labels[selected_nodes] if labels is not None else None
    
    return subgraph, sub_latent_positions, sub_labels

def visualize_node_embeddings(model, graph, latent_positions, labels=None):
    """Visualize node embeddings"""
    # Convert graph to PyTorch Geometric format
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        x = torch.tensor(latent_positions, dtype=torch.float)
        node_embeddings = model.encode(x, edge_index)
        
    # Reduce dimensionality to 2D for visualization
    from sklearn.manifold import TSNE
    node_embeddings_np = node_embeddings.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(node_embeddings_np)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Color by class
        unique_classes = len(np.unique(labels))
        scatter = plt.scatter(
            node_embeddings_2d[:, 0], 
            node_embeddings_2d[:, 1], 
            c=labels, 
            cmap='tab10', 
            s=50, 
            alpha=0.8
        )
        plt.colorbar(scatter, label='Class')
    else:
        # Color by degree
        degrees = dict(nx.degree(graph))
        scatter = plt.scatter(
            node_embeddings_2d[:, 0], 
            node_embeddings_2d[:, 1], 
            c=[degrees[i] for i in range(len(node_embeddings_2d))],
            cmap='viridis', 
            s=50, 
            alpha=0.8
        )
        plt.colorbar(scatter, label='Node Degree')
    
    plt.title("Node Embeddings Visualization")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.tight_layout()
    plt.show()
    

