import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import NormalizeFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import random
import time
import os

# Import the ReLearn model from your existing code
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

def load_dblp_dataset():
    """Load the DBLP dataset and convert to NetworkX graph"""
    # Load the dataset - DBLP is part of the CitationFull collection
    dataset = CitationFull(root='./data/DBLP', name='DBLP', transform=NormalizeFeatures())
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
        selected_nodes = sorted(np.random.choice(G.number_of_nodes(), min(num_nodes, G.number_of_nodes()), replace=False))
    
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

def analyze_dblp_dataset():
    """Analyze the DBLP dataset and print key statistics"""
    G, latent_positions, labels = load_dblp_dataset()
    
    print(f"DBLP Dataset Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of features: {latent_positions.shape[1]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Calculate graph metrics
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(G)}")
    
    # If not connected, analyze the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Largest component size: {len(largest_cc)} nodes")
    
    # Analyze class distribution
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
        print(f"  Class {cls}: {count} nodes ({count/len(labels)*100:.2f}%)")
    
    return G, latent_positions, labels

# Override the visualize_edge_embeddings function to use adaptive perplexity
def visualize_edge_embeddings(edge_embeddings, edge_index, graph, method='tsne', title="Edge Embeddings Visualization"):
    """Visualize edge embeddings using dimensionality reduction with adaptive parameters"""
    # Apply dimensionality reduction
    n_samples = edge_embeddings.shape[0]
    
    # Choose appropriate perplexity (must be less than n_samples)
    perplexity = min(30, max(5, n_samples // 5))
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        print(f"Using TSNE with perplexity={perplexity} for {n_samples} samples")
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    # Convert to numpy for dimensionality reduction
    edge_embeddings_np = edge_embeddings.cpu().numpy()
    reduced_embeddings = reducer.fit_transform(edge_embeddings_np)
    
    # Extract edge information for coloring - ensure same length as reduced_embeddings
    edge_count = edge_embeddings.shape[0]
    
    # Get edge weights (assuming weighted graph)
    edge_weights = []
    for i in range(edge_count):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        try:
            weight = graph[u][v].get('weight', 1.0)
        except KeyError:
            # Handle case where edge might not exist in graph (e.g., during testing)
            weight = 0.0
        edge_weights.append(weight)
    
    # Choose appropriate number of clusters based on dataset size
    n_clusters = min(5, max(2, n_samples // 20))
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(edge_embeddings_np)
    
    plt.figure(figsize=(16, 6))
    
    # Plot by edge weight
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=edge_weights, 
        cmap='plasma', 
        s=50, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Edge Weight')
    plt.title(f"{title} (Colored by Edge Weight)")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    # Plot by cluster
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=clusters, 
        cmap='tab10', 
        s=50, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} (Clustered)")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    plt.tight_layout()
    plt.show()
    
    return clusters

# Override visualize_node_embeddings with adaptive perplexity
def visualize_node_embeddings(model, graph, latent_positions, labels=None):
    """Visualize node embeddings with adaptive TSNE parameters"""
    # Convert graph to PyTorch Geometric format
    edge_list = list(graph.edges())
    edge_index = torch.tensor([[u, v] for u, v in edge_list] + [[v, u] for u, v in edge_list], dtype=torch.long).t()
    
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        x = torch.tensor(latent_positions, dtype=torch.float)
        node_embeddings = model.encode(x, edge_index)
        
    # Reduce dimensionality to 2D for visualization
    node_embeddings_np = node_embeddings.cpu().numpy()
    n_samples = node_embeddings_np.shape[0]
    
    # Adaptive perplexity (must be less than n_samples)
    perplexity = min(30, max(5, n_samples // 5))
    print(f"Using TSNE with perplexity={perplexity} for {n_samples} samples")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
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

def run_dblp_experiment(num_nodes=500, class_balanced=True):
    """Run ReLearn on the DBLP dataset"""
    print("Loading DBLP dataset...")
    G, latent_positions, labels = load_dblp_dataset()
    
    print(f"Getting subset of {num_nodes} nodes...")
    subgraph, sub_latent_positions, sub_labels = get_subset_graph(
        G, latent_positions, labels, num_nodes=num_nodes, class_balanced=class_balanced
    )
    
    # Print subset statistics
    print(f"Subset statistics:")
    print(f"  Nodes: {subgraph.number_of_nodes()}")
    print(f"  Edges: {subgraph.number_of_edges()}")
    print(f"  Features: {sub_latent_positions.shape[1]}")
    if sub_labels is not None:
        print(f"  Classes: {len(np.unique(sub_labels))}")
    
    print("Creating and training ReLearn model...")
    # Model configuration
    input_dim = sub_latent_positions.shape[1]
    model = ReLearnModel(
        input_dim=input_dim,
        gcn_hidden_dims=[64, 128],
        edge_mlp_hidden_dims=[128, 64],
        decoder_hidden_dims=[64, 32],
        z_dim=32,
        num_mixtures=5,
        dropout=0.1
    )
    
    # Train the model with smaller batches for edge visualization
    # Determine if we have enough edges for meaningful visualization
    edge_count = subgraph.number_of_edges()
    latent_viz = edge_count >= 10  # Only visualize if we have enough edges
    
    # Train the model
    model, loss_data, clusters = run_relearn_pipeline(
        subgraph,
        sub_latent_positions,
        gcn_hidden_dims=[64, 128],
        edge_mlp_hidden_dims=[128, 64],
        decoder_hidden_dims=[64, 32],
        z_dim=32,
        num_mixtures=5,
        num_epochs=100,
        kl_annealing=True,
        annealing_epochs=50,
        kl_start=0.0,
        kl_end=1.0,
        temperature_start=1.0,
        temperature_end=0.1,
        lambda1=0.5,
        lambda2=0.5,
        lr=0.001,
        dropout=0.1,
        latent_viz=latent_viz
    )
    
    # Evaluate link prediction
    print("\nEvaluating link prediction...")
    accuracy, auc = evaluate_link_prediction(model, subgraph, sub_latent_positions)
    print(f"Link prediction accuracy: {accuracy:.4f}")
    print(f"Link prediction AUC: {auc:.4f}")
    
    # Visualize node embeddings with class labels if we have enough nodes
    if subgraph.number_of_nodes() >= 10:
        print("\nVisualizing node embeddings...")
        visualize_node_embeddings(model, subgraph, sub_latent_positions, sub_labels)
    
    return model, subgraph, sub_latent_positions, sub_labels, loss_data

def compare_cora_dblp(num_nodes=500):
    """Compare ReLearn performance on Cora and DBLP datasets"""
    print("=== Comparing Cora and DBLP Datasets ===")
    
    # Load Cora (using your existing function)
    print("\nLoading Cora dataset...")
    from load_cora import load_cora_dataset  # Adjust the import to match your module name
    cora_G, cora_latent, cora_labels = load_cora_dataset()
    cora_subG, cora_sub_latent, cora_sub_labels = get_subset_graph(
        cora_G, cora_latent, cora_labels, num_nodes=num_nodes
    )
    
    # Load DBLP
    print("\nLoading DBLP dataset...")
    dblp_G, dblp_latent, dblp_labels = load_dblp_dataset()
    dblp_subG, dblp_sub_latent, dblp_sub_labels = get_subset_graph(
        dblp_G, dblp_latent, dblp_labels, num_nodes=num_nodes
    )
    
    # Print comparison statistics
    print("\n=== Dataset Statistics Comparison ===")
    print(f"{'Metric':<20} {'Cora':<10} {'DBLP':<10}")
    print(f"{'-'*40}")
    print(f"{'Nodes':<20} {cora_G.number_of_nodes():<10} {dblp_G.number_of_nodes():<10}")
    print(f"{'Edges':<20} {cora_G.number_of_edges():<10} {dblp_G.number_of_edges():<10}")
    print(f"{'Features':<20} {cora_latent.shape[1]:<10} {dblp_latent.shape[1]:<10}")
    print(f"{'Classes':<20} {len(np.unique(cora_labels)):<10} {len(np.unique(dblp_labels)):<10}")
    print(f"{'Avg Degree':<20} {2*cora_G.number_of_edges()/cora_G.number_of_nodes():.2f} {2*dblp_G.number_of_edges()/dblp_G.number_of_nodes():.2f}")
    
    # Subset comparison
    print("\n=== Subset Statistics Comparison ===")
    print(f"{'Metric':<20} {'Cora':<10} {'DBLP':<10}")
    print(f"{'-'*40}")
    print(f"{'Subset Nodes':<20} {cora_subG.number_of_nodes():<10} {dblp_subG.number_of_nodes():<10}")
    print(f"{'Subset Edges':<20} {cora_subG.number_of_edges():<10} {dblp_subG.number_of_edges():<10}")
    print(f"{'Subset Avg Degree':<20} {2*cora_subG.number_of_edges()/cora_subG.number_of_nodes():.2f} {2*dblp_subG.number_of_edges()/dblp_subG.number_of_nodes():.2f}")
    
    # Train models
    print("\n=== Training Models ===")
    
    # Common model hyperparameters
    common_params = {
        'gcn_hidden_dims': [64, 128],
        'edge_mlp_hidden_dims': [128, 64],
        'decoder_hidden_dims': [64, 32],
        'z_dim': 32,
        'num_mixtures': 5,
        'num_epochs': 100,
        'kl_annealing': True,
        'annealing_epochs': 50,
        'kl_start': 0.0,
        'kl_end': 1.0,
        'temperature_start': 1.0,
        'temperature_end': 0.1,
        'lambda1': 0.5,
        'lambda2': 0.5,
        'lr': 0.001,
        'dropout': 0.1,
        'latent_viz': False  # Turn off visualization for comparison
    }
    
    # Train Cora model
    print("\nTraining on Cora subset...")
    cora_model = ReLearnModel(
        input_dim=cora_sub_latent.shape[1],
        gcn_hidden_dims=common_params['gcn_hidden_dims'],
        edge_mlp_hidden_dims=common_params['edge_mlp_hidden_dims'],
        decoder_hidden_dims=common_params['decoder_hidden_dims'],
        z_dim=common_params['z_dim'],
        num_mixtures=common_params['num_mixtures'],
        dropout=common_params['dropout']
    )
    _, cora_loss_data, _ = run_relearn_pipeline(cora_subG, cora_sub_latent, **common_params)
    
    # Train DBLP model
    print("\nTraining on DBLP subset...")
    dblp_model = ReLearnModel(
        input_dim=dblp_sub_latent.shape[1],
        gcn_hidden_dims=common_params['gcn_hidden_dims'],
        edge_mlp_hidden_dims=common_params['edge_mlp_hidden_dims'],
        decoder_hidden_dims=common_params['decoder_hidden_dims'],
        z_dim=common_params['z_dim'],
        num_mixtures=common_params['num_mixtures'],
        dropout=common_params['dropout']
    )
    _, dblp_loss_data, _ = run_relearn_pipeline(dblp_subG, dblp_sub_latent, **common_params)
    
    # Evaluate link prediction
    print("\n=== Link Prediction Performance ===")
    cora_acc, cora_auc = evaluate_link_prediction(cora_model, cora_subG, cora_sub_latent)
    dblp_acc, dblp_auc = evaluate_link_prediction(dblp_model, dblp_subG, dblp_sub_latent)
    
    print(f"{'Metric':<20} {'Cora':<10} {'DBLP':<10}")
    print(f"{'-'*40}")
    print(f"{'Accuracy':<20} {cora_acc:.4f} {dblp_acc:.4f}")
    print(f"{'AUC':<20} {cora_auc:.4f} {dblp_auc:.4f}")
    
    # Plot comparison of loss curves
    plt.figure(figsize=(12, 10))
    
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(cora_loss_data['total'], label='Cora')
    plt.plot(dblp_loss_data['total'], label='DBLP')
    plt.title('Total Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Network loss
    plt.subplot(2, 2, 2)
    plt.plot(cora_loss_data['network'], label='Cora')
    plt.plot(dblp_loss_data['network'], label='DBLP')
    plt.title('Network Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Attribute loss
    plt.subplot(2, 2, 3)
    plt.plot(cora_loss_data['attribute'], label='Cora')
    plt.plot(dblp_loss_data['attribute'], label='DBLP')
    plt.title('Attribute Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # KL loss
    plt.subplot(2, 2, 4)
    plt.plot(cora_loss_data['kl'], label='Cora')
    plt.plot(dblp_loss_data['kl'], label='DBLP')
    plt.title('KL Divergence Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'cora': {'model': cora_model, 'graph': cora_subG, 'features': cora_sub_latent, 'labels': cora_sub_labels},
        'dblp': {'model': dblp_model, 'graph': dblp_subG, 'features': dblp_sub_latent, 'labels': dblp_sub_labels}
    }

if __name__ == "__main__":
    # Analyze the DBLP dataset
    print("\n=== DBLP Dataset Analysis ===")
    analyze_dblp_dataset()
    
    # Run ReLearn on DBLP with a smaller subset
    print("\n=== Running ReLearn on DBLP ===")
    model, subgraph, features, labels, loss_data = run_dblp_experiment(num_nodes=100)  # Reduced from 500 to avoid potential issues
    
    # If all works well, can try with more nodes
    try:
        print("\n=== Running ReLearn on larger DBLP subset ===")
        model_large, subgraph_large, features_large, labels_large, loss_data_large = run_dblp_experiment(num_nodes=500)
        
        # Optionally, compare with Cora
        print("\n=== Comparing with Cora dataset ===")
        comparison_results = compare_cora_dblp(num_nodes=300)  # Using 300 instead of 500 for safer comparison
    except Exception as e:
        print(f"Error running larger experiments: {e}")
        print("Try with smaller node counts or adjust parameters.")