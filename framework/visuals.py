import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
import seaborn as sns
from torch_geometric.utils import to_networkx

def visualize_training(history):
    """
    Visualize training history
    
    Args:
        history: Dict with training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot total, KL, and reconstruction losses
    plt.subplot(1, 2, 1)
    plt.plot(history["total_loss"], label="Total Loss")
    plt.plot(history["kl_loss"], label="KL Loss")
    plt.plot(history["recon_loss"], label="Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    
    # Plot decoder-specific losses
    plt.subplot(1, 2, 2)
    for name, losses in history["decoder_losses"].items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Decoder Losses")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_latent_space(model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize latent space using t-SNE
    
    Args:
        model: Trained GraphVAE model
        data_loader: DataLoader providing graph data
        device: Device to use for inference
    """
    model.eval()
    model = model.to(device)
    
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            
            # Get latent representations
            mu, _ = model.encode(x, edge_index=edge_index)
            z = mu.cpu().numpy()
            
            all_z.append(z)
            
            # Add labels if available
            if hasattr(batch, 'y') and batch.y is not None:
                all_labels.append(batch.y.cpu().numpy())
    
    # Concatenate batches
    all_z = np.vstack(all_z) if len(all_z) > 1 else all_z[0]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(all_z)
    
    plt.figure(figsize=(8, 6))
    
    # Plot with or without labels
    if all_labels:
        all_labels = np.concatenate(all_labels) if len(all_labels) > 1 else all_labels[0]
        unique_labels = np.unique(all_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            plt.scatter(z_tsne[mask, 0], z_tsne[mask, 1], c=[colors[i]], label=f"Class {label}")
        
        plt.legend()
    else:
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.8)
    
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

def visualize_graph_reconstruction(model, data, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize original and reconstructed graph
    
    Args:
        model: Trained GraphVAE model
        data: PyG Data object containing a single graph
        device: Device to use for inference
    """
    model.eval()
    model = model.to(device)
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_labels = data.edge_labels.to(device)
    
    # Get original graph
    #original_graph = to_networkx(data, edge_attrs=edge_labels, to_undirected=True)
    original_graph = nx.Graph()

    # Add edges with weights
    # Transpose edge_index to get a list of tuples [(src, trg), ...]
    edge_list = edge_index.T.tolist()

    # Iterate through edges and weights to add them to the graph
    for i, (u, v) in enumerate(edge_list):
        weight = edge_labels[i].item() # .item() to get scalar value from tensor
        original_graph.add_edge(u, v, weight=weight)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, edge_index)
        
        # Get reconstructed adjacency matrix
        adj_decoder = model.get_decoder("adj_decoder")
        if adj_decoder:
            recon_outputs = adj_decoder(outputs["z"])
            if "adj_logits" in recon_outputs:
                recon_adj = (recon_outputs["adj_logits"]).cpu().numpy()
                
                # Create reconstructed graph
                recon_graph = nx.from_numpy_array(recon_adj)
                
                # Visualize both graphs
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                pos = nx.spring_layout(original_graph, seed=42)
                nx.draw(original_graph, pos, with_labels=True, node_size=500, node_color="lightblue", 
                        font_weight="bold", width=1.5, edge_color="gray")
                plt.title("Original Graph")
                
                plt.subplot(2, 2, 2)
                nx.draw(recon_graph, pos, with_labels=True, node_size=500, node_color="lightgreen", 
                        font_weight="bold", width=1.5, edge_color="gray")
                plt.title(f"Reconstructed Graph")
                
                # Original Adjacency Matrix
                plt.subplot(2, 2, 3)
                sns.heatmap(nx.adjacency_matrix(original_graph).toarray(), cmap='YlGnBu', cbar=True)
                plt.title('Original Adjacency Matrix')
                
                plt.subplot(2, 2, 4)
                sns.heatmap(recon_outputs["adj_logits"], cmap='YlGnBu', cbar=True)
                plt.title('Reconstructed Adjacency Matrix')
                
                plt.tight_layout()
                plt.show()
            else:
                print("Adjacency logits not found in decoder outputs")
        else:
            print("Adjacency decoder not found")

def visualize_node_features_reconstruction(model, data, sample_features=20, 
                                         device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize original and reconstructed node features
    
    Args:
        model: Trained GraphVAE model
        data: PyG Data object containing a single graph
        sample_features: Number of feature dimensions to visualize
        device: Device to use for inference
    """
    model.eval()
    model = model.to(device)
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, edge_index=edge_index)
        
        # Get reconstructed node features
        node_decoder = model.get_decoder("node_attr_decoder")
        if node_decoder:
            recon_outputs = node_decoder(outputs["z"])
            if "node_features" in recon_outputs:
                original_features = x.cpu().numpy()
                recon_features = recon_outputs["node_features"].cpu().numpy()
                
                # Sample nodes and features
                num_nodes = min(5, original_features.shape[0])
                feature_dims = min(sample_features, original_features.shape[1])
                
                # Visualize feature reconstruction
                plt.figure(figsize=(12, 8))
                
                for i in range(num_nodes):
                    plt.subplot(num_nodes, 1, i+1)
                    
                    # Original features
                    plt.plot(original_features[i, :feature_dims], 'bo-', label=f"Original (Node {i})")
                    
                    # Reconstructed features
                    plt.plot(recon_features[i, :feature_dims], 'ro--', label=f"Reconstructed (Node {i})")
                    
                    plt.xlabel("Feature Dimension")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.suptitle("Node Feature Reconstruction")
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.show()
            else:
                print("Node features not found in decoder outputs")
        else:
            print("Node attribute decoder not found")

def interpolate_in_latent_space(model, data1, data2, steps=10, 
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Interpolate between two graphs in latent space and visualize the results
    
    Args:
        model: Trained GraphVAE model
        data1: First graph (PyG Data)
        data2: Second graph (PyG Data)
        steps: Number of interpolation steps
        device: Device to use for inference
    """
    model.eval()
    model = model.to(device)
    
    # Move data to device
    x1, edge_index1 = data1.x.to(device), data1.edge_index.to(device)
    x2, edge_index2 = data2.x.to(device), data2.edge_index.to(device)
    
    # Get latent representations
    with torch.no_grad():
        mu1, _ = model.encode(x1, edge_index1)
        mu2, _ = model.encode(x2, edge_index2)
        
        # Interpolate in latent space
        alphas = np.linspace(0, 1, steps)
        interp_graphs = []
        
        for alpha in alphas:
            # Interpolated latent representation
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode
            adj_decoder = model.get_decoder("adj_decoder")
            if adj_decoder:
                recon_outputs = adj_decoder(z_interp)
                if "adj_logits" in recon_outputs:
                    recon_adj = (recon_outputs["adj_logits"]).cpu().numpy()
                    threshold = 0.5
                    recon_adj_binary = (recon_adj > threshold).astype(int)
                    interp_graph = nx.from_numpy_array(recon_adj_binary)
                    interp_graphs.append(interp_graph)
        
        # Visualize interpolation
        if interp_graphs:
            rows = int(np.ceil(steps / 5))
            cols = min(steps, 5)
            
            plt.figure(figsize=(15, 3 * rows))
            
            for i, graph in enumerate(interp_graphs):
                plt.subplot(rows, cols, i + 1)
                pos = nx.spring_layout(graph, seed=42)
                nx.draw(graph, pos, with_labels=True, node_size=300, node_color="lightblue", 
                        font_size=8, width=1.0, edge_color="gray")
                plt.title(f"α = {alphas[i]:.2f}")
            
            plt.suptitle("Interpolation in Latent Space")
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_correlogram(*correlation_matrices, titles=None, cmap='viridis', organ_names_dict=None, 
                     remove_diagonal=False, triangular=False, annot=False, savefile=None):
    """
    Plots multiple correlograms (NumPy arrays) in a row as subplots with a shared color scale and adjusted axis labels.

    Args:
        *correlation_matrices: Variable number of NumPy arrays (correlation matrices).
        titles: Optional list of titles for each correlogram. If None, default titles are used.
        cmap: Optional colormap for the heatmaps.
        organ_names_dict: Optional dictionary mapping indices to organ names. If provided, axis labels are adjusted.
        savefile: Optional file name (without extension) for saving the figure
    """

    num_matrices = len(correlation_matrices)

    if num_matrices == 0:
        print("No correlation matrices provided.")
        return

    if titles is None:
        titles = [f"Correlogram {i+1}" for i in range(num_matrices)]
    elif len(titles) != num_matrices:
        print("Number of titles does not match number of matrices.")
        titles = [f"Correlogram {i+1}" for i in range(num_matrices)]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    numpy_matrices = []
    for matrix in correlation_matrices:
        if torch.is_tensor(matrix):
            matrix = matrix.detach().numpy()
        numpy_matrices.append(matrix)

    correlation_matrices = numpy_matrices

    # Find the global min and max for the color scale
    global_min = min(np.nanmin(corr_matrix) for corr_matrix in correlation_matrices)
    global_max = max(np.nanmax(corr_matrix) for corr_matrix in correlation_matrices)

    if num_matrices == 1:
        plt.figure(figsize=(8 + 2*(organ_names_dict is not None), 8 + 2*(organ_names_dict is not None)))
    else:
        plt.figure(figsize=(12 * num_matrices, 12))

    for i, corr_matrix in enumerate(correlation_matrices):
        if remove_diagonal:
            np.fill_diagonal(corr_matrix, np.nan)
        if triangular:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        else:
            mask = False

        plt.subplot(1, num_matrices, i + 1)
        ax = sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            fmt='.2f',
            linewidths=0,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12},
            square=True,
            linecolor='white',
            vmin=global_min,  # Set global min
            vmax=global_max,   # Set global max
            mask=mask
        )
        plt.title(titles[i])

        if organ_names_dict is not None:
            num_organs = corr_matrix.shape[0]
            if len(organ_names_dict) == num_organs:
                organ_names = [organ_names_dict.get(j, f"Index {j}") for j in range(num_organs)]
                ax.set_xticks(np.array(range(num_organs))+0.5)
                ax.set_xticklabels(organ_names, rotation=90)
                ax.set_yticks(np.array(range(num_organs))+0.5)
                ax.set_yticklabels(organ_names, rotation=0)
            else:
                print(f"Warning: Length of organ_names_dict ({len(organ_names_dict)}) does not match matrix size ({num_organs}).")

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile + ".png", dpi=600, bbox_inches='tight')
    plt.show()
    return plt