import torch
import torch.optim as optim

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

from tqdm import tqdm

from framework.utils import *
from framework.GraphVAE import GraphVAE
from framework.KLAnnealingScheduler import KLAnnealingScheduler
from framework.encoder import *
from framework.decoder import *

def create_graphvae_model(
    input_dim: int,
    latent_dim: int,
    encoder_hidden_dims: List[int] = [32, 64],
    gcn_layers: int = 2,
    fc_layers: int = 1,
    adj_decoder_hidden_dims: List[int] = [64, 32],
    node_decoder_hidden_dims: List[int] = [64, 32],
    dropout: float = 0.1,
    kl_annealing: bool = True,
    anneal_steps: int = 1000,
    anneal_end: float = 1.0,
) -> GraphVAE:
    """
    Helper function to create a GraphVAE model with standard configuration
    
    Args:
        input_dim: Input dimension of node features
        latent_dim: Latent space dimension
        encoder_hidden_dims: Hidden dimensions for encoder
        gcn_layers: Number of GCN layers in encoder
        fc_layers: Number of FC layers in encoder
        adj_decoder_hidden_dims: Hidden dimensions for adjacency decoder
        node_decoder_hidden_dims: Hidden dimensions for node attribute decoder
        dropout: Dropout probability
        kl_annealing: Whether to use KL annealing
        anneal_steps: Steps for KL annealing
        anneal_end: Max beta value for KL annealing
        
    Returns:
        Configured GraphVAE model
    """
    # Create encoder
    encoder = Encoder(
        input_dim=input_dim,
        hidden_dims=encoder_hidden_dims,
        latent_dim=latent_dim,
        gcn_layers=gcn_layers,
        fc_layers=fc_layers,
        dropout=dropout
    )
    
    # Create decoders
    adjacency_decoder = AdjacencyDecoder(
        latent_dim=latent_dim,
        hidden_dims=adj_decoder_hidden_dims,
        dropout=dropout
    )
    
    node_decoder = NodeAttributeDecoder(
        latent_dim=latent_dim,
        output_dim=input_dim,
        hidden_dims=node_decoder_hidden_dims,
        dropout=dropout
    )
    
    # Create KL annealing scheduler if needed
    kl_scheduler = None
    if kl_annealing:
        kl_scheduler = KLAnnealingScheduler(
            anneal_start=0.0,
            anneal_end=anneal_end,
            anneal_steps=anneal_steps,
            anneal_type='sigmoid'
        )
    
    # Create GraphVAE model
    model = GraphVAE(
        encoder=encoder,
        decoders=[adjacency_decoder, node_decoder],
        kl_scheduler=kl_scheduler
    )
    
    return model


def train_step(
    model: GraphVAE,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    adj_matrix: torch.Tensor,
    decoder_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Single training step
    
    Args:
        model: GraphVAE model
        optimizer: Optimizer
        x: Node features
        edge_index: Graph connectivity
        adj_matrix: Adjacency matrix
        decoder_weights: Optional weights for each decoder loss
        
    Returns:
        Dict with loss values
    """
    model.train()
    optimizer.zero_grad()

    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # Forward pass
    outputs = model(x, edge_index)

    src, dst = edge_index
    edge_labels = adj_matrix[src, dst].float()
    
    # Prepare targets
    targets = {
        "adj_decoder": {
            "adj_matrix": adj_matrix,
            "edge_labels": edge_labels,
            "edge_index": edge_index
        },
        "node_attr_decoder": {
            "node_features": x
        }
    }
    
    # Compute loss
    loss_dict = model.compute_loss(outputs, targets, decoder_weights)
    
    # Backward pass
    loss_dict["total_loss"].backward()
    optimizer.step()
    
    # Step KL annealing scheduler
    model.kl_scheduler.step()
    
    # Convert torch tensors to floats for logging
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

def train_model(
    model,
    data_loader,
    num_epochs=100,
    lr=0.001,
    weight_decay=1e-5,
    verbose=True,
    decoder_weights=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the GraphVAE model
    
    Args:
        model: GraphVAE model
        data_loader: DataLoader providing graph data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print progress
        decoder_weights: Optional weights for each decoder loss
        device: Device to use for training
        
    Returns:
        Dict with training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize training history
    history = {
        "total_loss": [],
        "kl_loss": [],
        "recon_loss": [],
        "decoder_losses": {}
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {
            "total_loss": 0.0,
            "kl_loss": 0.0,
            "recon_loss": 0.0,
            "decoder_losses": {}
        }
        
        # Initialize or update decoder-specific losses
        for decoder in model.decoders:
            name = decoder.name
            if name not in history["decoder_losses"]:
                history["decoder_losses"][name] = []
            if name not in epoch_losses["decoder_losses"]:
                epoch_losses["decoder_losses"][name] = 0.0
        
        num_batches = 0
        
        # Process batches
        iterator = tqdm(data_loader) if verbose else data_loader
        for batch in iterator:
            num_batches += 1
            
            # Move batch to device
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_labels = batch.edge_labels.to(device)
            adjacency_matrix = batch.adjacency_matrix.to(device)
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            outputs = model(x, edge_index=edge_index)
            
            # Prepare targets
            targets = {
                "adj_decoder": {
                    "adj_matrix": adjacency_matrix,
                    "edge_labels": edge_labels,
                    "edge_index": edge_index
                },
                "node_attr_decoder": {
                    "node_features": x
                }
            }
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, targets, decoder_weights)
            
            # Backward pass
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            # Step KL annealing scheduler
            model.kl_scheduler.step()
            
            # Update epoch losses
            epoch_losses["total_loss"] += loss_dict["total_loss"].item()
            epoch_losses["kl_loss"] += loss_dict["kl_loss"].item()
            epoch_losses["recon_loss"] += loss_dict["recon_loss"].item()
            
            for name, loss in loss_dict["decoder_losses"].items():
                epoch_losses["decoder_losses"][name] += loss.item()
        
        # Compute average losses
        for key in ["total_loss", "kl_loss", "recon_loss"]:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])
        
        for name in epoch_losses["decoder_losses"]:
            epoch_losses["decoder_losses"][name] /= num_batches
            history["decoder_losses"][name].append(epoch_losses["decoder_losses"][name])
        
        # Print epoch summary
        if verbose and (epoch + 1) % 10 == 0:
            kl_weight = model.kl_scheduler.get_weight()
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {epoch_losses['total_loss']:.4f}, "
                  f"KL: {epoch_losses['kl_loss']:.4f} (weight: {kl_weight:.4f}), "
                  f"Recon: {epoch_losses['recon_loss']:.4f}")
    
    return history

def create_fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Create a fully connected edge index (all nodes connected to all other nodes)
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        Edge index tensor [2, num_edges]
    """
    # Create all possible pairs of nodes
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                rows.append(i)
                cols.append(j)
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index

def create_full_loops_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Create a not connected edge index (all nodes connected only connected to themselves)
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        Edge index tensor [2, num_edges]
    """
    # Create all possible pairs of nodes
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:  
                rows.append(i)
                cols.append(j)
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index

def create_full_loops_adj_matrix(num_nodes: int) -> torch.Tensor:
    """
    Create a not connected adjacency matrix (all nodes connected only connected to themselves)
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    # Create all possible pairs of nodes
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    # Set diagonal to 0 (no self-loops)
    adj_matrix.fill_diagonal_(1)
    return adj_matrix

def create_fully_connected_adj_matrix(num_nodes: int) -> torch.Tensor:
    """
    Create a fully connected adjacency matrix (all 1s except diagonal)
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    adj_matrix = torch.ones(num_nodes, num_nodes)
    # Set diagonal to 0 (no self-loops)
    adj_matrix.fill_diagonal_(0)
    return adj_matrix


def train_phase1(
    model: GraphVAE,
    data_loader,
    num_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    verbose: bool = True,
    loss_coefficient: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Phase 1 training: Train with only node attribute decoder and fully connected graph
    
    Args:
        model: GraphVAE model
        data_loader: DataLoader providing graph data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print progress
        device: Device to use for training
        
    Returns:
        Dict with training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize training history
    history = {
        "total_loss": [],
        "kl_loss": [],
        "recon_loss": [],
        "decoder_losses": {"node_attr_decoder": []}
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {
            "total_loss": 0.0,
            "kl_loss": 0.0,
            "recon_loss": 0.0,
            "decoder_losses": {"node_attr_decoder": 0.0}
        }
        
        num_batches = 0
        
        # Process batches
        #iterator = tqdm(data_loader) if verbose else data_loader
        iterator = data_loader
        for batch in iterator:
            num_batches += 1
            
            # Move batch to device
            x = batch.x.to(device)
            
            # Create fully connected edge index
            num_nodes = x.size(0)
            edge_index = create_full_loops_edge_index(num_nodes).to(device)
            
            # Forward pass with fully connected edge index
            model.train()
            optimizer.zero_grad()
            outputs = model(x, edge_index=edge_index)
            
            # Prepare targets (only node features)
            targets = {
                "node_attr_decoder": {
                    "node_features": x
                }
            }
            
            # Active decoders - only node attribute decoder
            active_decoders = ["node_attr_decoder"]
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, targets, active_decoders=active_decoders)
            
            # Backward pass
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            # Step KL annealing scheduler
            model.kl_scheduler.step()
            
            # Update epoch losses
            epoch_losses["total_loss"] += loss_dict["total_loss"].item()
            epoch_losses["kl_loss"] += loss_dict["kl_loss"].item()
            epoch_losses["recon_loss"] += loss_dict["recon_loss"].item()
            
            for name, loss in loss_dict["decoder_losses"].items():
                epoch_losses["decoder_losses"][name] += loss_coefficient * loss.item()
        
        # Compute average losses
        for key in ["total_loss", "kl_loss", "recon_loss"]:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])
        
        for name in epoch_losses["decoder_losses"]:
            epoch_losses["decoder_losses"][name] /= num_batches
            history["decoder_losses"][name].append(epoch_losses["decoder_losses"][name])
        
        # Print epoch summary
        if verbose and (epoch + 1) % 10 == 0:
            kl_weight = model.kl_scheduler.get_weight()
            print(f"Phase 1 - Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {epoch_losses['total_loss']:.4f}, "
                  f"KL: {epoch_losses['kl_loss']:.4f} (weight: {kl_weight:.4f}), "
                  f"Node Recon: {epoch_losses['decoder_losses']['node_attr_decoder']:.4f}")
    
    return history


def train_phase2(
    model: GraphVAE,
    data_loader,
    latent_points: torch.tensor,
    num_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    decoder_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, List[float]]:
    """
    Phase 2 training: Freeze encoder and train with both decoders
    
    Args:
        model: GraphVAE model with pretrained encoder
        data_loader: DataLoader providing graph data with adjacency matrices
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        decoder_weights: Optional weights for each decoder loss
        verbose: Whether to print progress
        device: Device to use for training
        
    Returns:
        Dict with training history
    """
    # Move model to device
    model = model.to(device)
    
    # Freeze encoder
    model.zero_grad()
    model.set_encoder_freeze(True)
    
    # Default decoder weights if not provided
    if decoder_weights is None:
        decoder_weights = {
            "adj_decoder": 1.0,
            "node_attr_decoder": 0.5  # Lower weight for pretrained decoder
        }
        
    # Initialize optimizer (only decoder parameters will be updated)
    optimizer = optim.Adam(
        [p for name, p in model.named_parameters() if "encoder" not in name],
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Initialize training history
    history = {
        "total_loss": [],
        "kl_loss": [],
        "recon_loss": [],
        "decoder_losses": {
            "adj_decoder": [],
            "node_attr_decoder": []
        }
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {
            "total_loss": 0.0,
            "kl_loss": 0.0,
            "recon_loss": 0.0,
            "decoder_losses": {
                "adj_decoder": 0.0,
                "node_attr_decoder": 0.0
            }
        }
        
        num_batches = 0
        
        # Process batches
        #iterator = tqdm(data_loader) if verbose else data_loader
        iterator = data_loader
        for batch in iterator:
            num_batches += 1
            
            # Move batch to device
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_labels = batch.edge_labels.to(device) if hasattr(batch, 'edge_labels') else None
            adjacency_matrix = batch.adjacency_matrix.to(device) if hasattr(batch, 'adjacency_matrix') else None
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            outputs = model(x, edge_index=edge_index)

            model.set_encoder_freeze(True)
            # print("IS FULLY FROZEN?")
            # print(any(p.grad is not None for p in model.encoder.parameters()))
            # for p in model.encoder.named_parameters():
            #     print(p)
            
            # Prepare targets
            targets = {
                "adj_decoder": {
                    "adj_matrix": adjacency_matrix,
                    "edge_labels": edge_labels,
                    "edge_index": edge_index
                },
                "node_attr_decoder": {
                    "node_features": x
                }
            }
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, targets, decoder_weights=decoder_weights)
            
            # Backward pass
            loss_dict["total_loss"].backward()
            # loss_dict["total_loss"].backward(retain_graph=True)

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, "grad norm:", param.grad.norm().item())
            optimizer.step()
            
            # Step KL annealing scheduler
            model.kl_scheduler.step()
            
            # Update epoch losses
            epoch_losses["total_loss"] += loss_dict["total_loss"].item()
            epoch_losses["kl_loss"] += loss_dict["kl_loss"].item()
            epoch_losses["recon_loss"] += loss_dict["recon_loss"].item()
            
            for name, loss in loss_dict["decoder_losses"].items():
                epoch_losses["decoder_losses"][name] += loss.item()
        
        # Compute average losses
        for key in ["total_loss", "kl_loss", "recon_loss"]:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])
        
        for name in epoch_losses["decoder_losses"]:
            epoch_losses["decoder_losses"][name] /= num_batches
            history["decoder_losses"][name].append(epoch_losses["decoder_losses"][name])
        
        # Print epoch summary
        if verbose and (epoch + 1) % 10 == 0:
            kl_weight = model.kl_scheduler.get_weight()
            print(f"Phase 2 - Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {epoch_losses['total_loss']:.4f}, "
                  f"KL: {epoch_losses['kl_loss']:.4f} (weight: {kl_weight:.4f}), "
                  f"Adj: {epoch_losses['decoder_losses']['adj_decoder']:.4f}, "
                  f"Node: {epoch_losses['decoder_losses']['node_attr_decoder']:.4f}")

            if model.rbf_target is not None:
                model.train_rbf_layer(
                    decoder_name="node_attr_decoder",
                    decoder_variance_target_name="node_features_logvar",
                    latent_samples=latent_points,
                    n_centers=16,  
                    a=1,
                    n_epochs= 500,
                    lr=0.01,
                    force=True)    
    return history


def two_phase_training(
    input_dim: int,
    latent_dim: int,
    data_loader_phase1,
    data_loader_phase2,
    encoder_hidden_dims: List[int] = [32, 64],
    gcn_layers: int = 2,
    fc_layers: int = 1,
    adj_decoder_hidden_dims: List[int] = [64, 32],
    node_decoder_hidden_dims: List[int] = [64, 32],
    dropout: float = 0.1,
    phase1_epochs: int = 100,
    phase2_epochs: int = 100,
    lr_phase1: float = 0.001,
    lr_phase2: float = 0.0005,
    weight_decay: float = 1e-5,
    decoder_weights_phase2: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[GraphVAE, Dict[str, Any]]:
    """
    Complete two-phase training workflow
    
    Args:
        input_dim: Input dimension of node features
        latent_dim: Latent space dimension
        data_loader_phase1: DataLoader for phase 1 (node features only)
        data_loader_phase2: DataLoader for phase 2 (node features + adjacency)
        encoder_hidden_dims: Hidden dimensions for encoder
        gcn_layers: Number of GCN layers in encoder
        fc_layers: Number of FC layers in encoder
        adj_decoder_hidden_dims: Hidden dimensions for adjacency decoder
        node_decoder_hidden_dims: Hidden dimensions for node attribute decoder
        dropout: Dropout probability
        phase1_epochs: Number of epochs for phase 1
        phase2_epochs: Number of epochs for phase 2
        lr_phase1: Learning rate for phase 1
        lr_phase2: Learning rate for phase 2
        weight_decay: Weight decay for regularization
        decoder_weights_phase2: Optional weights for each decoder loss in phase 2
        verbose: Whether to print progress
        device: Device to use for training
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Step 1: Create encoder and node decoder for phase 1
    encoder = Encoder(
        input_dim=input_dim,
        hidden_dims=encoder_hidden_dims,
        latent_dim=latent_dim,
        gcn_layers=gcn_layers,
        fc_layers=fc_layers,
        dropout=dropout
    )
    
    node_decoder = NodeAttributeDecoder(
        latent_dim=latent_dim,
        output_dim=input_dim,
        hidden_dims=node_decoder_hidden_dims,
        dropout=dropout
    )
    
    # Create KL annealing scheduler
    kl_scheduler = KLAnnealingScheduler(
        anneal_start=0.0,
        anneal_end=1.0,
        anneal_steps=phase1_epochs * len(data_loader_phase1),
        anneal_type='sigmoid'
    )
    
    # Create initial model with only node decoder
    model_phase1 = GraphVAE(
        encoder=encoder,
        decoders=[node_decoder],
        kl_scheduler=kl_scheduler
    )
    
    if verbose:
        print("=== Starting Phase 1: Training encoder with node feature reconstruction ===")
    
    # Phase 1 training
    history_phase1 = train_phase1(
        model=model_phase1,
        data_loader=data_loader_phase1,
        num_epochs=phase1_epochs,
        lr=lr_phase1,
        weight_decay=weight_decay,
        verbose=verbose,
        device=device
    )
    
    if verbose:
        print("\n=== Phase 1 Complete ===")
        print("=== Starting Phase 2: Freezing encoder and adding adjacency decoder ===")
    
    # Step 2: Add adjacency decoder for phase 2
    adjacency_decoder = AdjacencyDecoder(
        latent_dim=latent_dim,
        hidden_dims=adj_decoder_hidden_dims,
        dropout=dropout
    )
    
    # Add adjacency decoder to model
    model_phase1.add_decoder(adjacency_decoder)
    
    # Reset KL scheduler for phase 2
    model_phase1.kl_scheduler = KLAnnealingScheduler(
        anneal_start=0.5,  # Start at midpoint since encoder is already trained
        anneal_end=1.0,
        anneal_steps=phase2_epochs * len(data_loader_phase2),
        anneal_type='sigmoid'
    )
    
    # Phase 2 training
    history_phase2 = train_phase2(
        model=model_phase1,
        data_loader=data_loader_phase2,
        num_epochs=phase2_epochs,
        lr=lr_phase2,
        weight_decay=weight_decay,
        decoder_weights=decoder_weights_phase2,
        verbose=verbose,
        device=device
    )
    
    if verbose:
        print("\n=== Phase 2 Complete ===")
    
    # Combine histories
    combined_history = {
        "phase1": history_phase1,
        "phase2": history_phase2
    }
    
    # Unfreeze encoder for potential further use
    model_phase1.set_encoder_freeze(False)
    
    return model_phase1, combined_history