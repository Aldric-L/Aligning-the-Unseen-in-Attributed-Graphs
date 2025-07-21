import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union


class Encoder(nn.Module):
    """
    Flexible graph encoder that combines GCN layers with fully connected layers.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        latent_dim: int,
        gcn_layers: int = 2,
        fc_layers: int = 1,
        dropout: float = 0.1,
        activation=nn.ReLU()
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden dimensions for GCN and FC layers
            latent_dim: Dimension of the latent space
            gcn_layers: Number of GCN layers
            fc_layers: Number of fully connected layers after GCN
            dropout: Dropout probability
            activation: Activation function to use
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.gcn_layers = gcn_layers
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.activation = activation
        
        # GCN layers
        self.gcn = nn.ModuleList()
        in_channels = input_dim
        for i in range(gcn_layers):
            out_channels = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            self.gcn.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels
        
        # Fully connected layers
        self.fc = nn.ModuleList()
        in_features = in_channels
        for i in range(fc_layers):
            idx = i + gcn_layers
            out_features = hidden_dims[idx] if idx < len(hidden_dims) else hidden_dims[-1]
            self.fc.append(nn.Linear(in_features, out_features))
            in_features = out_features
        
        # Mean and log variance projections
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the encoder
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        # GCN layers
        for i, gcn_layer in enumerate(self.gcn):
            x = gcn_layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fully connected layers
        for i, fc_layer in enumerate(self.fc):
            x = fc_layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Latent projections
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu.requires_grad_(True), logvar.requires_grad_(True)

class MLPEncoder(nn.Module):
    """
    Fully MLP-based encoder that processes node features without using graph structure.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        mlp_layers: int = 3,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        mu_activation=None,
        variance_dimension = None
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden dimensions for MLP layers
            latent_dim: Dimension of the latent space
            mlp_layers: Number of MLP layers
            dropout: Dropout probability
            activation: Activation function to use
        """
        super(MLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.activation = activation
        self.mu_activation = mu_activation
        self.variance_dimension = variance_dimension if variance_dimension is not None else latent_dim 

        # MLP layers
        self.mlp = nn.ModuleList()
        in_features = input_dim
        
        for i in range(mlp_layers):
            out_features = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            layer = nn.Linear(in_features, out_features)
            
            # Apply Glorot (Xavier) initialization to the weights of this layer
            # You can choose xavier_uniform_ or xavier_normal_
            nn.init.xavier_uniform_(layer.weight)
            # It's common to initialize biases to zero
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
                
            self.mlp.append(layer)
            in_features = out_features
        
        # Mean and log variance projections
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.variance_dimension)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        if self.fc_mu.bias is not None:
            nn.init.constant_(self.fc_mu.bias, 0)

        nn.init.xavier_uniform_(self.fc_logvar.weight)
        if self.fc_logvar.bias is not None:
            nn.init.constant_(self.fc_logvar.bias, 0)
        
    def forward(self, x, edge_index=None):
        """
        Forward pass through the encoder
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Not used in this encoder, included for API compatibility
        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        # MLP layers
        for i, mlp_layer in enumerate(self.mlp):
            x = mlp_layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Latent projections
        mu = self.fc_mu(x)
        if self.mu_activation is not None:
            mu = self.mu_activation(mu)
        logvar = self.fc_logvar(x)
        
        return mu, logvar