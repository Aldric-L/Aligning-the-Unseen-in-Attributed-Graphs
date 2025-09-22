import numpy as np
import torch
from sklearn.cluster import KMeans

class RBFLayer(torch.nn.Module):
    """
    Radial Basis Function layer for precision estimation (Arvanitidis style)
    """
    def __init__(self, latent_dim, data_dim, n_centers, a=1, min_precision=1e-6):
        super(RBFLayer, self).__init__()
        
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_centers = n_centers
        self.a = a
        
        # Centers in latent space (set by KMeans)
        self.centers = torch.nn.Parameter(torch.randn(n_centers, latent_dim), requires_grad=False)
        
        # Bandwidths (scalar per center, isotropic)
        self.lambdas = torch.nn.Parameter(torch.ones(n_centers, 1), requires_grad=False)
        
        # Learnable weights: (n_centers, data_dim)
        self.weights = torch.nn.Parameter(torch.rand(n_centers, data_dim), requires_grad=True)
        
        # Floor constant
        self.zeta = torch.nn.Parameter(torch.tensor(min_precision), requires_grad=False)
        
    def fit_centers(self, latent_samples: torch.Tensor):
        """
        Fit RBF centers with KMeans and initialize bandwidths
        following Arvanitidis (min diagonal covariance heuristic).
        """
        latent_np = latent_samples.detach().cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.n_centers, n_init=30, max_iter=1000, random_state=42)
        labels = kmeans.fit_predict(latent_np)
        centers = kmeans.cluster_centers_
        
        self.centers.data = torch.tensor(centers, dtype=torch.float32)
        
        # Bandwidth initialization
        sigmas = np.zeros((self.n_centers, 1))
        for k in range(self.n_centers):
            pts = latent_np[labels == k]
            if pts.shape[0] > 0:
                # covariance diagonal
                cov_diag = np.var(pts - centers[k], axis=0)
                sigma_k = np.sqrt(np.min(cov_diag))
                sigmas[k, 0] = sigma_k
            else:
                sigmas[k, 0] = 1.0

        # Global smoothing factor
        sigmas *= 1.25  

        # Avoid division by zero
        sigmas = np.maximum(sigmas, 1e-6)

        lambdas = 0.5 * self.a / (sigmas ** 2)
        self.lambdas.data = torch.tensor(lambdas, dtype=torch.float32)
        
    def forward(self, z):
        """
        Forward pass: predict precision vector in data space.
        z: (batch, latent_dim)
        Returns: (batch, data_dim)
        """
        # pairwise squared distances: (batch, n_centers)
        dists = torch.cdist(z, self.centers, p=2) ** 2
        
        # RBF activations: exp(-λ_k * ||z - c_k||^2)
        activations = torch.exp(-dists * self.lambdas.T)  # (batch, n_centers)
        
        # Precision field in data space
        precision = activations @ self.weights + self.zeta
        precision = torch.clamp(precision, min=1e-8)  # avoid degenerate precisions
        return precision


class PrecisionNetwork(torch.nn.Module):
    """Network for estimating precision (1/variance) in the latent space"""
    def __init__(self, latent_dim, data_dim, n_centers=10, a=1.0, min_precision=1e-6, dropout=0.1):
        super(PrecisionNetwork, self).__init__()
        self.rbf_layer = RBFLayer(latent_dim=latent_dim,
                                  data_dim=data_dim,
                                  n_centers=n_centers,
                                  min_precision=min_precision,
                                  a=a)
        
        self.dropout = torch.nn.Dropout(p=dropout) 
        
    def fit(self, latent_samples):
        """Fit the RBF centers using latent samples"""
        self.rbf_layer.fit_centers(latent_samples)
        
    def forward(self, z):
        """Forward pass to compute precision (1/variance) for latent points"""
        return self.dropout(self.rbf_layer(z))
    
    def get_variance(self, z):
        """Compute variance (1/precision) for latent points"""
        precision = self.forward(z)
        # Add small epsilon to avoid division by zero
        return 1.0 / (precision + 1e-10)
    
    def compute_precision_jacobian(self, z: torch.Tensor, inverse: bool = True):
        """
        Compute Jacobian of precision network with respect to latent input z.
        
        Args:
            precision_network: PrecisionNetwork instance
            z: latent points, shape (batch_size, latent_dim)
        
        Returns:
            jacobian: shape (batch_size, data_dim, latent_dim)
                    jacobian[i, j, k] = ∂precision[i,j] / ∂z[i,k]
        """
        self.eval() 
        batch_size, latent_dim = z.shape
        
        # Enable gradient computation
        z_grad = z.clone().detach().requires_grad_(True)
        
        # Forward pass
        precision = self.forward(z_grad)
        #precision = 1/ (precision + 1e-8)
        precision =  -0.5 * torch.log(precision + 1e-8)
        
        # Compute Jacobian
        jacobian = torch.zeros(batch_size, self.rbf_layer.data_dim, latent_dim)
        
        for i in range(self.rbf_layer.data_dim):
            # Compute gradients for i-th output dimension
            grad_outputs = torch.zeros_like(precision)
            grad_outputs[:, i] = 1.0
            
            grad = torch.autograd.grad(
                outputs=precision,
                inputs=z_grad,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=True
            )[0]
            
            jacobian[:, i, :] = grad
        
        return jacobian
    

    def compute_precision_jacobian_manual(self, z: torch.Tensor, inverse: bool = True):
        """
        Compute Jacobian of precision network with respect to latent input z.
        Manual computation for RBF network - faster than autograd.
        
        Args:
            precision_network: PrecisionNetwork instance
            z: latent points, shape (batch_size, latent_dim)
            inverse: compute jacobian with respect to the variance, not the precision
        
        Returns:
            jacobian: shape (batch_size, data_dim, latent_dim)
                    jacobian[i, j, k] = ∂precision[i,j] / ∂z[i,k]
        """
        self.eval() 
        batch_size, latent_dim = z.shape
        
        # Compute distances and activations
        # z: (batch, latent_dim), centers: (n_centers, latent_dim)
        diff = z.unsqueeze(1) - self.rbf_layer.centers.unsqueeze(0)  # (batch, n_centers, latent_dim)
        dists_sq = torch.sum(diff**2, dim=-1)  # (batch, n_centers)
        activations = torch.exp(-dists_sq * self.rbf_layer.lambdas.squeeze(-1))  # (batch, n_centers)
        
        # Jacobian computation
        # ∂activation_k/∂z_j = -2λ_k * (z_j - c_k,j) * activation_k
        jacobian = torch.zeros(batch_size, self.rbf_layer.data_dim, latent_dim)
        
        for j in range(latent_dim):
            # Derivative of activations w.r.t. z_j: (batch, n_centers)
            dactivation_dzj = -0.5 * self.rbf_layer.lambdas.squeeze(-1) * diff[:, :, j] * activations
            
            # Chain rule with weights: (batch, n_centers) @ (n_centers, data_dim)
            jacobian[:, :, j] = dactivation_dzj @ self.rbf_layer.weights  # (batch, data_dim)
        
        return jacobian


    def compute_precision_jacobian_vectorized(self, z, inverse=True):
        """
        More efficient vectorized version using torch.func.jacrev (if available)
        """
        self.eval() 
        try:
            import torch.func as func
            
            def precision_fn(z_single):
                if inverse:
                    #return 1/(self.forward(z_single.unsqueeze(0)).squeeze(0) + 1e-8)
                    return -0.5 * torch.log(self.forward(z_single.unsqueeze(0)).squeeze(0) + 1e-8)
                return self.forward(z_single.unsqueeze(0)).squeeze(0)
            
            # Vectorized Jacobian computation
            jacobian_fn = func.vmap(func.jacrev(precision_fn))
            return jacobian_fn(z)
        
        except ImportError:
            # Fallback to manual computation
            return self.compute_precision_jacobian(z)

