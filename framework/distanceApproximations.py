import torch
from typing import List, Callable
import numpy as np

from framework.boundedManifold import BoundedManifold

class DistanceApproximations:
    """
    A class containing various methods to approximate the distance between two points
    on a Riemannian manifold defined by a BoundedManifold instance.
    All calculations are performed using PyTorch for autograd compatibility.
    """

    @staticmethod
    def linear_interpolation_distance(manifold: BoundedManifold, u: torch.Tensor, v: torch.Tensor, num_points: int = 20) -> torch.Tensor:
        """
        Compute distance by integrating along the straight line path in the ambient space.
        Formula: ∫₀¹ √((v-u)ᵀg(x(t))(v-u))dt where x(t) = (1-t)u + t*v.

        Args:
            manifold (BoundedManifold): An instance of the BoundedManifold class.
            u (torch.Tensor): The starting point.
            v (torch.Tensor): The ending point.
            num_points (int): Number of integration points along the linear path.

        Returns:
            torch.Tensor: The approximate distance along the linearly interpolated path.
        """
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("Points u and v must be 1D tensors of the same shape.")

        t_vals = torch.linspace(0, 1, num_points, device=manifold.device)
        diff = v - u
        
        # Evaluate the length element at each point along the path
        integrand_values = []
        for t in t_vals:
            x_t = (1 - t) * u + t * v
            clamped_x_t = manifold._clamp_point_to_bounds(x_t)
            try:
                g = manifold.metric_tensor(clamped_x_t)
                # Length element: sqrt( (dx/dt)^T * G * (dx/dt) )
                # Since x(t) is linear, dx/dt = v - u = diff
                segment_length_sq = diff @ g @ diff
                integrand_values.append(torch.sqrt(torch.relu(segment_length_sq) + 1e-12))
            except (ValueError, RuntimeError):
                # Fallback for points where metric is not defined
                integrand_values.append(torch.linalg.norm(diff))

        # Use trapezoidal rule for numerical integration
        integral = torch.trapz(torch.stack(integrand_values), x=t_vals)
        return integral
    
    @staticmethod
    def linear_interpolation_distance(manifold: BoundedManifold,
            u: torch.Tensor,
            v: torch.Tensor,
            num_points: int = 50,
        ) -> torch.Tensor:
        """
        Per-pair linear interpolation distance fully in float precision,
        now batched over inputs of shape (D,) or (B, D).
        Returns a scalar if inputs are (D,), or a tensor of shape (B,) if inputs are (B, D).
        """
        # ensure float
        u = u.float()
        v = v.float()

        # make sure we have a batch dimension
        single = (u.dim() == 1)
        if single:
            # from (D,) to (1, D)
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)

        # now u, v are (B, D)
        B, D = u.shape

        # time samples
        t_vals = torch.linspace(0.0, 1.0, num_points, device=manifold.device, dtype=torch.float)
        T = num_points
        # shape (T, 1, 1) for broadcasting over (B, D)
        t = t_vals.view(T, 1, 1)

        # build all interpolation points: shape (T, B, D)
        # u.unsqueeze(0) is (1, B, D), same for v
        X = (1.0 - t) * u.unsqueeze(0) + t * v.unsqueeze(0)
        # clamp them, but _clamp_point_to_bounds wants (..., D)
        X_flat = X.view(-1, D)  # (T*B, D)
        Xc_flat = manifold._clamp_point_to_bounds(X_flat)
        Xc = Xc_flat.view(T, B, D)

        # compute metric tensor at each of the T*B points in one go:
        # self.metric_tensor(Xc_flat, True) -> (T*B, D, D)
        G_flat = manifold.metric_tensor(Xc_flat, True)
        G = G_flat.view(T, B, D, D)  # (T, B, D, D)

        # difference v - u: (B, D)
        diff = (v - u)
        # replicate over T: (T, B, D)
        diff_rep = diff.unsqueeze(0).expand(T, B, D)

        # squared speed at each (t, b)
        # seg_sq[t,b] = diff_rep[t,b] @ G[t,b] @ diff_rep[t,b]
        seg_sq = torch.einsum("tbi,tbij,tbj->tb", diff_rep, G, diff_rep)

        # speed = sqrt(seg_sq)
        integrand = torch.sqrt(torch.relu(seg_sq) + 1e-12)  # (T, B)

        # trapezoidal integration over t: returns shape (B,)
        dists = torch.trapz(integrand, x=t_vals, dim=0)  # (B,)

        # if original inputs were single points, return scalar
        if single:
            return dists.squeeze(0)
        return dists

    @staticmethod
    def midpoint_approximation(manifold: BoundedManifold, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Approximates distance using the metric at the midpoint of the straight line path.
        Formula: √((v-u)ᵀg((u+v)/2)(v-u))

        Args:
            manifold (BoundedManifold): An instance of the BoundedManifold class.
            u (torch.Tensor): The starting point.
            v (torch.Tensor): The ending point.

        Returns:
            torch.Tensor: The approximate distance.
        """
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("Points u and v must be 1D tensors of the same shape.")

        midpoint = (u + v) / 2
        clamped_midpoint = manifold._clamp_point_to_bounds(midpoint)
        diff = v - u
        
        try:
            g = manifold.metric_tensor(clamped_midpoint)
            quadratic_form_result = diff @ g @ diff
            return torch.sqrt(torch.relu(quadratic_form_result) + 1e-12)
        except (ValueError, RuntimeError):
            return torch.linalg.norm(diff)

    @staticmethod
    def weighted_midpoint_approximation(manifold: BoundedManifold, u: torch.Tensor, v: torch.Tensor, weights: List[float] = None) -> torch.Tensor:
        """
        Approximates distance using a weighted average of length elements at multiple points.
        Default weights correspond to Simpson's rule for integrating the length element.

        Args:
            manifold (BoundedManifold): An instance of the BoundedManifold class.
            u (torch.Tensor): The starting point.
            v (torch.Tensor): The ending point.
            weights (List[float], optional): Weights for each point. Defaults to Simpson's rule [1/6, 4/6, 1/6].

        Returns:
            torch.Tensor: The approximate distance.
        """
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("Points u and v must be 1D tensors of the same shape.")

        if weights is None:
            weights = [1/6, 4/6, 1/6]
        
        points = [u, (u + v) / 2, v]
        
        if len(points) != len(weights):
            raise ValueError("Number of points must match the number of weights.")

        total_weighted_sum = torch.tensor(0.0, device=manifold.device)
        diff = v - u

        for point, weight in zip(points, weights):
            clamped_point = manifold._clamp_point_to_bounds(point)
            try:
                g = manifold.metric_tensor(clamped_point)
                length_element_sq = diff @ g @ diff
                length_element = torch.sqrt(torch.relu(length_element_sq) + 1e-12)
                total_weighted_sum += weight * length_element
            except (ValueError, RuntimeError):
                total_weighted_sum += weight * torch.linalg.norm(diff)
        
        return total_weighted_sum

    @staticmethod
    def endpoint_average_approximation(manifold: BoundedManifold, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Approximates distance by averaging the metric at the two endpoints.
        Formula: √((v-u)ᵀ((g(u)+g(v))/2)(v-u))

        Args:
            manifold (BoundedManifold): An instance of the BoundedManifold class.
            u (torch.Tensor): The starting point.
            v (torch.Tensor): The ending point.

        Returns:
            torch.Tensor: The approximate distance.
        """
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("Points u and v must be 1D tensors of the same shape.")

        diff = v - u
        try:
            g_u = manifold.metric_tensor(manifold._clamp_point_to_bounds(u))
            g_v = manifold.metric_tensor(manifold._clamp_point_to_bounds(v))
            g_avg = (g_u + g_v) / 2
            
            quadratic_form_result = diff @ g_avg @ diff
            return torch.sqrt(torch.relu(quadratic_form_result) + 1e-12)
        except (ValueError, RuntimeError):
            return torch.linalg.norm(diff)

    @staticmethod
    def euclidean_distance(manifold: BoundedManifold, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the standard Euclidean distance.

        Args:
            manifold (BoundedManifold): An instance of the BoundedManifold class. (Added for consistency, though not strictly used in calculation)
            u (torch.Tensor): The first point.
            v (torch.Tensor): The second point.

        Returns:
            torch.Tensor: The Euclidean distance between u and v.
        """
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError("Points u and v must be 1D tensors of the same shape.")
            
        return torch.linalg.norm(u - v)