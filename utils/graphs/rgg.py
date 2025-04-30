import numpy as np
import networkx as nx
from typing import Callable, Tuple

class LatentSpaceRGG:
    def __init__(
        self,
        latent_sampler: Callable[[int], np.ndarray],
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        connection_rule: str = "radius",
        threshold: float = 0.2,
        k: int = 5,
        weight_fn: Callable[[float], float] = None,
    ):
        """
        Parameters:
        - latent_sampler: n -> np.ndarray of shape (n, d)
        - distance_fn: (x, y) -> float
        - connection_rule: "radius" or "knn"
        - threshold: distance cutoff for radius connection
        - k: number of neighbors for kNN
        - weight_fn: Optional function mapping distance -> weight
        """
        self.latent_sampler = latent_sampler
        self.distance_fn = distance_fn
        self.connection_rule = connection_rule
        self.threshold = threshold
        self.k = k
        self.weight_fn = weight_fn

    def generate_graph(self, n: int) -> Tuple[nx.Graph, np.ndarray]:
        Z = self.latent_sampler(n)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        def maybe_weighted_edge(i, j, dist):
            if self.weight_fn:
                G.add_edge(i, j, weight=self.weight_fn(dist))
            else:
                G.add_edge(i, j)

        if self.connection_rule == "radius":
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.distance_fn(Z[i], Z[j])
                    if dist <= self.threshold:
                        maybe_weighted_edge(i, j, dist)

        elif self.connection_rule == "knn":
            for i in range(n):
                dists = [(j, self.distance_fn(Z[i], Z[j])) for j in range(n) if j != i]
                dists.sort(key=lambda x: x[1])
                for j, dist in dists[:self.k]:
                    maybe_weighted_edge(i, j, dist)
        else:
            raise ValueError("Invalid connection rule.")

        return G, Z

    def generate_graphs(self, num_graphs: int, n: int) -> Tuple[list, list]:
        """
        Generate multiple independent RGGs.

        Parameters:
        - num_graphs: Number of graphs to generate
        - n: Number of nodes in each graph

        Returns:
        - graphs: List of networkx.Graph objects
        - latents: List of np.ndarray arrays (each of shape (n, d))
        """
        graphs = []
        latents = []
        for _ in range(num_graphs):
            G, Z = self.generate_graph(n)
            graphs.append(G)
            latents.append(Z)
        return graphs, latents
    
def euclidean_sampler(n, d=2):
    return np.random.uniform(0, 1, size=(n, d))

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def hyperbolic_distance(x, y):
    # Use Poincaré distance, for example
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    norm_diff = np.linalg.norm(x - y)
    return np.arccosh(1 + 2 * (norm_diff ** 2) / ((1 - norm_x ** 2) * (1 - norm_y ** 2)))

def sample_poincare_ball(n: int, dim: int = 2, max_radius: float = 0.999) -> np.ndarray:
    """
    Samples n points from a hyperbolic space using the Poincaré ball model.

    Parameters:
    - n: Number of points
    - dim: Dimensionality of the ambient Euclidean space (≥2)
    - max_radius: Maximum norm of points (must be <1 to stay inside Poincaré ball)

    Returns:
    - Z: Array of shape (n, dim) with points in the Poincaré ball
    """
    # Sample radius with exponential-like distribution to reflect hyperbolic geometry
    u = np.random.uniform(0, 1, size=n)
    r = np.arccosh(1 + u * (np.cosh(max_radius) - 1))  # Invert CDF of volume function
    r = np.tanh(r / 2)  # Map to radius in Poincaré ball

    # Sample directions uniformly on the (dim-1)-sphere
    v = np.random.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # Scale by sampled radius
    Z = r[:, np.newaxis] * v
    return Z

# rgg_hyper = LatentSpaceRGG(
#     latent_sampler=lambda n: sample_poincare_ball(n, dim=2),
#     distance_fn=poincare_distance,
#     connection_rule="radius",
#     threshold=2.0  # Adjust threshold for hyperbolic scale
# )
