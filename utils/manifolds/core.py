import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import squareform
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.linalg import det, inv, eigh

import torch
from torch.autograd.functional import jacobian

def compute_decoder_jacobian(model, z_point):
    """Compute the Jacobian of the decoder at a specific point in latent space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.tensor(z_point, dtype=torch.float32, requires_grad=True).to(device)
    
    # Define a function that takes z and returns the decoded output
    def decoder_function(z_input):
        return model.decode(z_input)
    
    # Compute Jacobian
    J = jacobian(decoder_function, z)
    
    # The Jacobian shape is (batch_size, output_dim, input_dim)
    return J.detach().cpu().numpy()

def compute_metric_tensor(jacobian, input_dim, latent_dim):
    """Compute the Riemannian metric tensor G = J^T J."""
    # Reshape jacobian: (input_dim, latent_dim)
    J = jacobian.reshape(input_dim, latent_dim)
    # Compute metric tensor: G = J^T J (latent_dim, latent_dim)
    G = np.matmul(J.T, J)
    return G

def decoder_point_metric(model, z):
    """
    Wrapper to compute the Riemannian metric from the jacobian matrix
    z: numpy array with shape (n,)
    
    Returns an (n x n) metric tensor.
    """
    l_dim = z.shape[0]
    G = None
    try:
        J = compute_decoder_jacobian(model, z)
        i_dim = J.shape[2]
        G = compute_metric_tensor(J, i_dim, l_dim)
                
    except Exception as e:
        print(f"Error at grid point ({z}): {e}")
    return G

def numerical_derivative(metric, z, i, h=1e-6):
    """
    Compute the numerical derivative of the metric function with respect to the i-th coordinate.
    metric: function returning metric tensor given z.
    z: point at which to compute the derivative.
    i: index with respect to which to differentiate.
    h: small step size.
    
    Returns: the derivative of the metric tensor along coordinate i.
    """
    z_forward = np.array(z, copy=True)
    z_backward = np.array(z, copy=True)
    z_forward[i] += h
    z_backward[i] -= h
    g_forward = metric(z_forward)
    g_backward = metric(z_backward)
    return (g_forward - g_backward) / (2*h)

def compute_christoffel(metric, z, h=1e-6):
    """
    Compute the Christoffel symbols at point z.
    
    Returns a 3-dimensional array Gamma of shape (n, n, n) where
    Gamma[k, i, j] corresponds to \Gamma^k_{ij}.
    """
    g = metric(z)
    n = len(z)
    # Invert the metric
    g_inv = np.linalg.inv(g)
    # Initialize derivative array: dg[i] is the derivative of g with respect to z[i]
    dg = [numerical_derivative(metric, z, i, h) for i in range(n)]
    
    Gamma = np.zeros((n, n, n))
    # Use the formula:
    # Gamma^k_{ij} = 1/2 * g^{k\ell} ( ∂_i g_{jℓ} + ∂_j g_{iℓ} - ∂_ℓ g_{ij} )
    for k in range(n):
        for i in range(n):
            for j in range(n):
                sum_term = 0.0
                for l in range(n):
                    term = (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                    sum_term += g_inv[k, l] * term
                Gamma[k, i, j] = 0.5 * sum_term
    return Gamma

# Function to estimate the Christoffel symbols numerically (another implementation...)
# def estimate_christoffel_symbols(metric_function, point, h=1e-5):
#     """
#     Estimate Christoffel symbols using numerical differentiation.
    
#     Args:
#         metric_function: Function that takes a point and returns the metric tensor
#         point: Point at which to compute the symbols
#         h: Step size for numerical differentiation
    
#     Returns:
#         Christoffel symbols (2x2x2 tensor)
#     """
#     dim = len(point)
#     gamma = np.zeros((dim, dim, dim))
    
#     # Get metric at the central point
#     g_center = metric_function(point)
#     g_inv = inv(g_center)
    
#     # Compute partial derivatives of the metric tensor
#     dg_dx = np.zeros((dim, dim, dim))
    
#     for k in range(dim):
#         # Create point with +h in kth dimension
#         point_plus = point.copy()
#         point_plus[k] += h
#         g_plus = metric_function(point_plus)
        
#         # Create point with -h in kth dimension
#         point_minus = point.copy()
#         point_minus[k] -= h
#         g_minus = metric_function(point_minus)
        
#         # Central difference for derivative
#         dg_dx[k] = (g_plus - g_minus) / (2 * h)
    
#     # Compute Christoffel symbols
#     for i in range(dim):
#         for j in range(dim):
#             for k in range(dim):
#                 for l in range(dim):
#                     # Christoffel symbols (first kind)
#                     gamma[i, j, k] += 0.5 * g_inv[i, l] * (
#                         dg_dx[j, l, k] + dg_dx[k, l, j] - dg_dx[l, j, k]
#                     )
    
#     return gamma

# Function to define the geodesic equation for numerical integration
def geodesic_equation(t, y, metric_function):
    """
    Define the geodesic equation for numerical integration.
    
    Args:
        t: Time parameter (not used explicitly but required by solve_ivp)
        y: State vector [position, velocity]
        metric_function: Function that computes the metric tensor at a point
    
    Returns:
        Derivatives [velocity, acceleration]
    """
    dim = len(y) // 2
    position = y[:dim]
    velocity = y[dim:]
    
    # Calculate Christoffel symbols at current position
    gamma = compute_christoffel(metric_function, position)
    
    # Calculate acceleration using the geodesic equation
    acceleration = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                acceleration[i] -= gamma[i, j, k] * velocity[j] * velocity[k]
    
    return np.concatenate([velocity, acceleration])


def geodesic_distance(z0, z1, metric, t_span=(0, 1), num_points=100):
    """
    Approximates the Riemannian distance between z0 and z1 by solving a shooting problem.
    
    z0, z1: numpy arrays of shape (n,)
    metric: function that returns the metric tensor g(z)
    Returns: approximate Riemannian distance
    """
    n = len(z0)
    t_eval = np.linspace(*t_span, num_points)

    def local_geodesic_solve(t,y):
        return geodesic_equation(t, y, metric)

    def shooting_loss(v0):
        y0 = np.concatenate((z0, v0))
        sol = solve_ivp(local_geodesic_solve, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
        z_traj = sol.y[:n, :]
        z_end = z_traj[:, -1]
        return np.sum((z_end - z1) ** 2)  # squared distance between endpoint and z1

    # Initial guess: straight line
    v_guess = z1 - z0

    result = minimize(shooting_loss, v_guess, method='BFGS', options={'disp': False})
    v_opt = result.x

    # Integrate geodesic with optimal v
    y0 = np.concatenate((z0, v_opt))
    sol = solve_ivp(local_geodesic_solve, t_span, y0, t_eval=t_eval, method='RK23', rtol=1e-3, atol=1e-5)
    z_traj = sol.y[:n, :]
    v_traj = sol.y[n:, :]

    # Compute Riemannian length along the geodesic
    length = 0.0
    for i in range(1, num_points):
        z_i = z_traj[:, i]
        v_i = v_traj[:, i]
        g = metric(z_i)
        ds2 = v_i.T @ g @ v_i
        dt = t_eval[i] - t_eval[i-1]
        length += np.sqrt(ds2) * dt

    return length


# Function to compute a geodesic between two points
def compute_geodesic(metric_function, start_point, end_point, num_points=50):
    """
    Compute a geodesic between two points by numerically solving the geodesic equation.
    
    Args:
        metric_function: Function that computes the metric tensor at a point
        start_point: Starting point
        end_point: Ending point
        num_points: Number of points in the resulting geodesic
    
    Returns:
        Array of points along the geodesic
    """
    dim = len(start_point)
    
    # Initial velocity is in the direction of end_point - start_point
    # Normalized to unit length in the metric at start_point
    initial_direction = end_point - start_point
    g_start = metric_function(start_point)
    
    # Compute metric-normalized velocity
    speed = np.sqrt(initial_direction @ g_start @ initial_direction)
    if speed > 0:
        initial_velocity = initial_direction / speed
    else:
        # Default to Euclidean direction if points are nearly coincident
        initial_velocity = initial_direction
    
    # Initial state: [position, velocity]
    initial_state = np.concatenate([start_point, initial_velocity])
    
    # Estimate the integration time based on distance
    euclidean_distance = np.linalg.norm(end_point - start_point)
    integration_time = 2.0 * euclidean_distance  # Adjust as needed
    
    # Solve the geodesic equation
    t_eval = np.linspace(0, integration_time, num_points)
    solution = solve_ivp(
        lambda t, y: geodesic_equation(t, y, metric_function),
        [0, integration_time],
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )
    
    # Extract positions from the solution
    geodesic_path = solution.y[:dim, :].T
    
    # Ensure the path ends near the desired end point
    # If it doesn't reach close enough, we need adjustments
    final_distance = np.linalg.norm(geodesic_path[-1] - end_point)
    
    if final_distance > 0.2 * euclidean_distance:
        print(f"Warning: Geodesic calculation may not be accurate. Final distance: {final_distance}")
        
        # Fall back to a simple interpolation that bends away from high curvature
        # This is a heuristic approach when numerical integration fails
        t = np.linspace(0, 1, num_points)
        geodesic_path = np.zeros((num_points, dim))
        
        # Find a suitable control point by sampling curvature in a grid between points
        grid_size = 5
        midpoint = (start_point + end_point) / 2
        span = max(0.5, euclidean_distance / 4)
        
        # Create a small grid around the midpoint
        x = np.linspace(midpoint[0] - span, midpoint[0] + span, grid_size)
        y = np.linspace(midpoint[1] - span, midpoint[1] + span, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate curvature at each grid point
        grid_curvature = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([X[i, j], Y[i, j]])
                try:
                    g = metric_function(point)
                    grid_curvature[i, j] = compute_gaussian_curvature(g)
                except:
                    grid_curvature[i, j] = 0
        
        # Find lowest curvature point as control point
        min_idx = np.argmin(np.abs(grid_curvature))
        i, j = np.unravel_index(min_idx, grid_curvature.shape)
        control_point = np.array([X[i, j], Y[i, j]])
        
        # Add slight offset perpendicular to the direct path
        path_vector = end_point - start_point
        perpendicular = np.array([-path_vector[1], path_vector[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-10)
        control_point += perpendicular * (euclidean_distance * 0.2)
        
        # Create a simple quadratic Bezier curve
        for i, ti in enumerate(t):
            geodesic_path[i] = (1-ti)**2 * start_point + 2*(1-ti)*ti * control_point + ti**2 * end_point
    
    return geodesic_path

def create_riemannian_distance_matrix(data_points, metric):
    """
    Creates a pairwise Riemannian distance matrix for the given data points with a more uniform loading bar.
    Assumes a constant metric tensor.

    Args:
        data_points (np.ndarray): Array of data points (each row is a point).
        metric: the riemannian metric function or tensor.

    Returns:
        np.ndarray: The pairwise Riemannian distance matrix.
    """
    n_points = data_points.shape[0]
    dist_matrix = np.zeros((n_points, n_points))
    total_calculations = n_points * (n_points - 1) // 2
    with tqdm(total=total_calculations, desc="Calculating distances") as pbar:
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = geodesic_distance(data_points[i], data_points[j], metric)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                pbar.update(1)
    return dist_matrix

# Function to compute Gaussian curvature from metric tensor
def compute_gaussian_curvature(G):
    """
    Compute the Gaussian curvature from a 2x2 metric tensor.
    
    G: 2x2 metric tensor
    Returns: Gaussian curvature K
    """
    # For demonstration purposes, use the determinant of the metric tensor 
    # as a proxy for distortion in the space
    det_g = det(G)
    if det_g > 0:
        # The magnitude of det_g indicates how much the space is distorted
        # Taking log to handle wide range of values
        return np.log1p(det_g)
    else:
        # Handle the case of degenerate metric
        return -1  # Less extreme negative value
