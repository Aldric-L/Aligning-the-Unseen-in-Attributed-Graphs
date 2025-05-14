import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

import numpy as np
from sklearn.manifold import MDS

from utils.manifolds.core import compute_gaussian_curvature, compute_geodesic

def visualize_manifold_embedding(data_points, riemannian_distances_matrix):
    """
    Embeds high-dimensional data points into 2D using Multidimensional Scaling (MDS)
    based on Euclidean and Riemannian distances and plots the results.

    Args:
        data_points (np.ndarray): Array of data points (each row is a point).
    """
    n_points = data_points.shape[0]

    # 1. Euclidean Distance and MDS
    #euclidean_distances = pdist(data_points, 'euclidean')
    #euclidean_distances_matrix = squareform(euclidean_distances)

    #mds_euclidean = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    #points_euclidean = mds_euclidean.fit_transform(euclidean_distances_matrix)
    points_euclidean = data_points

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(points_euclidean[:, 0], points_euclidean[:, 1])
    plt.title('2D Embedding (Euclidean Distances)')
    for i, point in enumerate(data_points):
        plt.text(points_euclidean[i, 0], points_euclidean[i, 1], f'Point {i+1}')

    # 2. Riemannian Distance and MDS (if metric tensor is provided)
    mds_riemannian = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    points_riemannian = mds_riemannian.fit_transform(riemannian_distances_matrix)
    #points_riemannian = TSNE(n_components=2, metric='precomputed').fit_transform(D)

    plt.subplot(1, 2, 2)
    plt.scatter(points_riemannian[:, 0], points_riemannian[:, 1])
    plt.title('2D Embedding (Riemannian Distances)')
    for i, point in enumerate(data_points):
        plt.text(points_riemannian[i, 0], points_riemannian[i, 1], f'Point {i+1}')

    plt.tight_layout()
    plt.show()


# Function to visualize the curvature
def visualize_manifold_curvature(metric_function, z_range=(-3, 3), resolution=30, data_points=None, labels=None):
    """
    Visualize the curvature of a 2D manifold in latent space with optional data points.
    
    Args:
        metric_function: Function that computes the metric tensor at a point
        z_range: Range of the latent space coordinates
        resolution: Number of points in each dimension
        data_points: Optional array containing data points on the manifold
        labels: Optional array of integer labels for the data points
    """
    # Create a grid of points in latent space
    z1 = np.linspace(z_range[0], z_range[1], resolution)
    z2 = np.linspace(z_range[0], z_range[1], resolution)
    Z1, Z2 = np.meshgrid(z1, z2)
    
    # Initialize array to store curvature values
    curvature = np.zeros((resolution, resolution))
    
    # Compute curvature at each point
    for i in range(resolution):
        for j in range(resolution):
            z = np.array([Z1[i, j], Z2[i, j]])
            try:
                G = metric_function(z)
                curvature[i, j] = compute_gaussian_curvature(G)
            except Exception as e:
                print(f"Error at grid point ({z}): {e}")
                curvature[i, j] = np.nan
    
    # Create plots
    fig = plt.figure(figsize=(18, 6))
    
    # Handle color normalization robustly
    vmin = np.nanmin(curvature)
    vmax = np.nanmax(curvature)
    
    # Make sure our color scaling works
    if np.isnan(vmin) or np.isnan(vmax):
        print("Warning: All curvature values are NaN. Check your metric function.")
        vmin, vmax = -1, 1  # Fallback values
    
    # Create a robust color normalization
    # If vmin and vmax are on the same side of zero, adjust them to include zero
    if vmin >= 0:
        vmin = 0
        norm = colors.Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        cmap = 'Reds'
    elif vmax <= 0:
        vmax = 0
        norm = colors.Normalize(vmin=min(vmin, -0.1), vmax=vmax)
        cmap = 'Blues_r'
    else:
        # We can use a diverging colormap with zero in the middle
        maxabs = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=-maxabs, vmax=maxabs)
        cmap = 'RdBu_r'
    
    # 1. 2D heatmap
    ax1 = fig.add_subplot(131)
    im = ax1.pcolormesh(Z1, Z2, curvature, cmap=cmap, norm=norm, shading='auto')
    ax1.set_title('Manifold Curvature (2D View)')
    ax1.set_xlabel('z1')
    ax1.set_ylabel('z2')
    plt.colorbar(im, ax=ax1, label='Curvature')
    
    # 2. 3D surface plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(Z1, Z2, curvature, cmap=cmap, 
                           linewidth=0, antialiased=True, norm=norm)
    ax2.set_title('Manifold Curvature (3D Surface)')
    ax2.set_xlabel('z1')
    ax2.set_ylabel('z2')
    ax2.set_zlabel('Curvature')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Curvature')
    
    # 3. Contour plot
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(Z1, Z2, curvature, 20, cmap=cmap, norm=norm)
    ax3.set_title('Manifold Curvature (Contour Plot)')
    ax3.set_xlabel('z1')
    ax3.set_ylabel('z2')
    plt.colorbar(contour, ax=ax3, label='Curvature')
    
    # Add zero contour line if it crosses zero
    if vmin < 0 and vmax > 0:
        ax3.contour(Z1, Z2, curvature, levels=[0], colors='k', linewidths=1.5)
    
    # Plot data points if provided
    if data_points is not None:
        # Verify we have labels for the data points
        if labels is None:
            print("Warning: Data points provided without labels. Using default label 0.")
            labels = np.zeros(len(data_points), dtype=int)
            
        # Check the shape of data_points and adjust if needed
        if len(data_points.shape) > 1 and data_points.shape[1] > 2:
            print(f"Warning: Data points have {data_points.shape[1]} dimensions, using only first 2 dimensions for plotting.")
            # Extract just the first two dimensions
            plot_points = data_points[:, :2]
        else:
            plot_points = data_points
            
        # Ensure points are correctly shaped for plotting
        if len(plot_points.shape) == 1:
            # If we have a single point with just a list of coordinates
            plot_points = plot_points.reshape(1, -1)

        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Create a colormap for the labels
        label_cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        
        # Scatter plot for each subplot
        for ax in [ax1, ax3]:
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(plot_points[mask, 0], plot_points[mask, 1], 
                          c=[label_cmap(i)], edgecolors='k', s=50, 
                          label=f'Class {label}', zorder=10)
            # Use automatic legend placement with small font and maximum 10 items per column
            if len(unique_labels) > 10:
                ncol = max(1, len(unique_labels) // 10)
                ax.legend(loc='best', fontsize='xx-small', ncol=ncol)
            else:
                ax.legend(loc='best', fontsize='small')
        
        # For 3D plot, we need to compute the curvature at each data point
        # and use that for z-coordinate
        data_curvatures = np.zeros(len(plot_points))
        for i, point in enumerate(plot_points):
            try:
                G = metric_function(point)
                data_curvatures[i] = compute_gaussian_curvature(G)
            except Exception as e:
                print(f"Error computing curvature for data point {i}: {e}")
                data_curvatures[i] = np.nan
        
        # Plot on 3D surface
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax2.scatter(plot_points[mask, 0], plot_points[mask, 1], 
                       data_curvatures[mask], c=[label_cmap(i)], 
                       edgecolors='k', s=50, label=f'Class {label}', zorder=10)
        # Use automatic legend placement with small font and maximum 10 items per column
        if len(unique_labels) > 10:
            ncol = max(1, len(unique_labels) // 10)
            ax2.legend(loc='best', fontsize='xx-small', ncol=ncol)
        else:
            ax2.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.show()
    
    return curvature, Z1, Z2

# Plot geodesics to show path avoidance around high curvature regions
def plot_geodesics(metric_function, start_point, end_point, curvature, Z1, Z2):
    """
    Plot geodesics between points to show path avoidance around high curvature regions.
    
    Args:
        metric_function: Function that computes the metric tensor at a point
        start_point: Starting point
        end_point: Ending point
        curvature: Array of curvature values from visualize_manifold_curvature
        Z1, Z2: Meshgrid arrays from visualize_manifold_curvature
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle color normalization robustly
    vmin = np.nanmin(curvature)
    vmax = np.nanmax(curvature)
    
    # Create a robust color normalization
    if vmin >= 0:
        vmin = 0
        norm = colors.Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        cmap = 'Reds'
    elif vmax <= 0:
        vmax = 0
        norm = colors.Normalize(vmin=min(vmin, -0.1), vmax=vmax)
        cmap = 'Blues_r'
    else:
        maxabs = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=-maxabs, vmax=maxabs)
        cmap = 'RdBu_r'
    
    # Plot curvature heatmap
    im = ax.pcolormesh(Z1, Z2, curvature, cmap=cmap, norm=norm, shading='auto', alpha=0.7)
    
    # Plot straight line (Euclidean path)
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
            'k--', linewidth=1.5, label='Euclidean Path')
    
    # Compute and plot true geodesic
    geodesic_path = compute_geodesic(metric_function, start_point, end_point, num_points=50)
    ax.plot(geodesic_path[:, 0], geodesic_path[:, 1], 'r-', linewidth=2, label='Geodesic Path')
    
    # Mark start and end points
    ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start')
    ax.plot(end_point[0], end_point[1], 'bo', markersize=8, label='End')
    
    ax.set_title('Geodesic vs Euclidean Path')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    plt.colorbar(im, label='Curvature')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Plot multiple geodesics to show the overall geodesic structure
def plot_geodesic_grid(metric_function, curvature, Z1, Z2, grid_size=3):
    """
    Plot a grid of geodesics to visualize the overall geodesic structure.
    
    Args:
        metric_function: Function that computes the metric tensor at a point
        curvature: Array of curvature values
        Z1, Z2: Meshgrid arrays
        grid_size: Number of points in each dimension of the start/end grid
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up color normalization for the curvature plot
    vmin = np.nanmin(curvature)
    vmax = np.nanmax(curvature)
    
    if vmin >= 0:
        vmin = 0
        norm = colors.Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        cmap = 'Reds'
    elif vmax <= 0:
        vmax = 0
        norm = colors.Normalize(vmin=min(vmin, -0.1), vmax=vmax)
        cmap = 'Blues_r'
    else:
        maxabs = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=-maxabs, vmax=maxabs)
        cmap = 'RdBu_r'
    
    # Plot curvature heatmap
    im = ax.pcolormesh(Z1, Z2, curvature, cmap=cmap, norm=norm, shading='auto', alpha=0.8)
    
    # Create a grid of start points
    x_min, x_max = np.min(Z1)/1.2, np.max(Z1)/1.2
    y_min, y_max = np.min(Z2)/1.2, np.max(Z2)/1.2
    
    x_points = np.linspace(x_min, x_max, grid_size)
    y_points = np.linspace(y_min, y_max, grid_size)
    
    # Plot geodesics from each start point to each end point
    for i, x_start in enumerate(x_points):
        for j, y_start in enumerate(y_points):
            start_point = np.array([x_start, y_start])
            
            # Plot geodesics in four directions (right, up-right, up, up-left)
            directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
            
            for dx, dy in directions:
                # Scale the direction to ensure it stays within bounds
                scale = min(
                    0.5,
                    (x_max - x_start) / max(0.001, abs(dx)) if dx > 0 else float('inf'),
                    (x_start - x_min) / max(0.001, abs(dx)) if dx < 0 else float('inf'),
                    (y_max - y_start) / max(0.001, abs(dy)) if dy > 0 else float('inf'),
                    (y_start - y_min) / max(0.001, abs(dy)) if dy < 0 else float('inf')
                )
                
                # Compute end point
                end_point = start_point + scale * np.array([dx, dy])
                
                try:
                    # Compute geodesic (with fewer points for cleaner visualization)
                    geodesic_path = compute_geodesic(metric_function, start_point, end_point, num_points=20)
                    
                    # Plot the geodesic
                    ax.plot(geodesic_path[:, 0], geodesic_path[:, 1], 'b-', linewidth=1, alpha=0.6)
                    
                    # Mark start points
                    if i == 0 and j == 0:  # Only add to legend once
                        ax.plot(start_point[0], start_point[1], 'ko', markersize=4, label='Grid Points')
                    else:
                        ax.plot(start_point[0], start_point[1], 'ko', markersize=4)
                except Exception as e:
                    print(f"Error computing geodesic from {start_point} to {end_point}: {e}")
    
    ax.set_title('Geodesic Grid Structure')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    plt.colorbar(im, label='Curvature')
    ax.legend()
    plt.tight_layout()
    plt.show()