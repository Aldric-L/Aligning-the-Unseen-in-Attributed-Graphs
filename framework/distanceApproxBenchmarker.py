import numpy as np
import time
from typing import List, Tuple, Callable # Callable is kept for type hinting other functions if needed, though not directly for BoundedManifold
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from framework.boundedManifold import BoundedManifold

class DistanceApproxBenchmarker:
    """
    Experiment class for comparing geodesic distance approximations on 2D Riemannian manifolds.
    It takes an instance of the BoundedManifold class as input and leverages its capabilities.
    """
    
    def __init__(self, manifold_instance: BoundedManifold, manifold_name: str = "Custom"):
        """
        Initialize the experiment with a BoundedManifold instance.
        
        Args:
            manifold_instance (BoundedManifold): An instance of the BoundedManifold class.
                                                 It must have caching enabled and its grid populated.
            manifold_name (str): Name of the manifold for plotting titles.
        
        Raises:
            ValueError: If the manifold_instance does not have caching enabled.
        """
        if not manifold_instance.cache_enabled:
            raise ValueError("The provided BoundedManifold instance must have 'cache' activated (cache=True) "
                             "and its grid fully computed using manifold_instance.compute_full_grid_metric_tensor().")
        if manifold_instance.get_dimension() != 2:
            raise ValueError("RiemannianManifoldExperiment is currently designed for 2D manifolds only.")

        self.manifold_instance = manifold_instance
        self.manifold_name = manifold_name
        
    def run_experiment(self, test_points: np.ndarray, pairs_to_test: List[Tuple[int, int]] = None, 
                      visualize: bool = True) -> dict:
        """
        Run the complete experiment comparing different distance approximations.
        
        Args:
            test_points (np.ndarray): Array of 2D test points.
            pairs_to_test (List[Tuple[int, int]], optional): List of (i,j) index pairs to test.
                                                             If None, tests all unique pairs.
            visualize (bool): Whether to create visualizations.
            
        Returns:
            dict: Dictionary containing results and statistics.
        """
        n_points = len(test_points)
        
        if pairs_to_test is None:
            # Generate all unique pairs
            pairs_to_test = [(i, j) for i in range(n_points) for j in range(i+1, n_points)]
        
        n_pairs = len(pairs_to_test)
        print(f"Testing {n_pairs} point pairs on {self.manifold_name} manifold...")
        
        # Initialize result arrays
        results = {
            'true_geodesic': np.zeros(n_pairs),
            'linear_interpolation': np.zeros(n_pairs),
            'midpoint': np.zeros(n_pairs),
            'weighted_midpoint': np.zeros(n_pairs),
            'endpoint_average': np.zeros(n_pairs),
            'euclidean': np.zeros(n_pairs),
            'computation_times': {}
        }
        
        # Initialize timing
        for method in ['true_geodesic', 'linear_interpolation', 'midpoint', 
                      'weighted_midpoint', 'endpoint_average', 'euclidean']:
            results['computation_times'][method] = 0.0
        
        # Compute distances using each method
        for idx, (i, j) in enumerate(pairs_to_test):
            u, v = test_points[i], test_points[j]
            
            # True geodesic distance (ground truth)
            start_time = time.time()
            try:
                results['true_geodesic'][idx] = self.manifold_instance._geodesic_distance(u, v)
            except Exception as e:
                print(f"Warning: Geodesic computation failed for pair ({i},{j}): {e}")
                results['true_geodesic'][idx] = np.nan
            results['computation_times']['true_geodesic'] += time.time() - start_time
            
            # Linear interpolation approximation
            start_time = time.time()
            results['linear_interpolation'][idx] = self.manifold_instance.linear_interpolation_distance(u, v)
            results['computation_times']['linear_interpolation'] += time.time() - start_time
            
            # Midpoint approximation
            start_time = time.time()
            results['midpoint'][idx] = self.manifold_instance.midpoint_approximation(u, v)
            results['computation_times']['midpoint'] += time.time() - start_time
            
            # Weighted midpoint approximation
            start_time = time.time()
            results['weighted_midpoint'][idx] = self.manifold_instance.weighted_midpoint_approximation(u, v)
            results['computation_times']['weighted_midpoint'] += time.time() - start_time
            
            # Endpoint average approximation
            start_time = time.time()
            results['endpoint_average'][idx] = self.manifold_instance.endpoint_average_approximation(u, v)
            results['computation_times']['endpoint_average'] += time.time() - start_time
            
            # Euclidean distance
            start_time = time.time()
            results['euclidean'][idx] = self.manifold_instance.euclidean_distance(u, v)
            results['computation_times']['euclidean'] += time.time() - start_time
        
        # Remove NaN values for analysis
        valid_mask = ~np.isnan(results['true_geodesic'])
        if not np.all(valid_mask):
            print(f"Warning: {np.sum(~valid_mask)} geodesic computations failed and will be excluded from analysis")
        
        # Compute error statistics
        stats = self._compute_error_statistics(results, valid_mask)
        results['statistics'] = stats
        results['test_points'] = test_points
        results['pairs_tested'] = pairs_to_test
        
        # Print results summary
        self._print_results_summary(stats)
        
        # Create visualizations
        if visualize:
            self._create_visualizations(results, test_points, valid_mask)
        
        return results
    
    def _compute_error_statistics(self, results: dict, valid_mask: np.ndarray) -> dict:
        """Compute error statistics for each approximation method."""
        stats = {}
        true_distances = results['true_geodesic'][valid_mask]
        
        methods = ['linear_interpolation', 'midpoint', 'weighted_midpoint', 'endpoint_average', 'euclidean']
        
        for method in methods:
            approx_distances = results[method][valid_mask]
            
            # Absolute errors
            abs_errors = np.abs(approx_distances - true_distances)
            
            # Relative errors (avoid division by zero)
            rel_errors = abs_errors / np.maximum(true_distances, 1e-10)
            
            stats[method] = {
                'mean_absolute_error': np.mean(abs_errors),
                'max_absolute_error': np.max(abs_errors),
                'mean_relative_error': np.mean(rel_errors),
                'max_relative_error': np.max(rel_errors),
                'rmse': np.sqrt(np.mean(abs_errors**2)),
                'mean_computation_time': results['computation_times'][method] / len(approx_distances)
            }
        
        return stats
    
    def _print_results_summary(self, stats: dict):
        """Print a summary of the experimental results."""
        print(f"\n=== Results Summary for {self.manifold_name} Manifold ===")
        print(f"{'Method':<20} {'MAE':<10} {'Max AE':<10} {'MRE':<10} {'Max RE':<10} {'RMSE':<10} {'Time (ms)':<12}")
        print("-" * 92)
        
        for method, method_stats in stats.items():
            print(f"{method:<20} "
                  f"{method_stats['mean_absolute_error']:<10.4f} "
                  f"{method_stats['max_absolute_error']:<10.4f} "
                  f"{method_stats['mean_relative_error']:<10.4f} "
                  f"{method_stats['max_relative_error']:<10.4f} "
                  f"{method_stats['rmse']:<10.4f} "
                  f"{method_stats['mean_computation_time']*1000:<12.4f}")
    
    def _create_visualizations(self, results: dict, test_points: np.ndarray, valid_mask: np.ndarray):
        """Create comprehensive visualizations of the experimental results."""
        
        # 1. Visualize the manifold curvature with test points
        print("Generating manifold curvature visualization...")
        labels = np.arange(len(test_points))  # Simple labels for points
        # Call the visualize_manifold_curvature method on the manifold instance
        # The z_range will default to the manifold's bounds if not specified by experiment
        # We explicitly set a z_range here to match the original function's behavior
        curvature, Z1, Z2 = self.manifold_instance.visualize_manifold_curvature(
            resolution=50,
            data_points=test_points,
            labels=labels
        )
        
        # 2. Error comparison plots
        self._plot_error_comparisons(results, valid_mask)
        
        # 3. Distance correlation plots
        self._plot_distance_correlations(results, valid_mask)
        
        # 4. Computation time comparison
        self._plot_computation_times(results)
        
        # 5. Example geodesic visualization
        if len(test_points) >= 2:
            print("Generating example geodesic visualization...")
            # Call the plot_geodesics method on the manifold instance
            self.manifold_instance.plot_geodesics(test_points[0], test_points[1], curvature, Z1, Z2)
    
    def _plot_error_comparisons(self, results: dict, valid_mask: np.ndarray):
        """Create error comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Distance Approximation Errors - {self.manifold_name} Manifold', fontsize=16)
        
        true_distances = results['true_geodesic'][valid_mask]
        methods = ['linear_interpolation', 'midpoint', 'weighted_midpoint', 'endpoint_average', 'euclidean']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Absolute errors vs true distance
        ax = axes[0, 0]
        for method, color in zip(methods, colors):
            approx_distances = results[method][valid_mask]
            abs_errors = np.abs(approx_distances - true_distances)
            ax.scatter(true_distances, abs_errors, alpha=0.6, label=method.replace('_', ' ').title(), 
                      color=color, s=30)
        ax.set_xlabel('True Geodesic Distance')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error vs True Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Relative errors vs true distance
        ax = axes[0, 1]
        for method, color in zip(methods, colors):
            approx_distances = results[method][valid_mask]
            rel_errors = abs_errors / np.maximum(true_distances, 1e-10) # Use abs_errors directly
            ax.scatter(true_distances, rel_errors, alpha=0.6, label=method.replace('_', ' ').title(), 
                      color=color, s=30)
        ax.set_xlabel('True Geodesic Distance')
        ax.set_ylabel('Relative Error')
        ax.set_title('Relative Error vs True Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error distribution (box plot)
        ax = axes[1, 0]
        error_data = []
        labels_list = []
        for method in methods:
            approx_distances = results[method][valid_mask]
            abs_errors = np.abs(approx_distances - true_distances)
            error_data.append(abs_errors)
            labels_list.append(method.replace('_', ' ').title())
        
        ax.boxplot(error_data, labels=labels_list)
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # RMSE comparison bar plot
        ax = axes[1, 1]
        rmse_values = [results['statistics'][method]['rmse'] for method in methods]
        bars = ax.bar(labels_list, rmse_values, color=colors, alpha=0.7)
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rmse:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_distance_correlations(self, results: dict, valid_mask: np.ndarray):
        """Create correlation plots between approximated and true distances."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Distance Correlations - {self.manifold_name} Manifold', fontsize=16)
        
        true_distances = results['true_geodesic'][valid_mask]
        methods = ['linear_interpolation', 'midpoint', 'weighted_midpoint', 'endpoint_average', 'euclidean']
        
        for idx, method in enumerate(methods):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            
            approx_distances = results[method][valid_mask]
            
            # Scatter plot
            ax.scatter(true_distances, approx_distances, alpha=0.6, s=30)
            
            # Perfect correlation line
            min_dist, max_dist = min(np.min(true_distances), np.min(approx_distances)), \
                                max(np.max(true_distances), np.max(approx_distances))
            ax.plot([min_dist, max_dist], [min_dist, max_dist], 'r--', alpha=0.8, label='Perfect Correlation')
            
            # Correlation coefficient
            corr_coef = np.corrcoef(true_distances, approx_distances)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr_coef:.4f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('True Geodesic Distance')
            ax.set_ylabel('Approximated Distance')
            ax.set_title(method.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_computation_times(self, results: dict):
        """Create computation time comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['true_geodesic', 'linear_interpolation', 'midpoint', 'weighted_midpoint', 'endpoint_average', 'euclidean']
        times = [results['computation_times'][method] * 1000 for method in methods]  # Convert to milliseconds
        labels = [method.replace('_', ' ').title() for method in methods]
        colors = ['darkred', 'blue', 'red', 'green', 'orange', 'purple']
        
        bars = ax.bar(labels, times, color=colors, alpha=0.7)
        ax.set_ylabel('Total Computation Time (ms)')
        ax.set_title(f'Computation Time Comparison - {self.manifold_name} Manifold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()