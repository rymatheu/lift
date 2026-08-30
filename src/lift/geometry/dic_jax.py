import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import math

from functools import partial

class DimensionInducedClusteringJAX:
    """
    Dimension Induced Clustering (DIC) rewritten in JAX.
    
    Maps each point X_i to a pair (d_i, c_i) where:
      - d_i is the local intrinsic dimension.
      - c_i is the local density.
      
    Features a batched exact kNN to handle immense datasets on GPU without OOM.
    """

    def __init__(self, batch_size=2048):
        # batch_size controls GPU memory consumption during kNN
        self.batch_size = batch_size

    def fit_transform(self, X, radius_range, k_min=None, k_max=None):
        X = jnp.asarray(X)
        N, dim = X.shape

        # Handle defaults
        if k_min is None:
            k_min_val = int(math.ceil(0.01 * N))
        elif isinstance(k_min, float):
            k_min_val = int(math.ceil(k_min * N))
        else:
            k_min_val = int(k_min)

        if k_max is None:
            k_max_val = int(math.floor(0.1 * N))
        elif isinstance(k_max, float):
            k_max_val = int(math.floor(k_max * N))
        else:
            k_max_val = int(k_max)

        r_min, r_max = radius_range

        k_min_val = max(2, k_min_val)
        if k_max_val >= N:
            k_max_val = N - 1

        # 1. Compute Nearest Neighbors in chunks to save memory
        distances = self._batched_knn(X, k_max_val + 1)
        
        # Exclude self (index 0 is distance ~0.0)
        distances_no_self = distances[:, 1:]

        # 2. Execute JIT-compiled vectorized logic
        return self._compute_dic(distances_no_self, r_min, r_max, k_min_val, k_max_val, N)

    def _batched_knn(self, X, k):
        """Computes top-k exact neighbors efficiently using batched processing."""
        N = X.shape[0]
        
        @jax.jit
        def get_k_neighbors_batch(X_batch):
            # Compute pairwise distances for the batch -> Shape: (Batch, N)
            dists_sq = jnp.sum((X_batch[:, None, :] - X[None, :, :])**2, axis=-1)
            # lax.top_k finds the largest elements, so we pass negative squared distances
            neg_sq_dists, _ = lax.top_k(-dists_sq, k)
            return jnp.sqrt(-neg_sq_dists)
            
        distances = []
        for i in range(0, N, self.batch_size):
            X_batch = X[i:i + self.batch_size]
            d = get_k_neighbors_batch(X_batch)
            distances.append(d)
            
        return jnp.concatenate(distances, axis=0)

    @staticmethod
    @jax.jit
    def _compute_dic(distances, r_min, r_max, k_min_val, k_max_val, N):
        """Vectorized JIT-compiled log-log linear fit and coordinate extraction."""
        K = distances.shape[1]
        ranks = jnp.arange(1, K + 1)[None, :]  # Shape: (1, K)
        
        # Create Boolean Masks for validity ranges
        mask_r = (distances >= r_min) & (distances <= r_max)
        mask_k = (ranks >= k_min_val) & (ranks <= k_max_val)
        mask = mask_r & mask_k
        
        # Prepare Data for Regression, utilizing masks to zero-out invalid points
        x = jnp.where(mask, jnp.log(jnp.maximum(distances, 1e-12)), 0.0)
        y = jnp.where(mask, jnp.log(ranks / N), 0.0)
        
        n_pts = jnp.sum(mask, axis=-1)
        sum_x = jnp.sum(x, axis=-1)
        sum_y = jnp.sum(y, axis=-1)
        sum_xy = jnp.sum(x * y, axis=-1)
        sum_xx = jnp.sum(x * x, axis=-1)
        
        # Linear Regression Math
        denom = n_pts * sum_xx - sum_x**2
        valid_fit = (n_pts >= 2) & (jnp.abs(denom) > 1e-12)
        
        safe_denom = jnp.where(denom == 0, 1.0, denom)
        safe_n = jnp.where(n_pts == 0, 1.0, n_pts)
        
        slope = jnp.where(valid_fit, (n_pts * sum_xy - sum_x * sum_y) / safe_denom, 0.0)
        intercept = jnp.where(valid_fit, (sum_y - slope * sum_x) / safe_n, 0.0)
        
        # Calculate optimal radius r* (Global)
        n_valid_total = jnp.sum(valid_fit)
        safe_n_valid_total = jnp.where(n_valid_total == 0, 1.0, n_valid_total)
        
        d_mean = jnp.sum(slope) / safe_n_valid_total
        b_mean = jnp.sum(intercept) / safe_n_valid_total
        
        numerator = jnp.sum(jnp.where(valid_fit, (slope - d_mean) * (intercept - b_mean), 0.0))
        denominator = jnp.sum(jnp.where(valid_fit, (slope - d_mean)**2, 0.0))
        
        safe_global_den = jnp.where(denominator == 0, 1.0, denominator)
        log_r_star = jnp.where((n_valid_total > 1) & (denominator > 1e-12), -numerator / safe_global_den, 0.0)
        
        # Final Output Computation
        d_final = jnp.where(valid_fit, slope, 0.0)
        c_final = jnp.where(valid_fit, slope * log_r_star + intercept, 0.0)
        
        return jnp.stack([d_final, c_final], axis=-1)


@jax.jit
def estimate_global_parameters(X, point_index, r_min, r_max):
    """
    JIT-compiled Global Parameter Estimator.
    Uses jnp.cumsum on a boolean mask to exactly replicate dynamic rank assignments.
    """
    target_point = X[point_index]
    dists = jnp.linalg.norm(X - target_point, axis=-1)
    dists = jnp.sort(dists)
    
    # Identify non-self points
    valid_nonzero = dists > 1e-12
    
    # Dynamically track global rank without dropping array dimensions
    global_ranks = jnp.cumsum(valid_nonzero)
    
    # Apply specific range interval boundaries
    range_mask = valid_nonzero & (dists >= r_min) & (dists <= r_max)
    
    x = jnp.where(range_mask, jnp.log(jnp.maximum(dists, 1e-12)), 0.0)
    y = jnp.where(range_mask, jnp.log(jnp.maximum(global_ranks, 1)), 0.0)
    
    n_pts = jnp.sum(range_mask)
    sum_x = jnp.sum(x)
    sum_y = jnp.sum(y)
    sum_xy = jnp.sum(x * y)
    sum_xx = jnp.sum(x * x)
    
    denom = n_pts * sum_xx - sum_x**2
    valid_fit = (n_pts >= 2) & (jnp.abs(denom) > 1e-12)
    
    safe_denom = jnp.where(denom == 0, 1.0, denom)
    safe_n = jnp.where(n_pts == 0, 1.0, n_pts)
    
    slope = jnp.where(valid_fit, (n_pts * sum_xy - sum_x * sum_y) / safe_denom, 0.0)
    intercept = jnp.where(valid_fit, (sum_y - slope * sum_x) / safe_n, 0.0)
    
    # Computes accurate R-squared entirely via vectorized equations
    mean_y = sum_y / safe_n
    ss_tot = jnp.sum(jnp.where(range_mask, (y - mean_y)**2, 0.0))
    ss_res = jnp.sum(jnp.where(range_mask, (y - (slope * x + intercept))**2, 0.0))
    r_squared = jnp.where(valid_fit & (ss_tot > 1e-12), 1.0 - (ss_res / jnp.where(ss_tot == 0, 1.0, ss_tot)), 0.0)
    
    n_est = jnp.where(valid_fit, slope, 0.0)
    K_est = jnp.where(valid_fit, jnp.exp(intercept), 0.0)
    
    return n_est, K_est, r_squared


@jax.jit
def estimate_global_parameters_ratio(X, point_index, r_min, r_max):
    """
    Extremely fast JIT-compiled ratio estimation.
    Bypasses costly sorting by recognizing that cumulative counts can 
    be achieved using independent sum reductions!
    """
    target_point = X[point_index]
    dists = jnp.linalg.norm(X - target_point, axis=-1)
    
    valid_nonzero = dists > 1e-12
    
    # Perform count evaluations directly on the un-sorted vector 
    N_max = jnp.sum(valid_nonzero & (dists <= r_max))
    N_min = jnp.sum(valid_nonzero & (dists <= r_min))
    
    valid_params = (N_min >= 1) & (r_min < r_max)
    
    safe_N_max = jnp.maximum(N_max, 1.0)
    safe_N_min = jnp.maximum(N_min, 1.0)
    
    log_count_ratio = jnp.log(safe_N_max / safe_N_min)
    log_radius_ratio = jnp.log(r_max / r_min)
    
    n_est = jnp.where(valid_params, log_count_ratio / log_radius_ratio, 0.0)
    K_est = jnp.where(valid_params, N_max / (r_max ** n_est), 0.0)
    
    return n_est, K_est

#@partial(jax.jit, static_argnames=('variance_threshold',))
@jax.jit
def estimate_local_id_pca(X, point_idx, radius, variance_threshold=0.90):
    """
    Estimates local intrinsic dimension using a fixed ball radius via masked Covariance.
    
    Args:
        X: jnp.array of shape (N, d) containing the full point cloud.
        point_idx: int, the index of the center point.
        radius: float, the Euclidean distance threshold defining the ball.
        variance_threshold: float, the target cumulative variance ratio.
    """
    # 1. Isolate target and compute all pairwise Euclidean distances
    target_point = X[point_idx]
    distances = jnp.linalg.norm(X - target_point, axis=1)
    
    # 2. Create a boolean mask for points inside the ball radius
    # Shape stays permanently (N,) which keeps JAX happy
    mask = distances <= radius
    N_local = jnp.sum(mask)
    
    # 3. Compute the local mean safely using the mask
    # We use jnp.where to prevent division-by-zero if a ball is empty
    local_sum = jnp.sum(X * mask[:, None], axis=0)
    local_mean = local_sum / jnp.where(N_local > 0, N_local, 1.0)
    
    # 4. Center the data and zero out any points outside the radius
    X_centered = (X - local_mean) * mask[:, None]
    
    # 5. Compute the (d, d) Covariance Matrix
    # Points outside the radius contribute exactly 0 to the matrix dot product
    degrees_of_freedom = jnp.where(N_local > 1, N_local - 1, 1.0)
    covariance_matrix = (X_centered.T @ X_centered) / degrees_of_freedom
    
    # 6. Solve for eigenvalues directly
    # eigh is optimized for symmetric matrices like Covariance
    eigenvalues = jnp.linalg.eigvalsh(covariance_matrix)
    
    # eigh returns values in ascending order; flip to descending (largest variance first)
    eigenvalues = eigenvalues[::-1]
    
    # 7. Calculate cumulative explained variance ratio
    total_variance = jnp.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / jnp.where(total_variance > 0, total_variance, 1.0)
    cumulative_variance = jnp.cumsum(explained_variance_ratio)
    
    # 8. Extract intrinsic dimension
    intrinsic_dim = jnp.argmax(cumulative_variance >= variance_threshold) + 1
    
    return intrinsic_dim, cumulative_variance