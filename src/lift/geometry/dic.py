
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial import KDTree
from scipy.stats import linregress


class DimensionInducedClustering:
    """
    Dimension Induced Clustering (DIC)from Gionis et al. (KDD 2005).
    
    Maps each point X_i to a pair (d_i, c_i) where:
      - d_i is the local intrinsic dimension.
      - c_i is the local density.
    """

    def __init__(self):
        pass

    def fit_transform(self, X, radius_range, k_min=None, k_max=None):
        """
        Computation of the (d_i, c_i) pairs for the dat X.

        Parameters:
        -----------
        X : array of shape (N, d)
        radius_range : tuple (r_min, r_max)
            Radii range [r_min, r_max] for the local growth curve.
            Only neighbors within these distances are considered for the fit.
        k_min : int or float, optional
            Min number of neighbors required.
            If float then interpreted as a fraction of N (k_min * N).
            Default is 0.01 * N.
        k_max : int or float, optional
            The maximum number of neighbors to consider.
            If float, then interpreted as a fraction of N (k_max * N).
            Default is 0.1 * N.

        Returns:
        --------
        result : ndarray of dimensions (N, 2)
            The i-th row is [d_i, c_i].
            Points that do not satisfy the neighbor constraints return [0.0, 0.0].
        """
        X = np.asarray(X)
        N, dim = X.shape

        # Handle defaults for k_min and k_max
        if k_min is None:
            k_min_val = int(np.ceil(0.01 * N))
        elif isinstance(k_min, float):
            k_min_val = int(np.ceil(k_min * N))
        else:
            k_min_val = int(k_min)

        if k_max is None:
            k_max_val = int(np.floor(0.1 * N))
        elif isinstance(k_max, float):
            k_max_val = int(np.floor(k_max * N))
        else:
            k_max_val = int(k_max)

        r_min, r_max = radius_range

        k_min_val = max(2, k_min_val) # Need at least 2 points to fit a line
        if k_max_val >= N:
            k_max_val = N - 1

        # We need neighbors up to k_max_val to check rank constraints,
        # but we also need to respect r_max.
        # Efficient approach: Query k_max_val + 1 neighbors (including self)
        # and filter by radius.
        
        # Note: If r_max is very large, k_max limits the search. 
        # If r_max is small, we might get fewer than k_max neighbors.
        # We start by getting k_max candidates.
        nbrs = NearestNeighbors(n_neighbors=k_max_val + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Arrays to store d_i and b_i (intercept)
        # Initialize with NaNs to mark invalid points initially
        d_vals = np.full(N, np.nan)
        b_vals = np.full(N, np.nan)
        valid_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            # Get distances for point i (excluding self at index 0)
            # distances[i] is sorted.
            dists_i = distances[i, 1:] 
            
            # Filter based on Radius Range [r_min, r_max]
            # AND Rank Range [k_min, k_max] (implicitly handled by taking first k_max neighbors)
            
            # 1. Select neighbors within radius range
            # Note: dists_i corresponds to ranks 1, 2, ..., k_max
            ranks = np.arange(1, len(dists_i) + 1)
            
            # Mask for radius range
            mask_r = (dists_i >= r_min) & (dists_i <= r_max)
            
            # Mask for rank range (k_min to k_max)
            mask_k = (ranks >= k_min_val) & (ranks <= k_max_val)
            
            # Intersection of constraints
            final_mask = mask_r & mask_k
            
            valid_dists = dists_i[final_mask]
            valid_ranks = ranks[final_mask]

            # Check if we have enough points to fit
            # The prompt implies: "points that do not have enough neighbors within the specified range"
            # We enforce at least 2 points for a line fit.
            if len(valid_dists) < 2:
                continue

            # Prepare data for log-log fit
            # x = log(r), y = log(k/N) = log(G_x(r))
            # Avoid log(0)
            valid_dists = np.maximum(valid_dists, 1e-12)
            
            log_r = np.log(valid_dists)
            log_G = np.log(valid_ranks / N)

            # Fit line: log_G = d_i * log_r + b_i
            # Polyfit returns [slope, intercept]
            slope, intercept = np.polyfit(log_r, log_G, deg=1)
            
            d_vals[i] = slope
            b_vals[i] = intercept
            valid_mask[i] = True

        # Calculate optimal radius r* (Section 4.3, Lemma 1 in paper)
        # We only use valid points to calculate the global r* statistics
        if np.sum(valid_mask) > 1:
            d_valid = d_vals[valid_mask]
            b_valid = b_vals[valid_mask]
            
            d_mean = np.mean(d_valid)
            b_mean = np.mean(b_valid)
            
            numerator = np.sum((d_valid - d_mean) * (b_valid - b_mean))
            denominator = np.sum((d_valid - d_mean)**2)
            
            if denominator == 0:
                # Fallback if variance of d is 0 (all points have same dimension)
                # In this case, r* doesn't matter for decorrelation, pick r* = 1 (log r* = 0)
                log_r_star = 0.0
            else:
                log_r_star = - (numerator / denominator)
        else:
            log_r_star = 0.0

        # Compute Final Output (d_i, c_i)
        # c_i = d_i * log_r_star + b_i
        results = np.zeros((N, 2))
        
        for i in range(N):
            if valid_mask[i]:
                d_i = d_vals[i]
                b_i = b_vals[i]
                c_i = d_i * log_r_star + b_i
                results[i] = [d_i, c_i]
            else:
                results[i] = [0.0, 0.0]

        return results
    

def estimate_global_parameters(X, point_index, r_min, r_max):
    """
    Estimates the intrinsic dimension (n) and density coefficient (K) 
    for a specific point using all neighbors within the interval [r_min, r_max].

    It fits the power law V(r) = K * r^n by transforming it to:
    log(Rank) = n * log(Radius) + log(K)

    Parameters:
    -----------
    X : array-like of shape (N, d)
        The dataset.
    point_index : int
        The index of the query point in X.
    r_min : float
        The minimum radius of the shell.
    r_max : float
        The maximum radius of the shell.

    Returns:
    --------
    n : float
        The estimated intrinsic dimension (slope).
    K : float
        The estimated density coefficient (exp(intercept)).
    r_squared : float
        The R^2 value of the fit, indicating how well the power law holds.
    """
    X = np.asarray(X)
    target_point = X[point_index].reshape(1, -1)
    
    # 1. Collect all neighbors up to r_max
    # We strictly need neighbors only up to r_max
    nn = NearestNeighbors(radius=r_max)
    nn.fit(X)
    
    dists, _ = nn.radius_neighbors(target_point, radius=r_max)
    dists = np.sort(dists[0])
    
    # Remove the point itself (distance ~0) to avoid log(0)
    dists = dists[dists > 1e-12]
    
    # 2. Assign Global Ranks
    # The rank 'i' corresponds to the volume V(r_i)
    # Ranks are 1, 2, 3... corresponding to the sorted distances
    global_ranks = np.arange(1, len(dists) + 1)
    
    # 3. Filter for the interval [r_min, r_max]
    # We select points strictly within this range for the regression,
    # BUT we keep their global ranks (preserving the K*r^n relationship).
    mask = (dists >= r_min) & (dists <= r_max)
    
    valid_dists = dists[mask]
    valid_ranks = global_ranks[mask]
    
    # Check if we have enough points for regression (need at least 2)
    if len(valid_dists) < 2:
        return 0.0, 0.0, 0.0

    # 4. Log-Log Transformation
    log_r = np.log(valid_dists)
    log_ranks = np.log(valid_ranks)

    # 5. Linear Regression
    # Equation: Y = slope * X + intercept
    # log(rank) = n * log(r) + log(K)
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_ranks)
    
    n_est = slope
    K_est = np.exp(intercept)
    r_squared = r_value**2
    
    return n_est, K_est, r_squared

def estimate_global_parameters_ratio(X, point_index, r_min, r_max):
    """
    Estimates intrinsic dimension (n) and density (K) using the ratio of 
    neighbor counts at r_max and r_min.

    Formula derived from V(r) = K * r^n:
      n = log(N_max / N_min) / log(r_max / r_min)
      K = N_max / (r_max^n)

    Parameters:
    -----------
    X : array-like of shape (N, d)
        The dataset.
    point_index : int
        The index of the query point.
    r_min : float
        The inner radius.
    r_max : float
        The outer radius.

    Returns:
    --------
    n : float
        The estimated intrinsic dimension.
    K : float
        The estimated density coefficient.
    """
    X = np.asarray(X)
    target_point = X[point_index].reshape(1, -1)
    
    # 1. Query neighbors up to r_max
    nn = NearestNeighbors(radius=r_max)
    nn.fit(X)
    
    dists, _ = nn.radius_neighbors(target_point, radius=r_max)
    dists = np.sort(dists[0])
    
    # Remove self (distance ~ 0)
    dists = dists[dists > 1e-12]
    
    # 2. Get counts V(r_max) and V(r_min)
    # N_max is simply the total count of neighbors found within r_max
    N_max = len(dists)
    
    # N_min is the count of neighbors with distance <= r_min
    # searchsorted returns the insertion index, which equals the count for sorted arrays
    N_min = np.searchsorted(dists, r_min, side='right')
    
    # 3. Handle Edge Cases
    if N_min < 1:
        # Cannot compute log(0). 
        # Return 0.0 or nan to indicate insufficient data at r_min.
        return 0.0, 0.0

    if r_min >= r_max:
        raise ValueError("r_min must be strictly smaller than r_max")

    # 4. Compute n using the Ratio Formula
    # n = ln(N_max / N_min) / ln(r_max / r_min)
    log_count_ratio = np.log(N_max / N_min)
    log_radius_ratio = np.log(r_max / r_min)
    
    n_est = log_count_ratio / log_radius_ratio
    
    # 5. Compute K
    # using the relation N_max = K * r_max^n  =>  K = N_max / r_max^n
    # (Alternatively, one could use N_min / r_min^n, theoretically identical)
    K_est = N_max / (r_max ** n_est)
    
    return n_est, K_est