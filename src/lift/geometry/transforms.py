"""Rigid-motion helpers for aligning two point clouds."""

import jax
import jax.numpy as jnp


@jax.jit
def find_rigid_transform(P, Q):
    """
    Finds the rotation matrix R and translation vector d that minimizes 
    the squared error between two paired point clouds P and Q.

    (R*P_i + d - Q_i)^2
    
    Args:
        P: jnp.ndarray of shape (N, D) - Source point cloud
        Q: jnp.ndarray of shape (N, D) - Target point cloud
        
    Returns:
        R: jnp.ndarray of shape (D, D) - Rotation matrix
        d: jnp.ndarray of shape (D,) - Translation vector
    """
    # 1. Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)
    
    # 2. Center the point clouds
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 3. Compute the covariance matrix
    H = jnp.dot(P_centered.T, Q_centered)
    
    # 4. Perform Singular Value Decomposition (SVD)
    U, S, Vt = jnp.linalg.svd(H)
    
    # 5. Calculate preliminary rotation matrix (V @ U^T)
    # Note: jnp.linalg.svd returns Vt (V transpose), so Vt.T is V
    R = jnp.dot(Vt.T, U.T)
    
    # 6. Handle the reflection case to guarantee a right-handed coordinate system
    det = jnp.linalg.det(R)
    num_dims = P.shape[1]
    
    # Create a correction diagonal matrix using jnp.where for full JIT compatibility
    diag = jnp.ones(num_dims)
    diag = diag.at[-1].set(jnp.where(det < 0, -1.0, 1.0))
    D = jnp.diag(diag)
    
    # Recompute R with reflection correction: R = V @ D @ U^T
    R = jnp.dot(Vt.T, jnp.dot(D, U.T))
    
    # 7. Compute the translation vector
    d = centroid_Q - jnp.dot(R, centroid_P)
    
    return R, d

def skew(v):
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

@jax.jit
def rod_rot(v, t):
    R = jnp.eye(3) + jnp.sin(t)*skew(v) + (1-jnp.cos(t))*skew(v)@skew(v)
    return R
