"""Curve alignment functions.

This module provides functions for aligning curves to enable comparison.
The primary method is quaternion-based alignment. A legacy Procrustes alignment
is included but NOT VALIDATED and should be used with caution.

Example:
    >>> from src.analyze.spline_fitting import quaternion_alignment
    >>>
    >>> # Align curve2 to curve1
    >>> R, t = quaternion_alignment(curve1, curve2)
    >>> curve2_aligned = (R @ curve2.T).T + t
"""

import warnings
import numpy as np


def centroid(X):
    """Compute centroid of point cloud.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_dims)
        Point cloud.

    Returns
    -------
    center : ndarray, shape (n_dims,)
        Centroid coordinates.
    """
    return np.mean(X, axis=0)


def quaternion_alignment(P, Q):
    """Align curve Q onto curve P using quaternion-based optimal rotation.

    This is the RECOMMENDED alignment method. Uses the Kearsley quaternion algorithm
    to find the optimal rotation matrix that minimizes RMSD between P and Q.

    Parameters
    ----------
    P : ndarray, shape (n_points, 3)
        Reference curve (target).
    Q : ndarray, shape (n_points, 3)
        Curve to align (source).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Optimal rotation matrix.
    t : ndarray, shape (3,)
        Optimal translation vector.

    Notes
    -----
    To apply alignment:
        Q_aligned = (R @ Q.T).T + t

    Algorithm:
    1. Center both point sets
    2. Compute correlation matrix M = Q'.T @ P'
    3. Build Kearsley 4x4 matrix from M
    4. Extract quaternion from largest eigenvalue
    5. Convert quaternion to rotation matrix
    6. Compute translation to align centroids

    References
    ----------
    Kearsley, S. K. (1989). "On the orthogonal transformation used for structural
    comparisons." Acta Crystallographica Section A, 45(2), 208-210.
    """
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape. Got P: {P.shape}, Q: {Q.shape}")

    # Center the points
    P_cent = centroid(P)
    Q_cent = centroid(Q)
    P_prime = P - P_cent
    Q_prime = Q - Q_cent

    # Correlation matrix
    M = Q_prime.T @ P_prime

    # Kearsley (Davenport) 4x4 matrix
    A = np.array([
        [ M[0,0]+M[1,1]+M[2,2],   M[1,2]-M[2,1],         M[2,0]-M[0,2],         M[0,1]-M[1,0]       ],
        [ M[1,2]-M[2,1],         M[0,0]-M[1,1]-M[2,2],  M[0,1]+M[1,0],         M[0,2]+M[2,0]       ],
        [ M[2,0]-M[0,2],         M[0,1]+M[1,0],         M[1,1]-M[0,0]-M[2,2],  M[1,2]+M[2,1]       ],
        [ M[0,1]-M[1,0],         M[0,2]+M[2,0],         M[1,2]+M[2,1],         M[2,2]-M[0,0]-M[1,1]]
    ], dtype=np.float64)
    A = A / 3.0

    # Find eigenvector with highest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    max_idx = np.argmax(eigenvalues)
    q = eigenvectors[:, max_idx]
    q = q / np.linalg.norm(q)

    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q2*q1 + q0*q3),             q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q3*q1 - q0*q2),             2*(q3*q2 + q0*q1),             q0**2 - q1**2 - q2**2 + q3**2]
    ])

    # Translation
    t = P_cent - R @ Q_cent

    return R, t


def procrustes_alignment(P, Q):
    """Align curve Q onto curve P using Procrustes analysis.

    **WARNING: LEGACY METHOD - NOT VALIDATED**

    This method is kept for historical reference only. It has not been validated
    against known-good implementations. Use quaternion_alignment() instead.

    Parameters
    ----------
    P : ndarray, shape (n_points, 3)
        Reference curve (target).
    Q : ndarray, shape (n_points, 3)
        Curve to align (source).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix (may not be optimal).
    t : ndarray, shape (3,)
        Translation vector.

    Warnings
    --------
    This function issues a warning every time it's called.
    """
    warnings.warn(
        "procrustes_alignment() is a LEGACY method that has NOT been validated. "
        "Results may be incorrect. Use quaternion_alignment() instead.",
        UserWarning,
        stacklevel=2
    )

    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape. Got P: {P.shape}, Q: {Q.shape}")

    # Center the points
    P_cent = centroid(P)
    Q_cent = centroid(Q)
    P_prime = P - P_cent
    Q_prime = Q - Q_cent

    # SVD-based Procrustes
    # M = P_prime.T @ Q_prime
    # U, S, Vt = np.linalg.svd(M)
    # R = U @ Vt

    # Alternative formulation (may differ from standard Procrustes)
    M = Q_prime.T @ P_prime
    U, S, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T

    # Check for reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = P_cent - R @ Q_cent

    return R, t
