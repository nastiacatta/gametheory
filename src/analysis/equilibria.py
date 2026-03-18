"""
Static equilibrium analysis for the El Farol threshold game.

Pure-strategy Nash equilibria:
    Exactly the profiles with A = L (attendance equals threshold).
    Count: C(n, L) = n! / (L! * (n-L)!)

Symmetric mixed-strategy Nash equilibrium:
    The equilibrium probability p* satisfies:
        Pr(X <= L-1) = 1/2,  where X ~ Bin(n-1, p*)
    
    This condition ensures indifference between attending and staying home:
        E[u(attend)] = Pr(X <= L-1) - Pr(X >= L) = 2*Pr(X <= L-1) - 1 = 0
"""

from __future__ import annotations

from math import comb
from typing import Tuple

from scipy import stats


def count_pure_ne(n_players: int, threshold: int) -> int:
    """
    Count pure-strategy Nash equilibria in the El Farol threshold game.
    
    Under weak-threshold payoffs, NE are exactly the profiles with A = L.
    The count is C(n, L) = n choose L.
    
    Args:
        n_players: Total number of players.
        threshold: Capacity threshold L.
    
    Returns:
        Number of pure-strategy Nash equilibria.
    
    Raises:
        ValueError: If threshold is out of valid range [0, n_players].
    """
    if not (0 <= threshold <= n_players):
        raise ValueError(f"threshold must be in [0, {n_players}], got {threshold}")
    if n_players < 0:
        raise ValueError(f"n_players must be non-negative, got {n_players}")
    
    return comb(n_players, threshold)


def solve_symmetric_mixed_p_star(
    n_players: int,
    threshold: int,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> float:
    """
    Solve for the symmetric mixed-strategy Nash equilibrium probability p*.
    
    The equilibrium condition is:
        Pr(X <= L-1) = 1/2,  where X ~ Bin(n-1, p*)
    
    Uses bisection on [0, 1] to find p* such that the CDF condition holds.
    
    Args:
        n_players: Total number of players (n).
        threshold: Capacity threshold (L).
        tol: Convergence tolerance for bisection.
        max_iter: Maximum iterations.
    
    Returns:
        The equilibrium probability p* in (0, 1).
    
    Raises:
        ValueError: If parameters are invalid or no solution exists.
    """
    if n_players < 2:
        raise ValueError("n_players must be at least 2 for mixed equilibrium")
    if not (1 <= threshold <= n_players - 1):
        raise ValueError(
            f"threshold must be in [1, {n_players - 1}] for interior mixed equilibrium, "
            f"got {threshold}"
        )
    
    n_others = n_players - 1
    target_k = threshold - 1
    
    def objective(p: float) -> float:
        if p <= 0:
            return 1.0 - 0.5
        if p >= 1:
            return 0.0 - 0.5
        cdf_val = stats.binom.cdf(target_k, n_others, p)
        return cdf_val - 0.5
    
    lo, hi = 0.0, 1.0
    f_lo = objective(lo)
    f_hi = objective(hi)
    
    if f_lo * f_hi > 0:
        raise ValueError(
            f"No root in (0,1) for n={n_players}, L={threshold}. "
            f"f(0)={f_lo:.4f}, f(1)={f_hi:.4f}"
        )
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = objective(mid)
        
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    
    return (lo + hi) / 2.0


def compute_expected_attendance_under_mixed(
    n_players: int,
    p_star: float,
) -> float:
    """
    Expected attendance under symmetric mixed equilibrium.
    
    E[A] = n * p*
    
    Args:
        n_players: Total number of players.
        p_star: Equilibrium attendance probability.
    
    Returns:
        Expected number of attendees.
    """
    return n_players * p_star


def static_equilibrium_summary(
    n_players: int,
    threshold: int,
) -> dict:
    """
    Compute full static equilibrium summary for the El Farol game.
    
    Args:
        n_players: Total number of players.
        threshold: Capacity threshold.
    
    Returns:
        Dictionary with equilibrium statistics.
    """
    pure_ne_count = count_pure_ne(n_players, threshold)
    
    try:
        p_star = solve_symmetric_mixed_p_star(n_players, threshold)
        expected_attendance = compute_expected_attendance_under_mixed(n_players, p_star)
    except ValueError:
        p_star = float("nan")
        expected_attendance = float("nan")
    
    return {
        "n_players": n_players,
        "threshold": threshold,
        "pure_ne_count": pure_ne_count,
        "mixed_p_star": p_star,
        "expected_attendance_under_mixed": expected_attendance,
    }
