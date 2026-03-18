"""
Theoretical benchmarks for the threshold El Farol game.

Provides closed-form expressions for i.i.d. random play and symmetric
mixed-strategy equilibrium, enabling comparison between learned strategies
and theory.
"""

from __future__ import annotations

from math import comb


def expected_iid_attendance(n_players: int, p_attend: float) -> float:
    """E[A_t] = n * p for i.i.d. Bernoulli(p) attendance."""
    return n_players * p_attend


def expected_iid_threshold_mse(
    n_players: int, p_attend: float, threshold: int
) -> float:
    """
    E[(A_t - L)^2] for i.i.d. Bernoulli(p) attendance.

    Decomposition: E[(A - L)^2] = Var(A) + (E[A] - L)^2
                                = np(1-p) + (np - L)^2

    When p = L/n (neutral baseline), this simplifies to np(1-p).
    """
    mu = n_players * p_attend
    var = n_players * p_attend * (1.0 - p_attend)
    return var + (mu - threshold) ** 2


def expected_iid_overcrowding_rate(
    n_players: int, p_attend: float, threshold: int
) -> float:
    """
    P(A_t > L) for i.i.d. Bernoulli(p) attendance.

    Computes sum_{k=L+1}^{n} C(n,k) p^k (1-p)^{n-k}.
    """
    return sum(
        comb(n_players, k) * (p_attend**k) * ((1.0 - p_attend) ** (n_players - k))
        for k in range(threshold + 1, n_players + 1)
    )


def symmetric_mixed_equilibrium_p(
    n_players: int, threshold: int, tol: float = 1e-12
) -> float:
    """
    Symmetric mixed Nash equilibrium probability for the threshold game.

    At equilibrium, attending must be indifferent to staying home (payoff 0).
    The expected payoff from attending is:
        E[u | attend] = P(not overcrowded) * (+1) + P(overcrowded) * (-1)
                      = 2 * P(A_{-i} < L) - 1

    Setting this to 0 gives the equilibrium condition:
        sum_{k=0}^{L-1} C(n-1, k) p^k (1-p)^{n-1-k} = 1/2

    Uses bisection to find p satisfying this condition.

    Args:
        n_players: Total number of players (n >= 2)
        threshold: Capacity threshold L in [0, n-1]
        tol: Convergence tolerance for bisection

    Returns:
        Equilibrium probability p*
    """
    if n_players < 2:
        raise ValueError("n_players must be at least 2")
    if not (0 <= threshold <= n_players - 1):
        raise ValueError("threshold must be in [0, n_players - 1]")

    def attend_payoff(p: float) -> float:
        """Expected payoff from attending when others play Bernoulli(p)."""
        prob_not_overcrowded = sum(
            comb(n_players - 1, k) * (p**k) * ((1.0 - p) ** (n_players - 1 - k))
            for k in range(threshold)
        )
        return 2.0 * prob_not_overcrowded - 1.0

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if attend_payoff(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
