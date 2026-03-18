# Minority game definition

## Static normal-form version

We define the single-shot minority game as a normal-form game

\[
\Gamma = \langle N, (A_i)_{i \in N}, (u_i)_{i \in N} \rangle
\]

where:

- \(N = \{1,\dots,n\}\) is the set of **players**.
- Each player \(i\) chooses an **action** \(a_i \in A_i = \{0,1\}\):
  - \(a_i = 1\): attend the bar.
  - \(a_i = 0\): stay home.
- A **pure strategy** for player \(i\) is the action chosen at the sole information set of this simultaneous game.

Let aggregate attendance be

\[
A(\mathbf{a}) = \sum_{i=1}^{n} a_i
\]

and let \(L\) be the attendance threshold (bar capacity).

### Payoff convention (Arthur-style strict threshold)

The payoff for player \(i\) is
\[
u_i(\mathbf{a})=
\begin{cases}
+1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) < L, \\
-1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) \ge L, \\
0 & \text{if } a_i = 0.
\end{cases}
\]

This matches Arthur's El Farol statement: the evening is enjoyable only when attendance is strictly below the capacity threshold.

### Pure-strategy Nash equilibria

Under this payoff, the unique pure-strategy Nash equilibrium attendance level is
\[
A = L - 1.
\]

- If \(A < L-1\), a stay-home player can deviate to attend and move from \(0\) to \(+1\).
- If \(A \ge L\), an attending player can deviate to stay home and move from \(-1\) to \(0\).
- If \(A = L-1\), no attendee wants to switch from \(+1\) to \(0\), and no non-attendee wants to switch from \(0\) to \(-1\).

There are \(\binom{n}{L-1}\) such profiles.

### Symmetric mixed-strategy equilibrium

Under the symmetric mixed strategy where all players use the same \(p\), let
\[
X \sim \mathrm{Bin}(n-1, p).
\]

Indifference between attending and staying home requires
\[
\mathbb{E}[u_i(1)] = 0,
\]
which gives
\[
\Pr(X \le L-2) = \tfrac{1}{2}.
\]

This determines the symmetric mixed-strategy Nash equilibrium.

## Repeated game

The repeated version uses the same stage game over \(m\) rounds \(t=1,\dots,m\), with cumulative payoff

\[
U_i(T) = \sum_{t=1}^{T} u_i^{(t)}.
\]

Each round, agents observe past attendance history \(H_t = (A_1,\dots,A_{t-1})\) and may adapt their decisions through inductive strategies (see below).

## Inductive strategies (Arthur-inspired predictor-based adaptation)

Each agent \(i\) holds a bank of \(k\) attendance predictors \(f_{ij}\), sampled from a fixed master library. Before round \(t\), each predictor produces a forecast:

\[
\hat{A}_{ij}(t) = f_{ij}(H_t).
\]

The agent maintains a cumulative accuracy score for each predictor:

\[
s_{ij}(t+1) = s_{ij}(t) - |\hat{A}_{ij}(t) - A_t|.
\]

**Implementation note:** This implementation uses a fixed master library of predictors (see `src/agents/predictors.py`). Each adaptive agent is assigned \(k\) predictors sampled without replacement from this master library. This introduces heterogeneity across agents while maintaining reproducibility through seeded sampling.

**Strategy A (best-predictor):**
\[
j_i^*(t) = \arg\max_j \, s_{ij}(t), \qquad
a_i(t) = \mathbf{1}[\hat{A}_{ij_i^*}(t) < L].
\]

**Strategy B (softmax / temperature-based):**
\[
\Pr(j \mid t) = \frac{e^{\beta \, s_{ij}(t)}}{\sum_{\ell} e^{\beta \, s_{i\ell}(t)}},
\]
then the agent attends iff the chosen predictor forecasts a value \(< L\).

## Key observables

For the repeated game, the main analysis metrics are:

\[
\sigma_L^2 = \frac{1}{T}\sum_{t=1}^{T}(A_t - L)^2, \qquad
\mathrm{MAD}_L = \frac{1}{T}\sum_{t=1}^{T} |A_t - L|,
\]

\[
\mathrm{OvercrowdingRate} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[A_t \ge L],
\]

\[
\bar{U}(T) = \frac{1}{n}\sum_{i=1}^{n} U_i(T), \qquad
\mathrm{sd}_U(T) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(U_i(T) - \bar{U}(T))^2},
\]

\[
\mathrm{SwitchRate} = \frac{1}{n(T-1)}\sum_{i=1}^{n}\sum_{t=2}^{T}\mathbf{1}[j_i(t) \neq j_i(t-1)].
\]

In this repo, SwitchRate is computed for inductive experiments that store predictor histories; it is not part of the base repeated_summary.csv exported by RepeatedGameResult.summary().

This project uses the El Farol threshold version with default values \(n=101\), \(L=60\), and \(m=200\).
