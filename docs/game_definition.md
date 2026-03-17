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

### Payoff convention (lecture-consistent)

The payoff for player \(i\) is

\[
u_i(\mathbf{a})=
\begin{cases}
+1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) \le L, \\
-1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) > L, \\
 0 & \text{if } a_i = 0.
\end{cases}
\]

This follows the formulation in Lecture 15: attendees are happy when the bar is not crowded and unhappy when it is overcrowded; staying home yields a neutral payoff of zero.

### Modelling choice: \(A = L\) convention

We adopt \(A \le L\) as the condition under which attendees are happy. This is a modelling choice that is stated explicitly here and implemented consistently throughout the code.

### Pure-strategy Nash equilibria

Under this payoff, the unique pure-strategy Nash equilibrium attendance level is \(A = L\):

- If \(A < L\), a stay-home player can deviate to attend and move from \(0\) to \(+1\).
- If \(A > L\), an attending player can deviate to stay home and move from \(-1\) to \(0\).
- If \(A = L\), no attendee wants to switch from \(+1\) to \(0\), and no non-attendee wants to switch from \(0\) to \(-1\).

There are \(\binom{n}{L}\) such profiles (any subset of exactly \(L\) players attending), but they all share the same attendance level.

### Symmetric mixed-strategy equilibrium

A mixed strategy for player \(i\) is a probability distribution over \(\{0,1\}\):

\[
\sigma_i(1)=p_i, \qquad \sigma_i(0)=1-p_i, \qquad p_i \in [0,1].
\]

Under the symmetric mixed strategy where all players use the same \(p\), let \(X \sim \mathrm{Bin}(n-1, p)\). Indifference between attending and staying home requires \(\mathbb{E}[u_i(1)] = 0\), which gives

\[
\Pr(X \le L-1) = \tfrac{1}{2}.
\]

This determines the unique symmetric mixed-strategy Nash equilibrium.

## Repeated game

The repeated version uses the same stage game over \(m\) rounds \(t=1,\dots,m\), with cumulative payoff

\[
U_i(T) = \sum_{t=1}^{T} u_i^{(t)}.
\]

Each round, agents observe past attendance history \(H_t = (A_1,\dots,A_{t-1})\) and may adapt their decisions through inductive strategies (see below).

## Inductive strategies (Arthur 1994)

Each agent \(i\) holds a bank of \(k\) attendance predictors \(f_{ij}\). Before round \(t\), each predictor produces a forecast:

\[
\hat{A}_{ij}(t) = f_{ij}(H_t).
\]

The agent maintains a cumulative accuracy score for each predictor:

\[
s_{ij}(t+1) = s_{ij}(t) - |\hat{A}_{ij}(t) - A_t|.
\]

**Strategy A (best-predictor):** The agent uses the predictor with the highest score:

\[
j_i^*(t) = \arg\max_j \, s_{ij}(t), \qquad a_i(t) = \mathbf{1}[\hat{A}_{ij_i^*}(t) \le L].
\]

**Strategy B (softmax / temperature-based):** The active predictor is chosen stochastically:

\[
\Pr(j \mid t) = \frac{e^{\beta \, s_{ij}(t)}}{\sum_{\ell} e^{\beta \, s_{i\ell}(t)}},
\]

then the agent attends if the chosen predictor's forecast \(\le L\). The parameter \(\beta\) controls exploration (\(\beta \to 0\)) versus exploitation (\(\beta \to \infty\)).

## Key observables

For the repeated game, the main analysis metrics are:

\[
\sigma_L^2 = \frac{1}{T}\sum_{t=1}^{T}(A_t - L)^2, \qquad
\mathrm{MAD}_L = \frac{1}{T}\sum_{t=1}^{T} |A_t - L|,
\]

\[
\mathrm{OvercrowdingRate} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[A_t > L],
\]

\[
\bar{U}(T) = \frac{1}{n}\sum_{i=1}^{n} U_i(T), \qquad
\mathrm{sd}_U(T) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(U_i(T) - \bar{U}(T))^2},
\]

\[
\mathrm{SwitchRate} = \frac{1}{n(T-1)}\sum_{i=1}^{n}\sum_{t=2}^{T}\mathbf{1}[j_i(t) \neq j_i(t-1)].
\]

This project uses the El Farol threshold version with default values \(n=101\), \(L=60\), and \(m=200\).
