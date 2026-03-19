# El Farol Threshold Minority Game: Mathematical Definition

This document provides the formal game-theoretic definition of the El Farol threshold minority game implemented in this repository. The model is inspired by Arthur's (1994) inductive reasoning framework but should not be conflated with the canonical Challet–Zhang Minority Game.

## Key Design Choices

This implementation makes the following explicit design choices, which are consistent throughout the codebase:

1. **Binary actions $\{0, 1\}$:** Players choose to stay home (0) or attend (1). This differs from the canonical MG's symmetric $\{-1, +1\}$ action space.
2. **Strict threshold payoff:** Attendees win (+1) when $A < L$, and lose (-1) when $A \ge L$.
3. **Fixed stay-home payoff:** The payoff for staying home is always $0$, serving as a neutral outside option. This is a modelling assumption; alternative formulations might assign a non-zero or state-dependent payoff to the outside option.
4. **Predictor-based induction:** Agents use Arthur-inspired forecasting heuristics, not the canonical MG's lookup-table strategy space.

---

## 1. Game Definition

### 1.1 Static Normal-Form Game

We define the single-shot El Farol threshold game as a normal-form game

$$
\Gamma = \langle N, (A_i)_{i \in N}, (u_i)_{i \in N} \rangle
$$

where:

- $N = \{1, \dots, n\}$ is the set of **players** (default: $n = 101$).
- Each player $i$ chooses an **action** $a_i \in A_i = \{0, 1\}$:
  - $a_i = 1$: attend the bar.
  - $a_i = 0$: stay home.
- $L$ is the **capacity threshold** (default: $L = 60$).

**Aggregate attendance:**

$$
A(\mathbf{a}) = \sum_{i=1}^{n} a_i
$$

### 1.2 Payoff Function (Strict Threshold Convention)

This implementation uses the **strict threshold** convention:

$$
u_i(\mathbf{a}) =
\begin{cases}
+1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) < L, \\
-1 & \text{if } a_i = 1 \text{ and } A(\mathbf{a}) \ge L, \\
0 & \text{if } a_i = 0.
\end{cases}
$$

**Interpretation:**
- Attendees receive payoff $+1$ if fewer than $L$ people attend (strictly below capacity).
- Attendees receive payoff $-1$ when attendance meets or exceeds capacity.
- Staying home yields a neutral payoff of $0$.

**Remark 1 (Strict vs. weak threshold).** Some formulations use a weak inequality ($A \le L$) for the positive payoff. The choice affects equilibrium attendance by one unit. This repository uses the strict inequality ($A < L$) throughout.

**Remark 2 (Fixed outside option).** The stay-home payoff is fixed at $u_i(0) = 0$ for all players in all states. This serves as a neutral outside option and simplifies the equilibrium analysis. Alternative models might specify a positive stay-home utility (e.g., enjoying a quiet evening) or a state-dependent outside option. The choice of $u_i(0) = 0$ is a modelling assumption that must be stated explicitly when comparing results across different formulations.

---

## 2. Static Analysis

### 2.1 Pure-Strategy Nash Equilibria

**Proposition.** Under the strict threshold payoff, the pure-strategy Nash equilibria are:
- If $L = 0$: only the all-stay-home profile ($A = 0$).
- If $L \ge 1$: exactly the action profiles $\mathbf{a}$ with aggregate attendance $A(\mathbf{a}) = L - 1$.

*Proof sketch.*

1. **$A < L - 1$ is not stable (when $L \ge 1$):** Any player with $a_i = 0$ can deviate to $a_i = 1$. After deviation, $A' = A + 1 < L$, so the deviator's payoff increases from $0$ to $+1$.

2. **$A \ge L$ is not stable:** Any player with $a_i = 1$ receives payoff $-1$ and can deviate to $a_i = 0$, obtaining payoff $0$.

3. **$A = L - 1$ is stable (when $L \ge 1$):**
   - Every attendee receives $+1$ (since $A = L - 1 < L$). Deviating to stay home yields $0$, which is worse.
   - Every non-attendee receives $0$. Deviating to attend gives $A' = L \ge L$, yielding payoff $-1$, which is worse.

4. **$L = 0$ special case:** When $L = 0$, any attendance $A \ge 0$ means $A \ge L$, so all attendees lose. The only NE is all-stay-home.

**Corollary.** There are exactly $\binom{n}{L-1}$ pure-strategy Nash equilibria for $L \ge 1$, corresponding to all ways of selecting $L - 1$ players to attend. For $L = 0$, there is exactly 1 NE (all stay home).

### 2.2 Symmetric Mixed-Strategy Nash Equilibrium

Consider the symmetric mixed strategy where each player independently attends with probability $p \in (0, 1)$.

**Proposition.** For interior thresholds $2 \le L \le n$, the symmetric
mixed-strategy Nash equilibrium probability $p^* \in (0,1)$ is the unique
solution to

$$
\Pr(X \le L - 2) = \frac{1}{2}, \qquad X \sim \mathrm{Bin}(n - 1, p^*).
$$

For boundary thresholds $L = 0$ or $L = 1$, this interior mixed-equilibrium
condition does not apply (no interior solution exists).

*Proof sketch.*

For player $i$, let $X = \sum_{j \ne i} a_j \sim \mathrm{Bin}(n - 1, p)$ denote the attendance of other players.

The expected payoff from attending is:

$$
\mathbb{E}[u_i(1)] = \Pr(X + 1 < L) \cdot (+1) + \Pr(X + 1 \ge L) \cdot (-1)
$$

$$
= \Pr(X \le L - 2) - \Pr(X \ge L - 1)
$$

$$
= 2 \Pr(X \le L - 2) - 1.
$$

For indifference between attending ($\mathbb{E}[u_i(1)] = 0$) and staying home ($u_i(0) = 0$):

$$
2 \Pr(X \le L - 2) - 1 = 0 \implies \Pr(X \le L - 2) = \frac{1}{2}.
$$

**Remark.** For the default parameters $n = 101$, $L = 60$, this gives $\Pr(X \le 58) = 1/2$ where $X \sim \mathrm{Bin}(100, p^*)$. The equilibrium probability $p^*$ can be computed numerically.

---

## 3. Repeated Game

### 3.1 Formulation

The repeated version plays the same stage game over $m$ rounds (default: $m = 200$), indexed by $t = 1, \dots, m$.

**Cumulative payoff:**

$$
U_i(T) = \sum_{t=1}^{T} u_i^{(t)}
$$

where $u_i^{(t)}$ is player $i$'s stage payoff in round $t$.

**History:** At the start of round $t$, agents observe the attendance history

$$
H_t = (A_1, A_2, \dots, A_{t-1}).
$$

### 3.2 Scope and Limitations

This repository implements the repeated game as an environment for studying adaptive and inductive strategies. We make the following scope limitations explicit:

1. **No folk-theorem analysis.** We do not derive or claim subgame-perfect equilibria for the infinite-horizon or finitely-repeated versions.

2. **No discounting.** Payoffs are summed without discounting ($\delta = 1$).

3. **Adaptive, not strategic.** Agents adapt to history using heuristics; they do not engage in strategic punishment or reward schemes.

The primary purpose of the repeated framework is to study the emergent dynamics of bounded rationality and inductive learning, not to characterise equilibria of the supergame.

---

## 4. Inductive Strategies

### 4.1 Predictor-Based Adaptation

Inspired by Arthur's (1994) description of inductive reasoning, each adaptive agent maintains a finite library of attendance predictors. This implementation does not claim to replicate Arthur's exact predictor list but draws on the same conceptual framework.

**Structure:** Each agent $i$ holds a bank of $k$ predictors $\{f_{i1}, \dots, f_{ik}\}$, sampled from a master library. Before round $t$, each predictor produces a forecast:

$$
\hat{A}_{ij}(t) = f_{ij}(H_t).
$$

### 4.2 Score Updating Rules

The implementation provides two scoring rules for predictor evaluation.

**Forecast-error scoring** (used by `BestPredictorAgent`, `EpsilonGreedyPredictorAgent`):

$$
s_{ij}(t+1) = s_{ij}(t) - \lvert \hat{A}_{ij}(t) - A_t \rvert.
$$

Higher scores indicate better historical forecast accuracy. This rule asks: *which predictor forecasts attendance most accurately?*

**Virtual-payoff scoring** (used by `VirtualPayoffPredictorAgent`, `SoftmaxPredictorAgent`, `InductivePredictorAgent`, `RecencyWeightedPredictorAgent`, `TurnoverPredictorAgent`):

$$
s_{ij}(t+1) = s_{ij}(t) + \tilde{u}_{ij}(t),
$$

where $\tilde{u}_{ij}(t)$ is the payoff the predictor's implied action would have earned under the strict-threshold game payoff:

$$
\tilde{u}_{ij}(t) =
\begin{cases}
+1 & \text{if implied attend and } A_t < L, \\
-1 & \text{if implied attend and } A_t \ge L, \\
0 & \text{if implied stay home}.
\end{cases}
$$

The implied action is attend if $\hat{A}_{ij}(t) < L$, stay home otherwise. This matches the game's own payoff function, where the outside option is always neutral.

**Recency-weighted scoring** applies exponential decay to either rule:

$$
s_{ij}(t+1) = \lambda \cdot s_{ij}(t) + \Delta_j(t), \qquad \lambda \in (0, 1],
$$

where $\Delta_j(t)$ is either $-|\hat{A}_{ij}(t) - A_t|$ (forecast-error) or $\tilde{u}_{ij}(t)$ (virtual-payoff).

Lower $\lambda$ means faster forgetting of past performance, allowing quicker adaptation to regime changes.

- $\lambda = 1$: equivalent to cumulative scoring (no decay).
- $\lambda < 1$: recent performance weighted more heavily.

### 4.3 Selection Rules

**Best-predictor (hard argmax):**

$$
j_i^*(t) = \arg\max_j \, s_{ij}(t), \qquad
a_i(t) = \mathbf{1}[\hat{A}_{i j_i^*}(t) < L].
$$

Ties are broken randomly (uniform selection among equally-best predictors).

**Softmax (Boltzmann selection):**

$$
\Pr(j \mid t) = \frac{\exp(\beta \, s_{ij}(t))}{\sum_{\ell} \exp(\beta \, s_{i\ell}(t))},
$$

where $\beta \ge 0$ is the inverse temperature. The agent attends if the selected predictor forecasts attendance strictly below $L$.

- $\beta = 0$: uniform random selection (pure exploration).
- $\beta \to \infty$: hard argmax (pure exploitation).

### 4.4 Terminology Note

The predictors in this implementation are described as **Arthur-inspired** or **inductive heuristics**. We do not claim that Arthur (1994) specified a fixed canonical list of predictors. The master library (see `src/agents/predictors.py`) includes:

- Last-value extrapolation
- Contrarian (mirror) strategies
- Rolling means and medians
- Linear trend extrapolation
- Lagged-cycle predictors

These are motivated by bounded rationality and pattern-seeking behaviour, consistent with the spirit of inductive reasoning in Arthur's framework.

---

## 5. Relation to the Minority Game Literature

### 5.1 Distinction from the Canonical Minority Game

The El Farol threshold game implemented here is related to but distinct from the **canonical Minority Game** of Challet and Zhang (1997). Key differences:

| Feature | This Implementation | Canonical MG |
|---------|---------------------|--------------|
| Action space | $\{0, 1\}$ (attend/stay) | $\{-1, +1\}$ (symmetric) |
| Payoff | Threshold-based ($A < L$) | Sign of $-a_i \cdot A$ |
| Memory | Attendance history | Binary outcome history of length $M$ |
| Strategy space | Predictor library | $2^{2^M}$ lookup tables |
| Control parameter | Threshold $L$ | $\alpha = 2^M / N$ |

### 5.2 Canonical Minority Game Notation (for Reference)

For readers familiar with the Minority Game literature, we provide the standard notation:

- Actions: $a_i(t) \in \{-1, +1\}$
- Aggregate: $A(t) = \sum_{i=1}^{N} a_i(t)$
- Memory length: $M$
- Number of possible histories: $P = 2^M$
- Control parameter: $\alpha = P / N = 2^M / N$
- Volatility: $\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} A(t)^2$
- Predictability: $H = \frac{1}{P} \sum_{\mu=1}^{P} \langle A \mid \mu \rangle^2$

### 5.3 What This Implementation Does Not Include

This repository does **not** implement:

1. The standard MG strategy space (binary lookup tables over $\{0,1\}^M$).
2. The $\alpha = 2^M / N$ control parameter structure.
3. The phase transition at $\alpha_c \approx 0.34$.
4. Standard MG observables such as $\sigma^2 / N$ scaling or predictability $H$.

Any comparison to canonical MG results should be made with care. The inductive adaptation in this repo is inspired by Arthur's El Farol framework, not the Challet–Zhang MG architecture.

---

## 6. Evaluation Metrics

### 6.1 Threshold-Centred Metrics

The primary analysis metrics are centred on the threshold $L$, reflecting the El Farol model structure:

**Mean squared deviation from threshold:**

$$
\sigma_L^2 = \frac{1}{T} \sum_{t=1}^{T} (A_t - L)^2
$$

**Mean absolute deviation from threshold:**

$$
\mathrm{MAD}_L = \frac{1}{T} \sum_{t=1}^{T} |A_t - L|
$$

**Overcrowding rate:**

$$
\mathrm{OvercrowdingRate} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[A_t \ge L]
$$

Rounds with $A_t = L$ are counted as overcrowded under the strict-threshold convention.

### 6.2 Payoff Metrics

**Mean cumulative payoff:**

$$
\bar{U}(T) = \frac{1}{n} \sum_{i=1}^{n} U_i(T)
$$

**Payoff dispersion:**

$$
\mathrm{sd}_U(T) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (U_i(T) - \bar{U}(T))^2}
$$

### 6.3 Adaptation Metrics

**Predictor switch rate** (for inductive agents):

$$
\mathrm{SwitchRate} = \frac{1}{n(T-1)} \sum_{i=1}^{n} \sum_{t=2}^{T} \mathbf{1}[j_i(t) \ne j_i(t-1)]
$$

### 6.4 Note on Volatility

When "volatility" is reported, it refers to $\sigma_L^2$ (threshold-deviation volatility), **not** the canonical MG volatility $\sigma^2 = \frac{1}{T} \sum_t A_t^2$. The distinction is important: threshold-deviation volatility measures coordination efficiency around the capacity constraint, while MG volatility measures total fluctuation around zero.

---

## 7. Limitations and Scope

### 7.1 Modelling Limitations

1. **Symmetric agents.** The base model assumes homogeneous agents with identical action spaces and payoff functions. Heterogeneous experiments extend this but do not change the underlying payoff structure.

2. **No outside option dynamics.** The stay-home payoff is fixed at $0$. In reality, alternative activities may have variable attractiveness.

3. **Perfect observation.** All agents observe exact aggregate attendance. In practice, agents might have noisy or delayed information.

### 7.2 Theoretical Limitations

1. **No equilibrium selection.** With $\binom{n}{L-1}$ pure-strategy NE, we do not address which equilibrium might emerge from learning dynamics.

2. **No convergence guarantees.** The inductive strategies are heuristics without proven convergence to Nash equilibrium or correlated equilibrium.

3. **No welfare analysis.** We report payoff distributions but do not formally analyse social welfare or efficiency.

### 7.3 Claims Not Made

To avoid overclaiming, we explicitly note that this repository:

- Does **not** prove that inductive agents converge to equilibrium.
- Does **not** replicate the canonical MG phase transition.
- Does **not** claim that Arthur (1994) specified the exact predictors used here.
- Does **not** derive folk-theorem results for the repeated game.

---

## References

- Arthur, W. B. (1994). Inductive reasoning and bounded rationality. *American Economic Review*, 84(2), 406–411.
- Challet, D., & Zhang, Y.-C. (1997). Emergence of cooperation and organization in an evolutionary game. *Physica A*, 246, 407–418.
- Challet, D., Marsili, M., & Zhang, Y.-C. (2005). *Minority Games: Interacting Agents in Financial Markets*. Oxford University Press.

---

**Implementation defaults:** $n = 101$, $L = 60$, $m = 200$, seed = 42.
