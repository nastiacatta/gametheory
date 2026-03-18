# Report Outline: El Farol Threshold Game

This outline provides a suggested structure for the coursework report. Section headings and content should align with the mathematical framework in `game_definition.md`.

---

## 1. Game Definition

### 1.1 Static Normal-Form Game
- Define the player set \(N = \{1, \dots, n\}\) with \(n = 101\).
- Binary action space \(A_i = \{0, 1\}\).
- Aggregate attendance \(A(\mathbf{a}) = \sum_i a_i\).
- Capacity threshold \(L = 60\).

### 1.2 Payoff Function
- State the weak threshold payoff convention explicitly:
  - \(+1\) if attend and \(A \le L\)
  - \(-1\) if attend and \(A > L\)
  - \(0\) if stay home
- Note: this differs from strict-threshold formulations (which use \(A < L\)).

### 1.3 Real-World Motivation
- El Farol bar problem (Arthur, 1994).
- Congestion games, resource allocation, coordination under uncertainty.

---

## 2. Static Analysis

### 2.1 Pure-Strategy Nash Equilibria
- State the result: NE profiles have \(A = L\).
- Provide proof sketch:
  - \(A < L\): stay-home player has profitable deviation.
  - \(A > L\): attendee has profitable deviation.
  - \(A = L\): no player has profitable deviation.
- Count: \(\binom{n}{L}\) equilibria.

### 2.2 Symmetric Mixed-Strategy Equilibrium
- Derive the indifference condition: \(\Pr(X \le L - 1) = 1/2\) where \(X \sim \mathrm{Bin}(n-1, p)\).
- Discuss numerical solution for default parameters.
- Interpret expected attendance under mixed equilibrium.

---

## 3. Repeated Game

### 3.1 Formulation
- Horizon \(m = 200\) rounds.
- Cumulative payoff \(U_i(T) = \sum_{t=1}^T u_i^{(t)}\).
- History observable: \(H_t = (A_1, \dots, A_{t-1})\).

### 3.2 Scope and Limitations
- No discounting.
- No folk-theorem analysis or subgame-perfect equilibrium claims.
- Focus: emergent dynamics under adaptive heuristics, not equilibrium characterisation.

---

## 4. Inductive Strategies

### 4.1 Predictor-Based Framework
- Each agent holds \(k\) predictors sampled from a master library.
- Forecasts: \(\hat{A}_{ij}(t) = f_{ij}(H_t)\).
- Score updating: \(s_{ij}(t+1) = s_{ij}(t) - |\hat{A}_{ij}(t) - A_t|\).

### 4.2 Selection Mechanisms
- **Best-predictor (argmax):** deterministic selection of highest-scoring predictor.
- **Softmax:** stochastic selection with inverse temperature \(\beta\).

### 4.3 Arthur-Inspired Heuristics
- Terminology: use "Arthur-inspired predictors" or "inductive heuristics."
- Do not claim Arthur (1994) specifies a canonical predictor list.
- Describe predictor types: last-value, contrarian, rolling mean, linear trend, lagged cycle.

---

## 5. Relation to Minority Game Literature

### 5.1 Distinction from Canonical Minority Game
- Clearly state that this is **not** the canonical Challet–Zhang Minority Game.
- Table comparing: action space, payoff structure, memory, strategy space, control parameter.

### 5.2 Canonical MG Overview (for Context)
- Actions \(a_i \in \{-1, +1\}\), aggregate \(A = \sum_i a_i\).
- Memory length \(M\), history space \(P = 2^M\).
- Control parameter \(\alpha = 2^M / N\).
- Phase transition at \(\alpha_c \approx 0.34\).

### 5.3 What This Implementation Does Not Include
- Standard MG strategy tables.
- \(\alpha\) parameter structure.
- Phase transition analysis.
- Canonical MG volatility scaling.

---

## 6. Implementation and Simulation

### 6.1 Model Choices
- Parameters: \(n = 101\), \(L = 60\), \(m = 200\).
- Reproducibility: seeded randomness.
- Agent types: RandomAgent, FixedAttendanceAgent, BestPredictorAgent, SoftmaxPredictorAgent, ProducerAgent.

### 6.2 Experiments
- Baseline: all-random agents.
- Homogeneous inductive: all best-predictor or all softmax.
- Heterogeneous populations: mixtures, producer-speculator.

### 6.3 Outputs
- Round-level data: attendance, overcrowding, mean payoff.
- Player-level data: cumulative payoffs.
- Summary metrics.

---

## 7. Evaluation Metrics

### 7.1 Threshold-Centred Metrics
- Variance from threshold: \(\sigma_L^2 = \frac{1}{T} \sum_t (A_t - L)^2\).
- MAD from threshold: \(\mathrm{MAD}_L = \frac{1}{T} \sum_t |A_t - L|\).
- Overcrowding rate.

### 7.2 Payoff Metrics
- Mean cumulative payoff: \(\bar{U}(T)\).
- Payoff dispersion: \(\mathrm{sd}_U(T)\).
- Min/max cumulative payoff.

### 7.3 Adaptation Metrics
- Predictor switch rate (for inductive agents).

### 7.4 Volatility Clarification
- Distinguish threshold-deviation volatility (\(\sigma_L^2\)) from canonical MG volatility (\(\sigma^2 = \frac{1}{T} \sum_t A_t^2\)).

---

## 8. Analysis of Simulation Outcomes

### 8.1 Baseline (Non-Adaptive) Results

**Figures to include** (from `outputs/baselines/`):
- `attendance_over_time.png`: Attendance per round (expected: fluctuating around ~79, always above threshold)
- `cumulative_average_attendance.png`: Cumulative mean settling near 78.6

**Key observations:**
- Mean attendance ~78.6, far above threshold \(L = 60\)
- Overcrowding rate = 100% (non-adaptive population always overcrowds)
- Mean cumulative payoff strongly negative (~-156)
- This population cannot coordinate because agents do not adapt to history

### 8.2 Inductive (Adaptive) Results

**Figures to include** (from `outputs/inductive_best/` and `outputs/inductive_softmax/`):
- `attendance.png`: Attendance per round (expected: oscillating around \(L = 60\))
- `cum_avg_attendance.png`: Cumulative mean converging toward threshold
- `predictor_share.png`: Predictor usage over time
- `payoff_hist.png`: Distribution of cumulative payoffs

**Key observations:**
- Mean attendance ~60 (close to threshold \(L\))
- Overcrowding rate ~50% (much improved coordination)
- Mean cumulative payoff much higher (~-40) than baseline
- Variance remains high due to adaptive oscillation around threshold

### 8.3 Baseline vs Inductive Comparison

| Metric | Baseline | Best-Predictor | Softmax |
|--------|----------|----------------|---------|
| Mean attendance | 78.6 | 59.8 | 60.0 |
| Overcrowding rate | 100% | 53.5% | 51.5% |
| Mean cumulative payoff | -155.7 | -40.5 | -38.3 |
| Variance from threshold | 359.2 | 387.6 | 455.2 |

**Discussion:**
- Adaptive agents converge toward the threshold attendance (\(L = 60\)), matching Arthur's prediction
- Non-adaptive populations consistently overcrowd, demonstrating coordination failure
- Higher variance for inductive runs reflects the oscillatory nature of adaptive coordination
- Softmax selection achieves slightly better payoffs due to exploration-exploitation balance

### 8.4 Coordination Efficiency
- Overcrowding frequency: compare 100% (baseline) vs ~50% (inductive)
- Utilisation: fraction of capacity used when not overcrowded

### 8.5 Fairness and Dispersion
- Payoff inequality across agents
- Do some agents consistently outperform?

### 8.6 Inductive Dynamics
- Predictor switching patterns (see `predictor_share.png`)
- Dominant predictors over time

---

## 9. Real-World Applications

- Traffic routing and congestion.
- Server load balancing.
- Event attendance prediction.
- Market timing decisions.

Discuss how the model abstracts these scenarios and what insights transfer.

---

## 10. Quality and Synthesis

### 10.1 Limitations
- Symmetric agents, homogeneous payoffs.
- No equilibrium convergence guarantees.
- No welfare optimality analysis.
- Fixed threshold (no dynamic capacity).

### 10.2 Assumptions
- Perfect observation of aggregate attendance.
- Rational adaptation within bounded heuristic space.
- No communication or collusion.

### 10.3 Main Takeaways
- Summarise key findings from simulations.
- Contrast static equilibrium predictions with adaptive dynamics.
- Discuss the gap between theoretical equilibrium and heuristic coordination.

### 10.4 Future Directions
- Variable thresholds.
- Network effects.
- More sophisticated learning (reinforcement learning, Bayesian updating).
- Empirical validation against real congestion data.

---

## Appendix Suggestions

- A. Predictor library details (from `src/agents/predictors.py`).
- B. Full metric definitions.
- C. Selected raw outputs and figures.

---

## Figure and Output Directory Reference

| Experiment | Output Directory | Key Figures |
|------------|------------------|-------------|
| Baseline (non-adaptive) | `outputs/baselines/` | `attendance_over_time.png`, `cumulative_average_attendance.png` |
| Best-predictor inductive | `outputs/inductive_best/` | `attendance.png`, `cum_avg_attendance.png`, `predictor_share.png`, `payoff_hist.png` |
| Softmax inductive | `outputs/inductive_softmax/` | `attendance.png`, `cum_avg_attendance.png`, `predictor_share.png`, `payoff_hist.png` |
| Heterogeneous (producer-speculator) | `outputs/heterogeneous/` | `attendance.png`, `cum_avg_attendance.png`, `payoff_hist.png` |

**Regeneration commands:**
```bash
python -m src.experiments.run_repeated_baselines --n_rounds 200 --output_dir outputs/baselines
python -m src.experiments.run_inductive --mode best --n_rounds 200 --output_dir outputs/inductive_best
python -m src.experiments.run_inductive --mode softmax --beta 1.0 --n_rounds 200 --output_dir outputs/inductive_softmax
python -m src.experiments.run_heterogeneous --mode producer_speculator --n_producers 50 --n_rounds 200 --output_dir outputs/heterogeneous
```
