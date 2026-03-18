# El Farol Threshold Minority Game: Static, Repeated, and Inductive Simulators

Simulation code for the **El Farol threshold game** with minority-game-inspired inductive extensions. This repository implements a threshold-based coordination game where:

1. **Actions are binary (0/1):** Each player chooses to **stay home** (0) or **go to the bar** (1).
2. **Payoff rule is threshold-based with the weak convention:** Attendees receive +1 if total attendance $A \le L$, and −1 if $A > L$. Staying home always yields 0.
3. **Arthur-inspired but not canonical MG:** This implementation draws on Arthur's (1994) El Farol problem and uses predictor-based inductive reasoning. It is **not** the canonical Challet–Zhang Minority Game, which uses symmetric actions $\{-1, +1\}$, binary history strings of length $M$, lookup-table strategies, and the control parameter $\alpha = 2^M / N$.

Configurable parameters: player count $n$ (default 101), capacity threshold $L$ (default 60), and horizon $m$ (default 200 rounds). Designed for coursework; emphasis on reproducibility and theoretical accuracy.

See `docs/game_definition.md` for the full mathematical definition and distinction from the canonical Minority Game.

## Requirements

- **Python**: 3.9 or higher (tested on 3.10).
- **Dependencies**: See `requirements.txt`; install with `pip install -r requirements.txt`.

## Reproducibility

- All randomness is driven by a **seed** (default `42`). Use `--seed` to reproduce runs.
- Dependency versions use bounded ranges in `requirements.txt` for consistent environments.
- Run from the **project root** so that `src` is on the module path (e.g., `python -m src.main`).

## Quickstart

```bash
# From the project root:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Run tests (should pass)
python -m pytest
```

## Running the Code

Default coursework parameters: **n = 101**, **L = 60**, **m = 200**.

### Unified CLI

All experiments can be run through the unified CLI with subcommands:

**Static (single-shot) game:**

```bash
python -m src.main static [--n_players 101] [--threshold 60] [--seed 42]
```

**Static probability sweep (p from 0 to 1):**

```bash
python -m src.main static-sweep --n_players 101 --threshold 60 --n_samples 10000 --grid_size 201 --seed 42
```

This sweeps attendance probability $p \in [0, 1]$ and outputs:
- `static_probability_sweep.csv`: mean attendance, payoff, overcrowding rate for each p
- `static_payoff_vs_p.png`: mean payoff per player vs p
- `static_attendance_vs_p.png`: mean attendance vs p
- `static_overcrowding_vs_p.png`: overcrowding rate vs p

**Repeated game with basic populations:**

```bash
python -m src.main repeated [--n_players 101] [--threshold 60] [--n_rounds 200] [--seed 42] [--output_dir outputs/repeated]
```

**Inductive agents (predictor-based):**

```bash
# Best-predictor (hard argmax)
python -m src.main inductive --mode best --n_rounds 200

# Softmax selection with temperature
python -m src.main inductive --mode softmax --beta 1.0 --n_rounds 200

# Recency-weighted with exponential forgetting
python -m src.main inductive --mode recency --lambda_decay 0.95 --n_rounds 200
```

**Heterogeneous populations:**

```bash
# Mixed best/softmax/random
python -m src.main heterogeneous --mode mix --p_best 0.4 --p_softmax 0.4 --p_random 0.2

# Producer/speculator split
python -m src.main heterogeneous --mode producer_speculator --n_producers 50
```

**Multi-seed parameter sweep:**

```bash
python -m src.main sweep --n_seeds 50 --n_rounds 200 1000 --output_dir outputs/sweep
```

### Additional Experiment Runners

```bash
# Static equilibrium theory (pure NE count, mixed p*)
python -m src.experiments.run_static_theory --n_players 101 --threshold 60

# Case study with real-world-inspired agent types
python -m src.experiments.run_case_study --n_rounds 200 --output_dir outputs/case_study
```

Outputs (CSVs and figures) are written to `--output_dir`. The repeated runner writes:

- `rounds.csv`: round-by-round attendance, deviations, overcrowding, mean payoff, cumulative metrics
- `players.csv`: player-level cumulative payoffs, agent types, attend rates
- `summary.csv`: summary metrics (threshold-centred deviation measures)
- Plots: `attendance.png`, `attendance_deviation.png`, `cum_avg_attendance.png`, etc.

## Game Definition (Weak Threshold)

This implementation uses the **weak threshold** convention:

- Each player chooses **attend** (1) or **stay home** (0).
- Let $A = \sum_i a_i$ be total attendance and $L$ be the capacity threshold.
- Payoffs:
  - If $A \le L$: attendees receive $+1$.
  - If $A > L$: attendees receive $-1$.
  - Stay home: $0$ (neutral).

**Pure-strategy Nash equilibria:** Exactly the profiles with $A = L$. There are $\binom{n}{L}$ such equilibria.

**Symmetric mixed equilibrium:** The equilibrium probability $p^*$ satisfies $\Pr(X \le L-1) = 1/2$ where $X \sim \mathrm{Bin}(n-1, p^*)$.

See `docs/game_definition.md` for full definitions, proofs, and theoretical analysis.

## Inductive Strategies and Experiments

### Agent Types

**Basic agents:**
- `RandomAgent`: Independent random choice with fixed probability
- `FixedAttendanceAgent`: Fixed prediction-based decision
- `ProducerAgent`: Non-adaptive noisy threshold predictor

**Predictor-based inductive agents:**
- `BestPredictorAgent`: Hard argmax over predictor scores (Arthur-inspired)
- `SoftmaxPredictorAgent`: Boltzmann selection with temperature parameter β
- `RecencyWeightedPredictorAgent`: Exponential score decay for faster adaptation to regime changes

### Predictor Library

Agents maintain accuracy scores for a bank of forecasting heuristics. The master library (`src/agents/predictors.py`) includes:
- Last-value extrapolation
- Contrarian (mirror) strategies
- Rolling means and medians
- Linear trend extrapolation
- Lagged-cycle predictors

These are **Arthur-inspired inductive heuristics**, not a reconstruction of any canonical predictor list.

### Score Update Rules

**Cumulative scoring (Best, Softmax):**

$$s_{ij}(t+1) = s_{ij}(t) - |\hat{A}_{ij}(t) - A_t|$$

**Recency-weighted scoring (Recency):**

$$s_{ij}(t+1) = \lambda \cdot s_{ij}(t) - |\hat{A}_{ij}(t) - A_t|, \quad \lambda \in (0,1]$$

Lower λ = faster forgetting of past performance.

### Experiment Runners

```bash
python -m src.experiments.run_repeated_baselines --n_rounds 200 --output_dir outputs/baselines
python -m src.experiments.run_inductive --mode best --n_rounds 200 --output_dir outputs/inductive
python -m src.experiments.run_inductive --mode softmax --beta 1.0 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode mix --p_best 0.4 --p_softmax 0.4 --p_random 0.2 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode producer_speculator --n_producers 50 --n_rounds 200
python -m src.experiments.run_case_study --output_dir outputs/case_study
```

Each experiment writes `rounds.csv`, `players.csv`, `summary.csv` plus figures into its `--output_dir`.

## Analysis Metrics

**Threshold-centred metrics:**
- Mean squared deviation from threshold: $\sigma_L^2 = \frac{1}{T} \sum_t (A_t - L)^2$
- MAD from threshold: $\mathrm{MAD}_L = \frac{1}{T} \sum_t |A_t - L|$
- Overcrowding rate: fraction of rounds with $A_t > L$

**Payoff metrics:**
- Mean cumulative payoff
- Payoff dispersion (standard deviation across agents)

**Note:** When volatility is reported, it refers to threshold-deviation volatility ($\sigma_L^2$), **not** canonical Minority Game volatility.

## Relation to the Minority Game

This implementation is an **El Farol threshold game** with **minority-game-inspired** inductive adaptation. It is **not** the canonical Challet–Zhang Minority Game, which uses:

- Symmetric action space $\{-1, +1\}$
- Binary history of length $M$
- Strategy tables mapping histories to actions
- Control parameter $\alpha = 2^M / N$
- Phase transition at $\alpha_c \approx 0.34$

This repository does not implement standard MG strategy spaces, the $\alpha$ parameter structure, or phase transition analysis. See `docs/game_definition.md` Section 5 for detailed comparison.

## Tests

```bash
pytest
# or, with verbose output:
pytest -v
```

Tests cover configuration, payoff logic, population builders, the static game,
repeated game, predictors, inductive agents (including threshold and advanced variants),
metrics, equilibria calculations, and static probability sweeps.

## Project Layout

```
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── src/
│   ├── __init__.py
│   ├── config.py          # StaticGameConfig, RepeatedGameConfig
│   ├── main.py            # CLI: static | repeated
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── fixed_attendance_agent.py
│   │   ├── predictors.py       # Arthur-inspired predictor library
│   │   ├── best_predictor_agent.py
│   │   ├── softmax_predictor_agent.py
│   │   └── producer_agent.py
│   ├── analysis/
│   │   ├── metrics.py          # Threshold-centred and payoff metrics
│   │   ├── plots.py
│   │   └── export.py
│   ├── experiments/
│   │   ├── populations.py
│   │   ├── run_repeated_baselines.py
│   │   ├── run_inductive.py
│   │   └── run_heterogeneous.py
│   └── game/
│       ├── payoff.py           # Stage payoff (weak threshold)
│       ├── static_game.py      # Single-shot game
│       └── repeated_game.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_payoff.py
│   ├── test_static_game.py
│   ├── test_repeated_game.py
│   ├── test_predictors.py
│   ├── test_populations.py
│   ├── test_metrics.py
│   ├── test_inductive_agents.py
│   ├── test_equilibria.py
│   ├── test_threshold_agents.py
│   ├── test_advanced_agents.py
│   ├── test_fixed_predictor_agent.py
│   ├── test_math_consistency.py
│   └── test_static_probability_sweep.py
├── docs/
│   ├── game_definition.md      # Full mathematical definition
│   ├── report_outline.md       # Suggested report structure
│   └── genai_declaration.md
└── outputs/                    # Generated CSVs and figures (git-ignored)
```

## Licence and Use

For academic use in line with coursework instructions. See `docs/genai_declaration.md` for generative-AI disclosure.
