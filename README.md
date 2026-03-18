# El Farol Threshold Game: Static and Repeated Simulations

Simulation code for the El Farol threshold game in normal form, with minority-game-inspired inductive extensions. Configurable player count \(n\), capacity threshold \(L\), and horizon \(m\) for the repeated game. Designed for coursework; emphasis on reproducibility and theoretical accuracy.

**Note:** This implementation is based on Arthur's (1994) El Farol problem and should not be conflated with the canonical Challet–Zhang Minority Game without explicit justification. See `docs/game_definition.md` for the precise mathematical distinction.

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

**Static (single-shot) game:**

```bash
python -m src.main static [--n_players 101] [--threshold 60] [--seed 42]
```

**Repeated game (default 200 rounds):**

```bash
python -m src.main repeated [--n_players 101] [--threshold 60] [--n_rounds 200] [--seed 42] [--output_dir outputs]
```

Outputs (CSVs and figures) are written to `--output_dir`. The repeated runner writes:

- `repeated_rounds.csv`: round-by-round attendance, overcrowding, mean round payoff
- `repeated_players.csv`: player-level cumulative payoffs
- `repeated_summary.csv`: summary metrics (threshold-centred deviation measures)
- `attendance_over_time.png`, `cumulative_average_attendance.png`

## Game Definition (Strict Threshold)

This implementation uses the **strict threshold** convention:

- Each player chooses **attend** (1) or **stay home** (0).
- Let \(A = \sum_i a_i\) be total attendance and \(L\) be the capacity threshold.
- Payoffs:
  - If \(A < L\): attendees receive \(+1\).
  - If \(A \ge L\): attendees receive \(-1\).
  - Stay home: \(0\) (neutral).

**Pure-strategy Nash equilibria:** Exactly the profiles with \(A = L - 1\). There are \(\binom{n}{L-1}\) such equilibria.

**Symmetric mixed equilibrium:** The equilibrium probability \(p^*\) satisfies \(\Pr(X \le L-2) = 1/2\) where \(X \sim \mathrm{Bin}(n-1, p^*)\).

See `docs/game_definition.md` for full definitions, proofs, and theoretical analysis.

## Inductive Strategies and Experiments

**Agents:** `RandomAgent`, `FixedAttendanceAgent`, `BestPredictorAgent` (Arthur-inspired argmax), `SoftmaxPredictorAgent` (temperature-based), `ProducerAgent` (non-adaptive noisy threshold).

**Predictor-based adaptation:** Agents maintain accuracy scores for a bank of forecasting heuristics. The master library (`src/agents/predictors.py`) includes last-value, contrarian, rolling mean/median, linear trend, and lagged-cycle predictors. These are **Arthur-inspired inductive heuristics**, not a reconstruction of any canonical predictor list.

**Experiment runners:**

```bash
python -m src.experiments.run_repeated_baselines --n_rounds 200 --output_dir outputs/baselines
python -m src.experiments.run_inductive --mode best --n_rounds 200 --output_dir outputs/inductive
python -m src.experiments.run_inductive --mode softmax --beta 1.0 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode mix --p_best 0.4 --p_softmax 0.4 --p_random 0.2 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode producer_speculator --n_producers 50 --n_rounds 200
```

Each experiment writes `rounds.csv`, `players.csv`, `summary.csv` plus figures into its `--output_dir`.

## Analysis Metrics

**Threshold-centred metrics:**
- Variance from threshold: \(\sigma_L^2 = \frac{1}{T} \sum_t (A_t - L)^2\)
- MAD from threshold: \(\mathrm{MAD}_L = \frac{1}{T} \sum_t |A_t - L|\)
- Overcrowding rate: fraction of rounds with \(A_t \ge L\)

**Payoff metrics:**
- Mean cumulative payoff
- Payoff dispersion (standard deviation across agents)

**Note:** When volatility is reported, it refers to threshold-deviation volatility (\(\sigma_L^2\)), **not** canonical Minority Game volatility.

## Relation to the Minority Game

This implementation is an **El Farol threshold game** with **minority-game-inspired** inductive adaptation. It is **not** the canonical Challet–Zhang Minority Game, which uses:

- Symmetric action space \(\{-1, +1\}\)
- Binary history of length \(M\)
- Strategy tables mapping histories to actions
- Control parameter \(\alpha = 2^M / N\)
- Phase transition at \(\alpha_c \approx 0.34\)

This repository does not implement standard MG strategy spaces, the \(\alpha\) parameter structure, or phase transition analysis. See `docs/game_definition.md` Section 5 for detailed comparison.

## Tests

```bash
pytest
# or, with verbose output:
pytest -v
```

Tests cover payoff logic, configs, population builders, experiment runners, the static game, and the repeated game.

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
│       ├── payoff.py           # Stage payoff (strict threshold)
│       ├── static_game.py      # Single-shot game
│       └── repeated_game.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_payoff.py
│   ├── test_static_game.py
│   ├── test_repeated_game.py
│   ├── test_predictors.py
│   └── test_populations.py
├── docs/
│   ├── game_definition.md      # Full mathematical definition
│   ├── report_outline.md       # Suggested report structure
│   └── genai_declaration.md
└── outputs/                    # Generated CSVs and figures (git-ignored)
```

## Licence and Use

For academic use in line with coursework instructions. See `docs/genai_declaration.md` for generative-AI disclosure.
