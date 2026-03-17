# Minority Game: Static and Repeated Simulators

Simulation code for the El Farol / threshold minority game in normal-form: configurable player count *n*, capacity threshold *L*, and round count *m* for the repeated game. Implemented for coursework; designed for reproducibility and extension (e.g. inductive strategies).

## Requirements

- **Python**: 3.9 or higher (tested on 3.10).
- **Dependencies**: See `requirements.txt`; install with `pip install -r requirements.txt`.

## Reproducibility

- All randomness is driven by a **seed** (default `42`). Use `--seed` to reproduce runs.
- Dependency versions use bounded ranges in `requirements.txt` (e.g. `numpy>=1.24,<3`) for consistent environments; not strictly pinned to exact versions.
- Run from the **project root** so that `src` is on the module path (e.g. `python -m src.main`).

## Quickstart (clean run)

```bash
# From the project root:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Run tests (should pass)
python -m pytest
```

## Running the code

The default coursework-style parameters are **n=101**, **L=60**, **m=200**.

Notes:
- **Odd n**: the game engines enforce odd `--n_players` (typical in minority-game setups).
- **A = L convention**: attendees are treated as “not overcrowded” when \(A \le L\).

**Static (single-shot) game:**

```bash
python -m src.main static [--n_players 101] [--threshold 60] [--seed 42]
```

**Repeated game (default 200 rounds):**

```bash
python -m src.main repeated [--n_players 101] [--threshold 60] [--n_rounds 200] [--seed 42] [--output_dir outputs]
```

Outputs (CSVs and figures) are written to `--output_dir`; the directory is created if it does not exist. The repeated runner writes:

- `repeated_rounds.csv`: round-by-round attendance, overcrowding, mean round payoff
- `repeated_players.csv`: player-level cumulative payoffs
- `repeated_summary.csv`: summary metrics (including threshold-centred deviation measures)
- `attendance_over_time.png`, `cumulative_average_attendance.png`

## Tests

From the project root:

```bash
pytest
# or, with verbose output:
pytest -v
```

Tests cover payoff logic, configs, predictors, inductive agents, metrics, the static game, and the repeated game.

## Project layout

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
│   │   ├── predictors.py       # Arthur predictor library
│   │   ├── best_predictor_agent.py
│   │   ├── softmax_predictor_agent.py
│   │   └── producer_agent.py
│   ├── analysis/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── export.py
│   ├── experiments/
│   │   ├── populations.py
│   │   ├── run_repeated_baselines.py
│   │   ├── run_inductive.py
│   │   └── run_heterogeneous.py
│   └── game/
│       ├── payoff.py      # Payoff rules (threshold formulation)
│       ├── static_game.py # Single-shot game
│       └── repeated_game.py
├── tests/
│   ├── conftest.py
│   ├── test_payoff.py
│   ├── test_static_game.py
│   ├── test_repeated_game.py
│   ├── test_predictors.py
│   ├── test_inductive_agents.py
│   └── test_metrics.py
├── docs/
│   ├── game_definition.md
│   ├── report_outline.md
│   └── genai_declaration.md
└── outputs/               # Generated CSVs and figures (git-ignored)
```

## Game definition (lecture-consistent)

- Each player chooses **attend** (1) or **stay home** (0).
- *A* = total attendance. *L* = capacity threshold.
- If *A* ≤ *L*: attendees receive payoff +1.
- If *A* > *L*: attendees receive payoff −1.
- Stay home: payoff 0 (neutral).

**Report requirement:** The brief allows consistent definitions but does not force a unique stay-home utility. You must state explicitly in the report that stay-home payoff is 0 (neutral); otherwise the marker may assume a different formulation.

See `docs/game_definition.md` for the full normal-form definition, Nash equilibria, and inductive strategies.

## Inductive strategies and experiments

**Agents:** `RandomAgent`, `FixedAttendanceAgent`, `BestPredictorAgent` (Arthur-style), `SoftmaxPredictorAgent` (temperature-based), `ProducerAgent` (non-adaptive noisy threshold).

**Experiment runners:**

```bash
python -m src.experiments.run_repeated_baselines --n_rounds 200 --output_dir outputs/baselines
python -m src.experiments.run_inductive --mode best --n_rounds 200 --output_dir outputs/inductive
python -m src.experiments.run_inductive --mode softmax --beta 1.0 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode mix --p_best 0.5 --p_softmax 0.5 --n_rounds 200
python -m src.experiments.run_heterogeneous --mode producer_speculator --n_producers 50 --n_rounds 200
```

Each experiment runner writes `rounds.csv`, `players.csv`, `summary.csv` plus figures into its `--output_dir`.

## Licence and use

For academic use in line with the coursework instructions. See `docs/genai_declaration.md` for generative-AI disclosure.
