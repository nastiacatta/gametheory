import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# IMPORT YOUR PREDICTORS HERE
# ============================================================
from predictors import (
    same_as_last_week,
    same_as_2_weeks_ago,
    same_as_3_weeks_ago,
    same_as_5_weeks_ago,
    moving_average_2,
    moving_average_3,
    moving_average_4,
    moving_average_5,
    moving_average_8,
    mean_of_all_history,
    mirror_around_50_last,
    mirror_around_60_last,
    mirror_around_capacity_last,
    linear_trend_2,
    linear_trend_3,
    linear_trend_5,
    repeat_last_change,
    median_of_last_3,
    median_of_last_5,
    min_of_last_3,
    max_of_last_3,
    weighted_average_recent,
    weighted_average_recent_5,
    alternating_cycle_2,
    alternating_cycle_3,
    bounded_trend_last_8,
    revert_to_mean_all_history,
    revert_to_capacity,
    average_of_last_and_5_ago,
    average_of_last_2_and_last_5,
    contrarian_last,
    exponential_smoothing,
)

# ============================================================
# PARAMETERS
# ============================================================
N_PLAYERS = 1001
BAR_CAPACITY = 600
N_ROUNDS = 200
INITIAL_HISTORY = [0]
FALLBACK_GO_PROBABILITY = 0.50
RANDOM_SEED = 42

PREDICTOR_POOL = [
    same_as_last_week,
    same_as_2_weeks_ago,
    same_as_3_weeks_ago,
    same_as_5_weeks_ago,
    moving_average_2,
    moving_average_3,
    moving_average_4,
    moving_average_5,
    moving_average_8,
    mean_of_all_history,
    mirror_around_50_last,
    mirror_around_60_last,
    mirror_around_capacity_last,
    linear_trend_2,
    linear_trend_3,
    linear_trend_5,
    repeat_last_change,
    median_of_last_3,
    median_of_last_5,
    min_of_last_3,
    max_of_last_3,
    weighted_average_recent,
    weighted_average_recent_5,
    alternating_cycle_2,
    alternating_cycle_3,
    bounded_trend_last_8,
    revert_to_mean_all_history,
    revert_to_capacity,
    average_of_last_and_5_ago,
    average_of_last_2_and_last_5,
    contrarian_last,
    exponential_smoothing,
]

OUTPUT_PREFIX = "repeated_random_fixed_predictors"


# ============================================================
# HELPERS
# ============================================================
def compute_payoffs(attendance: int, decisions: np.ndarray, capacity: int) -> np.ndarray:
    """
    Goers get +1 if attendance <= capacity, else -1.
    Stayers get 0.
    """
    payoffs = np.zeros_like(decisions, dtype=int)
    if attendance <= capacity:
        payoffs[decisions == 1] = 1
    else:
        payoffs[decisions == 1] = -1
    return payoffs


def assign_predictors(predictor_pool, n_players: int, rng: np.random.Generator):
    """Assign one random fixed predictor to each player."""
    idx = rng.integers(0, len(predictor_pool), size=n_players)
    return [predictor_pool[i] for i in idx]


def predictor_name_counts(player_predictors):
    """Count how many players were assigned each predictor."""
    counts = {}
    for func in player_predictors:
        name = func.__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def run_repeated_game(
    player_predictors,
    n_players: int,
    bar_capacity: int,
    n_rounds: int,
    initial_history,
    fallback_go_probability: float,
):
    """
    Repeated but non-inductive game:
    - each player is assigned one predictor at the beginning
    - each player uses that same predictor in every round
    - no scoring, switching, or learning across predictors
    """
    attendance_history = list(initial_history)

    round_attendance = np.zeros(n_rounds, dtype=int)
    round_avg_payoff = np.zeros(n_rounds, dtype=float)
    round_go_rate = np.zeros(n_rounds, dtype=float)
    round_mean_predicted_others = np.full(n_rounds, np.nan, dtype=float)
    player_cum_payoff = np.zeros(n_players, dtype=float)

    predicted_matrix = np.full((n_rounds, n_players), np.nan, dtype=float)
    decision_matrix = np.zeros((n_rounds, n_players), dtype=int)
    payoff_matrix = np.zeros((n_rounds, n_players), dtype=int)

    for t in range(n_rounds):
        decisions = np.zeros(n_players, dtype=int)
        predictions = np.full(n_players, np.nan, dtype=float)

        for i in range(n_players):
            predictor_func = player_predictors[i]
            pred_others, go = predictor_func(
                history=attendance_history,
                fallback_go_probability=fallback_go_probability,
                capacity=bar_capacity - 1,
                min_attendance=0,
                max_attendance=n_players - 1,
            )
            predictions[i] = pred_others
            decisions[i] = go

        attendance = int(decisions.sum())
        payoffs = compute_payoffs(attendance, decisions, bar_capacity)

        round_attendance[t] = attendance
        round_avg_payoff[t] = payoffs.mean()
        round_go_rate[t] = decisions.mean()
        round_mean_predicted_others[t] = np.nanmean(predictions) if np.any(~np.isnan(predictions)) else np.nan

        predicted_matrix[t, :] = predictions
        decision_matrix[t, :] = decisions
        payoff_matrix[t, :] = payoffs

        player_cum_payoff += payoffs
        attendance_history.append(attendance)

    return {
        "attendance_history_full": np.array(attendance_history, dtype=float),
        "round_attendance": round_attendance,
        "round_avg_payoff": round_avg_payoff,
        "round_go_rate": round_go_rate,
        "round_mean_predicted_others": round_mean_predicted_others,
        "player_cum_payoff": player_cum_payoff,
        "predicted_matrix": predicted_matrix,
        "decision_matrix": decision_matrix,
        "payoff_matrix": payoff_matrix,
    }


def save_plots(results, bar_capacity: int, output_prefix: str):
    rounds = np.arange(1, len(results["round_attendance"]) + 1)

    # 1. Attendance by round
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, results["round_attendance"], linewidth=1.5, label="Actual attendance")
    plt.axhline(bar_capacity, linestyle="--", alpha=0.7, label=f"Capacity = {bar_capacity}")
    plt.xlabel("Round")
    plt.ylabel("Attendance")
    plt.title("Attendance by Round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_attendance_by_round.png", dpi=150)
    plt.close()

    # 2. Average payoff by round
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, results["round_avg_payoff"], linewidth=1.5)
    plt.axhline(0, linestyle="--", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Average payoff per player")
    plt.title("Average Payoff per Player by Round")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_avg_payoff_by_round.png", dpi=150)
    plt.close()

    # 3. Go rate by round
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, results["round_go_rate"], linewidth=1.5)
    plt.xlabel("Round")
    plt.ylabel("Fraction going")
    plt.title("Go Rate by Round")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_go_rate_by_round.png", dpi=150)
    plt.close()

    # 4. Mean predicted others vs actual others
    actual_others = np.maximum(results["round_attendance"] - 1, 0)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, results["round_mean_predicted_others"], linewidth=1.5, label="Mean predicted others")
    plt.plot(rounds, actual_others, linewidth=1.2, alpha=0.8, label="Actual others (attendance - 1)")
    plt.axhline(bar_capacity - 1, linestyle="--", alpha=0.7, label=f"Others threshold = {bar_capacity - 1}")
    plt.xlabel("Round")
    plt.ylabel("Attendance of others")
    plt.title("Mean Predicted Others vs Actual Others")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_predicted_vs_actual_others.png", dpi=150)
    plt.close()

    # 5. Distribution of cumulative player payoffs
    plt.figure(figsize=(10, 5))
    plt.hist(results["player_cum_payoff"], bins=20)
    plt.xlabel("Cumulative payoff")
    plt.ylabel("Number of players")
    plt.title("Distribution of Cumulative Player Payoffs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cum_payoff_distribution.png", dpi=150)
    plt.close()

def save_predictor_summary_bar_chart(player_predictors, player_cum_payoff, output_prefix: str):
    """
    Save a large bar chart with one bar per predictor.
    Left axis: number of users assigned to that predictor
    Right axis: mean cumulative payoff per user for that predictor
    """
    grouped_payoffs = defaultdict(list)

    for func, payoff in zip(player_predictors, player_cum_payoff):
        grouped_payoffs[func.__name__].append(payoff)

    predictor_names = sorted(grouped_payoffs.keys())
    n_users = np.array([len(grouped_payoffs[name]) for name in predictor_names], dtype=float)
    mean_payoff = np.array([np.mean(grouped_payoffs[name]) for name in predictor_names], dtype=float)

    x = np.arange(len(predictor_names))
    width = 0.42

    fig, ax1 = plt.subplots(figsize=(22, 10))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, n_users, width=width, label="Number of users")
    bars2 = ax2.bar(x + width / 2, mean_payoff, width=width, alpha=0.75, label="Mean cumulative payoff per user")

    ax1.set_xlabel("Predictor")
    ax1.set_ylabel("Number of users")
    ax2.set_ylabel("Mean cumulative payoff per user")
    ax1.set_title("Predictor Assignment Counts and Mean Payoff per User")

    ax1.set_xticks(x)
    ax1.set_xticklabels(predictor_names, rotation=75, ha="right")

    ax1.grid(True, axis="y", alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_predictor_summary_bar_chart.png", dpi=180)
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    player_predictors = assign_predictors(PREDICTOR_POOL, N_PLAYERS, rng)
    assignment_counts = predictor_name_counts(player_predictors)

    results = run_repeated_game(
        player_predictors=player_predictors,
        n_players=N_PLAYERS,
        bar_capacity=BAR_CAPACITY,
        n_rounds=N_ROUNDS,
        initial_history=INITIAL_HISTORY,
        fallback_go_probability=FALLBACK_GO_PROBABILITY,
    )

    save_plots(results, BAR_CAPACITY, OUTPUT_PREFIX)
    save_predictor_summary_bar_chart(
    player_predictors=player_predictors,
    player_cum_payoff=results["player_cum_payoff"],
    output_prefix=OUTPUT_PREFIX,
)

    print("Fixed predictor assignment counts:")
    for name, count in assignment_counts.items():
        print(f"  {name}: {count}")

    print()
    print(f"Rounds: {N_ROUNDS}")
    print(f"Mean attendance: {results['round_attendance'].mean():.3f}")
    print(f"Std attendance: {results['round_attendance'].std():.3f}")
    print(f"Mean average payoff/player: {results['round_avg_payoff'].mean():.3f}")
    print(f"Mean go rate: {results['round_go_rate'].mean():.3f}")

    print()
    print("Saved plots:")
    print(f"  {OUTPUT_PREFIX}_attendance_by_round.png")
    print(f"  {OUTPUT_PREFIX}_avg_payoff_by_round.png")
    print(f"  {OUTPUT_PREFIX}_go_rate_by_round.png")
    print(f"  {OUTPUT_PREFIX}_predicted_vs_actual_others.png")
    print(f"  {OUTPUT_PREFIX}_cum_payoff_distribution.png")
    print(f"  {OUTPUT_PREFIX}_predictor_summary_bar_chart.png")