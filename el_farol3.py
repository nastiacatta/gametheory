import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N_PLAYERS = 101
CAPACITY = 60
N_ROUNDS = 10_000
PROBABILITIES = np.linspace(0, 1, 201)

def simulate(p, n_players, capacity, n_rounds):
    """Simulate the El Farol game for a given attendance probability p."""
    # Each entry: 1 if player goes, 0 if stays (n_rounds x n_players)
    decisions = np.random.random((n_rounds, n_players)) < p
    attendance = decisions.sum(axis=1)  # total attendance per round

    # Payoffs: +1 if go and attendance <= capacity, -1 if go and attendance > capacity, 0 if stay
    overcrowded = attendance > capacity  # shape (n_rounds,)
    payoffs_if_go = np.where(overcrowded, -1, 1)  # per-round payoff for goers
    # Total payoff per round = (number who went) * payoff_if_go + (number who stayed) * 0
    total_payoff_per_round = decisions.sum(axis=1) * payoffs_if_go
    avg_payoff_per_person = total_payoff_per_round.sum() / (n_rounds * n_players)
    avg_attendance = attendance.mean()

    # Cumulative payoff per agent across all rounds
    per_agent_payoffs = (decisions * payoffs_if_go[:, np.newaxis]).sum(axis=0)

    # Average number of agents per round in each payoff bucket
    goers = decisions.sum(axis=1)
    avg_n_positive = goers[~overcrowded].sum() / n_rounds   # went, not overcrowded → +1
    avg_n_negative = goers[overcrowded].sum() / n_rounds    # went, overcrowded     → -1
    avg_n_zero = (n_players - goers).mean()                 # stayed                →  0

    return avg_payoff_per_person, avg_attendance, per_agent_payoffs, avg_n_positive, avg_n_negative, avg_n_zero

# --- Main ---
if __name__ == "__main__":
    avg_payoffs = np.empty(len(PROBABILITIES))
    avg_attendances = np.empty(len(PROBABILITIES))
    avg_n_pos = np.empty(len(PROBABILITIES))
    avg_n_neg = np.empty(len(PROBABILITIES))
    avg_n_zer = np.empty(len(PROBABILITIES))

    for i, p in enumerate(PROBABILITIES):
        avg_payoffs[i], avg_attendances[i], _, avg_n_pos[i], avg_n_neg[i], avg_n_zer[i] = simulate(p, N_PLAYERS, CAPACITY, N_ROUNDS)

    # Find optimal probability
    best_idx = np.argmax(avg_payoffs)
    best_p = PROBABILITIES[best_idx]
    print(f"Best p = {best_p:.4f}  |  Avg payoff/person = {avg_payoffs[best_idx]:.4f}  |  Avg attendance = {avg_attendances[best_idx]:.2f}")

    # Plot 1: average payoff per person vs p
    plt.figure()
    plt.plot(PROBABILITIES, avg_payoffs, linewidth=1.5)
    plt.axvline(best_p, color="red", linestyle="--", alpha=0.6, label=f"best p = {best_p:.3f}")
    plt.axvline(0.59463, color="gray", linestyle="--", alpha=0.6, label="MSNE")
    plt.xlabel("Probability p")
    plt.ylabel("Average payoff per person")
    plt.title("El Farol Bar Game — Payoff vs Attendance Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("el_farol_payoff.png", dpi=150)

    # Plot 2: average attendance vs p
    plt.figure()
    plt.plot(PROBABILITIES, avg_attendances, linewidth=1.5)
    plt.axhline(CAPACITY, color="red", linestyle="--", alpha=0.6, label=f"capacity = {CAPACITY}")
    plt.xlabel("Probability p")
    plt.ylabel("Average attendance")
    plt.title("El Farol Bar Game — Attendance vs Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("el_farol_attendance.png", dpi=150)

    # Plot 3: average number of agents per round in each payoff bucket vs p
    plt.figure()
    plt.plot(PROBABILITIES, avg_n_pos, color="green",  linewidth=1.5, label="+1 agents (went, not crowded)")
    plt.plot(PROBABILITIES, avg_n_neg, color="red",    linewidth=1.5, label="-1 agents (went, crowded)")
    plt.plot(PROBABILITIES, avg_n_zer, color="steelblue", linewidth=1.5, label=" 0 agents (stayed)")
    plt.axvline(best_p,  color="black",  linestyle="--", alpha=0.5, label=f"best p = {best_p:.3f}")
    plt.axvline(0.59463, color="gray",   linestyle="--", alpha=0.5, label="MSNE")
    plt.xlabel("Probability p")
    plt.ylabel("Avg number of agents per round")
    plt.title("El Farol Bar Game — Agent Payoff Counts vs Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("el_farol_counts.png", dpi=150)

    # Plot 4: cumulative payoff histogram of agents at best_p
    _, _, agent_payoffs, *_ = simulate(best_p, N_PLAYERS, CAPACITY, N_ROUNDS)
    plt.figure()
    plt.hist(agent_payoffs, bins=30, edgecolor="black", linewidth=0.5)
    plt.axvline(agent_payoffs.mean(), color="red", linestyle="--", alpha=0.8, label=f"mean = {agent_payoffs.mean():.1f}")
    plt.xlabel("Cumulative payoff")
    plt.ylabel("Number of agents")
    plt.title(f"El Farol Bar Game — Agent Cumulative Payoff Distribution (p = {best_p:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("el_farol_payoff_hist.png", dpi=150)

    print("Saved el_farol_payoff.png, el_farol_attendance.png, el_farol_counts.png, and el_farol_payoff_hist.png")
