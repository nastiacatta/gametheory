import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N_PLAYERS = 101
CAPACITY = 60
N_ROUNDS = 10_000
PROBABILITIES = np.linspace(0, 1, 501)

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

    return avg_payoff_per_person, avg_attendance

# --- Main ---
if __name__ == "__main__":
    avg_payoffs = np.empty(len(PROBABILITIES))
    avg_attendances = np.empty(len(PROBABILITIES))

    for i, p in enumerate(PROBABILITIES):
        avg_payoffs[i], avg_attendances[i] = simulate(p, N_PLAYERS, CAPACITY, N_ROUNDS)

    # Find optimal probability
    best_idx = np.argmax(avg_payoffs)
    best_p = PROBABILITIES[best_idx]
    print(f"Best p = {best_p:.4f}  |  Avg payoff/person = {avg_payoffs[best_idx]:.4f}  |  Avg attendance = {avg_attendances[best_idx]:.2f}")

    # Plot 1: average payoff per person vs p
    plt.figure()
    plt.plot(PROBABILITIES, avg_payoffs, linewidth=1.5)
    plt.axvline(best_p, color="red", linestyle="--", alpha=0.6, label=f"best p = {best_p:.3f}")
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

    print("Saved el_farol_payoff.png and el_farol_attendance.png")
