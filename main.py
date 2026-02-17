import numpy as np

from multi_armed_bandit import BanditProblem
from utils import plot_learning  # use the improved plotting function name


def run_experiment(
    epsilons=(0.0, 0.1),
    num_steps=1000,
    num_runs=2000,
    num_actions=10,
    alpha=0.1,
    seed=0,
):
    """
    Run a k-armed bandit experiment for multiple epsilon values.

    For each epsilon:
      - Perform `num_runs` independent bandit runs of length `num_steps`
      - Aggregate:
          * average reward per step
          * probability of choosing the optimal action per step

    Parameters
    ----------
    epsilons : iterable of float
        Epsilon values for epsilon-greedy.
    num_steps : int
        Steps per run.
    num_runs : int
        Number of independent runs per epsilon.
    num_actions : int
        Number of bandit arms.
    alpha : float
        Constant step size (if you use constant-alpha updates).
    seed : int
        Seed for reproducibility.

    Returns
    -------
    avg_rewards : np.ndarray, shape (len(epsilons), num_steps)
        Mean reward at each step, averaged over runs.
    opt_action_rate : np.ndarray, shape (len(epsilons), num_steps)
        Fraction of runs that selected the optimal action at each step.
    """
    rng = np.random.default_rng(seed)

    epsilons = list(epsilons)
    avg_rewards = np.zeros((len(epsilons), num_steps), dtype=float)
    opt_action_rate = np.zeros((len(epsilons), num_steps), dtype=float)

    for i, eps in enumerate(epsilons):
        for run in range(num_runs):
            # -----------------------------
            # Sample true action values
            # -----------------------------
            values = rng.normal(loc=0.0, scale=1.0, size=num_actions)

            # Optional: force a "best" arm for eps=0 experiments (as in your original code)
            if eps == 0.0:
                values[0] = 5.0

            # -----------------------------
            # Run bandit algorithm
            # -----------------------------
            bandit = BanditProblem(
                values=values,
                num_arms=num_actions,
                eps=eps,
                num_steps=num_steps,
                alpha=alpha,
                seed=rng.integers(0, 2**32 - 1),
            )

            rewards, optimal_flags = bandit.run()

            avg_rewards[i] += rewards
            opt_action_rate[i] += optimal_flags

        # average over runs for this epsilon
        avg_rewards[i] /= num_runs
        opt_action_rate[i] /= num_runs

    return avg_rewards, opt_action_rate, epsilons


def main():
    # -----------------------------
    # Experiment config
    # -----------------------------
    num_steps = 1000
    num_runs = 2000
    num_actions = 10
    epsilons = [0.0, 0.1]
    alpha = 0.1

    avg_rewards, opt_action_rate, epsilons = run_experiment(
        epsilons=epsilons,
        num_steps=num_steps,
        num_runs=num_runs,
        num_actions=num_actions,
        alpha=alpha,
        seed=0,
    )

    # -----------------------------
    # Plot results
    # -----------------------------
    plot_learning(
        scores=avg_rewards,
        epsilons=epsilons,
        filename="average_rewards.png",
        xlabel="Steps",
        ylabel="Average reward",
        window=1,
        title="Average Reward vs Steps",
    )

    plot_learning(
        scores=opt_action_rate,
        epsilons=epsilons,
        filename="optimal_actions.png",
        xlabel="Steps",
        ylabel="P(optimal action)",
        window=1,
        title="Optimal Action Rate vs Steps",
    )


if __name__ == "__main__":
    main()
