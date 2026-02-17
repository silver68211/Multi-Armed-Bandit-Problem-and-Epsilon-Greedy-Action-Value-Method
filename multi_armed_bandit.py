import numpy as np


class BanditProblem:
    """
    Epsilon-greedy k-armed bandit.

    This class simulates a stationary k-armed bandit environment with
    Gaussian rewards and an epsilon-greedy agent that learns action-value
    estimates using either:
      - sample-average updates (alpha = 1 / N_a), or
      - constant step-size updates (alpha fixed)

    Parameters
    ----------
    values : array-like, shape (num_arms,)
        True mean reward for each arm (ground-truth).
    num_arms : int, default=10
        Number of arms.
    var_val : float, default=1.0
        Standard deviation of the reward noise for each arm.
    eps : float, default=0.1
        Exploration probability for epsilon-greedy.
    num_steps : int, default=1000
        Number of interaction steps.
    alpha : float or None, default=None
        If None, uses sample-average learning rate (1 / N_a).
        If float, uses constant step-size alpha.

    Attributes
    ----------
    est_values : np.ndarray
        Estimated action-values Q_t(a).
    avg_reward : np.ndarray
        Reward observed at each time step.
    optimal_action_count : np.ndarray
        Indicator array: 1 if optimal action chosen at step t, else 0.
    """

    def __init__(
        self,
        values,
        num_arms=10,
        var_val=1.0,
        eps=0.1,
        num_steps=int(1e3),
        alpha=None,
        seed=None,
    ):
        # -----------------------------
        # Validate and store config
        # -----------------------------
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError("values must be a 1D array of shape (num_arms,).")

        self.num_arms = int(num_arms)
        if self.num_arms != self.values.size:
            raise ValueError("num_arms must match len(values).")

        self.var_val = float(var_val)
        if self.var_val <= 0:
            raise ValueError("var_val must be > 0.")

        self.eps = float(eps)
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError("eps must be in [0, 1].")

        self.num_steps = int(num_steps)
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = float(self.alpha)
            if not (0.0 < self.alpha <= 1.0):
                raise ValueError("alpha must be in (0, 1] if provided.")

        self.rng = np.random.default_rng(seed)

        # -----------------------------
        # Ground-truth optimal action
        # -----------------------------
        self.opt_action = int(np.argmax(self.values))

        # -----------------------------
        # Learning state
        # -----------------------------
        self.est_values = np.zeros(self.num_arms, dtype=float)   # Q estimates
        self.action_counts = np.zeros(self.num_arms, dtype=int)  # N_a

        # Logs per step
        self.rewards = np.zeros(self.num_steps, dtype=float)
        self.optimal_action_count = np.zeros(self.num_steps, dtype=int)

    # -----------------------------
    # Policy + environment
    # -----------------------------
    def choose_action(self):
        """
        Epsilon-greedy action selection.

        Returns
        -------
        action : int
            Selected arm index in [0, num_arms-1].
        """
        explore = self.rng.random() < self.eps
        if explore:
            return int(self.rng.integers(self.num_arms))
        return int(np.argmax(self.est_values))

    def get_reward(self, action):
        """
        Sample reward from N(true_mean[action], var_val^2).

        Parameters
        ----------
        action : int
            Arm index.

        Returns
        -------
        reward : float
        """
        mean = self.values[action]
        return float(self.rng.normal(loc=mean, scale=self.var_val))

    # -----------------------------
    # Learning update
    # -----------------------------
    def update_action_values(self, action, reward):
        """
        Update estimated Q-value for the selected action.

        Uses:
          - alpha = 1/N_a if self.alpha is None (sample average),
          - otherwise constant alpha.

        Parameters
        ----------
        action : int
        reward : float
        """
        self.action_counts[action] += 1

        if self.alpha is None:
            step = 1.0 / self.action_counts[action]
        else:
            step = self.alpha

        self.est_values[action] += step * (reward - self.est_values[action])

    # -----------------------------
    # Main loop
    # -----------------------------
    def run(self):
        """
        Run the bandit interaction loop for num_steps steps.

        Returns
        -------
        rewards : np.ndarray
            Reward at each step (shape: num_steps,).
        optimal_action_rate : np.ndarray
            Indicator of selecting the optimal action at each step (0/1).
        """
        for t in range(self.num_steps):
            action = self.choose_action()
            reward = self.get_reward(action)
            self.update_action_values(action, reward)

            self.rewards[t] = reward
            self.optimal_action_count[t] = int(action == self.opt_action)

        return self.rewards, self.optimal_action_count
