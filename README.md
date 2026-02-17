
# Multi-Armed Bandits & Continuous REINFORCE (TensorFlow)

This repository contains clean and educational implementations of fundamental reinforcement learning algorithms:

* 🎯 **k-Armed Bandit (Epsilon-Greedy)**
* 🎲 **REINFORCE (Policy Gradient) for Continuous Actions**

The code is designed to be:

* ✔ Clear and readable
* ✔ Minimal and modular
* ✔ Suitable for teaching and experimentation
* ✔ Easy to extend to more complex environments

---

# 📂 Repository Structure

```
.
├── multi_armed_bandit.py     # Epsilon-greedy bandit implementation
├── main.py                  # Policy gradient training script
├── utils.py                  # Plotting utilities
└── README.md
```

---

# 1️⃣ Multi-Armed Bandit

## Problem

A k-armed bandit with Gaussian rewards:

$$
R_t \sim \mathcal{N}(q^*(a), \sigma^2)
$$

where:

* $q^*(a)$ is the true mean of arm (a)
* $\sigma$ is fixed reward noise

---

## Algorithm: Epsilon-Greedy

At each step:

$$a_t =
\begin{cases}
\text{random arm}, & \text{with probability } \epsilon \\
\arg\max_a Q_t(a), & \text{otherwise}
\end{cases}$$

Action-value update:

* Sample average:
 $$Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)}(R_t - Q_t(a))$$

* Or constant step size:
$$Q_{t+1}(a) = Q_t(a) + \alpha(R_t - Q_t(a))$$

---

## Running the Bandit Experiment

```bash
python main.py
```

Outputs:

* `average_rewards.png`
* `optimal_actions.png`

Plots show:

* Average reward over time
* Probability of selecting the optimal arm

---

# 2️⃣ Continuous REINFORCE (Policy Gradient)

## Objective

Monte Carlo policy gradient for continuous action spaces:

$$ \mathcal{L}(\theta) =\mathbb{E}\left[
G_t \log \pi_\theta(a_t \mid s_t)
\right]$$

where:

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$$

---

## Policy Network

Implemented in `network_con.py`.

* Two hidden layers (ReLU)
* Output mean ( \mu(s) )
* Optional learned variance

Action distribution:

$$a \sim \mathcal{N}(\mu, \sigma^2)$$

Uses **TensorFlow Probability** for stable log-prob computation.

---

## Running REINFORCE

```bash
python main.py
```

Outputs:

* `score.png`
* `mu.png`
* `sigma.png`

These visualize:

* Episode return
* Policy mean evolution
* Policy variance (if learned)

---

# 🔧 Installation

## Requirements

* Python 3.9+
* NumPy
* Matplotlib
* TensorFlow 2.x
* TensorFlow Probability

Install:

```bash
pip install numpy matplotlib tensorflow tensorflow-probability
```

---

# 🎯 Design Philosophy

This repository focuses on:

* Simplicity over abstraction
* Clear mathematical correspondence
* Explicit algorithm implementation
* Minimal external dependencies

It is ideal for:

* Reinforcement learning coursework
* Understanding policy gradient fundamentals
* Experimenting with bandit strategies
* Rapid prototyping

---

# 📊 Example Experiments

You can easily modify:

* `epsilons` list for bandit comparison
* `alpha` (learning rate)
* `gamma` (discount factor)
* Hidden layer sizes
* Exploration parameters

---

# 🚀 Future Extensions

Possible improvements:

* UCB and Thompson Sampling bandits
* Actor-Critic methods
* Baseline subtraction for variance reduction
* Gym integration
* Non-stationary bandits
* Batch policy updates

---

# 📜 License

MIT License.


