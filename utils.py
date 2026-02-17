import numpy as np
import matplotlib.pyplot as plt


def plot_learning(
    scores,
    epsilons,
    filename,
    window=1,
    xlabel="Episode",
    ylabel="Score",
    title=None,
    dpi=200,
    show=False,
):
    """
    Plot learning curves for multiple epsilon values.

    Parameters
    ----------
    scores : array-like, shape (n_eps, n_steps)
        2D array containing score history for each epsilon value.
        Each row corresponds to one epsilon.

    epsilons : list or array-like
        List of epsilon values corresponding to each row in `scores`.

    filename : str
        Path where the plot will be saved (e.g., 'learning.png').

    window : int, optional
        Running average window size for smoothing (default=1 means no smoothing).

    xlabel, ylabel : str
        Axis labels.

    title : str or None
        Optional plot title.

    dpi : int
        Resolution of saved figure.

    show : bool
        If True, display the figure.
    """

    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array (n_eps, n_steps).")

    n_eps, n_steps = scores.shape

    if len(epsilons) != n_eps:
        raise ValueError("Length of epsilons must match number of rows in scores.")

    if window < 1:
        raise ValueError("window must be >= 1.")

    x = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots()

    for i, eps in enumerate(epsilons):
        curve = scores[i]

        # Apply running average smoothing if window > 1
        if window > 1:
            cumsum = np.cumsum(np.insert(curve, 0, 0))
            smoothed = (cumsum[window:] - cumsum[:-window]) / window
            x_plot = x[window - 1 :]
        else:
            smoothed = curve
            x_plot = x

        ax.plot(x_plot, smoothed, label=fr"$\epsilon$ = {eps}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)
