import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rewards(rewards, cur_epoch):
    sns.set_theme(style="whitegrid")  # Clean background
    avg_rewards = np.mean(rewards, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        avg_rewards[: cur_epoch + 1],
        label="Average Reward",
        color="tab:blue",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="red",
        markeredgewidth=1.5,
    )

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.title("Training Progress", fontsize=16, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/avg_rewards.png", dpi=300, transparent=True)
    plt.show()

