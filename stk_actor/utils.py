import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, cur_epoch):
    # average rewards for each track
    avg_rewards = np.mean(rewards, axis=1)
    plt.plot(avg_rewards[:cur_epoch+1])
    plt.grid()
    plt.savefig('plots/avg_rewards')