import numpy as np
import matplotlib.pyplot as plt

from pgg import compute_pgg_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from statistics import stdev


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)

# Hyperparameters for the simulation
n_players = 10
starting_money = 100
mult_factor = 1.5
n_rounds = 30
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor

# Initializations
players_money = np.array([starting_money]*n_players)
player_strategies = np.random.random(size=n_players)*starting_money
contribs = np.zeros((n_rounds+1, n_players))
contribs[0, :] = player_strategies.copy()
mean_contribs = np.zeros((2, n_rounds+1)) # data structure for the mean plot

for i_round in range(n_rounds):
    # Play one round
    payoffs = compute_pgg_payoffs(players_money, player_strategies, mult_factor)

    # Update the players strategies
    for i_player in range(len(player_strategies)):
        player_strategies[i_player] = update_strategy(players_money[i_player],
                                                      player_strategies[i_player],
                                                      payoffs[i_player],
                                                      player_strategies,
                                                      payoffs,
                                                      alpha,
                                                      noise_intensity)

    # Save contributions made this round
    mean_contribs[:, i_round+1] = [sum(player_strategies) / n_players, stdev(player_strategies)] # for mean plot
    contribs[i_round+1, :] = player_strategies.copy() # Save contributions made this round

# --- Mean plot ---
plot_mean_contribs = plt.figure(0)
mean_color = (np.random.rand(), np.random.rand(), np.random.rand(), 0.5)
x = list(range(0, n_rounds+1))
plt.plot(mean_contribs[0, :], color=mean_color)
plt.fill_between(x,
                 (mean_contribs[0, :]+2*mean_contribs[1, :]),
                 (mean_contribs[0, :]-2*mean_contribs[1, :]),
                 color=mean_color)
plt.title('Mean contribution over time (+/- 2SD)')
plt.xlabel('Round number')
plt.ylabel('Average Contribution')
plt.grid()
plot_mean_contribs.show()

# --- Individuals plot ---
plot_contribs = plt.figure(1)
plt.plot(contribs, '.-')
plt.title('Individual contribution over time')
plt.xlabel('Round number')
plt.ylabel('Contributions')
plt.grid()
plot_contribs.show()

input()

