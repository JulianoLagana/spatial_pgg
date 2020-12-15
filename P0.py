import numpy as np
import matplotlib.pyplot as plt

from pgg import compute_pgg_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)

# Hyperparameters for the simulation
n_players = 30
starting_money = 100
mult_factor = 1.5
n_rounds = 100
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False

# Initializations
players_money = np.array([starting_money]*n_players)
player_strategies = np.random.random(size=n_players)*starting_money
contribs = np.zeros((n_rounds+1, n_players))
contribs[0, :] = player_strategies.copy()
mean_contribs = np.zeros((3, n_rounds+1)) # data structure for the mean plot
mean_contribs[:, 0] = [np.median(player_strategies),
                       np.percentile(player_strategies, 25),
                       np.percentile(player_strategies, 75)]

for i_round in range(n_rounds):
    # Play one round
    payoffs = compute_pgg_payoffs(players_money, player_strategies, mult_factor)

    # Update the players strategies
    new_player_strategies = np.zeros(n_players)
    for i_player in range(len(player_strategies)):
        new_player_strategies[i_player] = update_strategy(players_money[i_player],
                                                          player_strategies[i_player],
                                                          payoffs[i_player],
                                                          player_strategies,
                                                          payoffs,
                                                          alpha,
                                                          noise_intensity)
    player_strategies = np.copy(new_player_strategies)

    # Save contributions made this round
    mean_contribs[:, i_round+1] = [np.median(player_strategies),
                                 np.percentile(player_strategies, 25),
                                 np.percentile(player_strategies, 75)] # for mean plot
    contribs[i_round+1, :] = player_strategies.copy() # Save contributions made this round

# --- Mean plot ---
plot_mean_contribs = plt.figure(0)
mean_color = (np.random.rand(), np.random.rand(), np.random.rand(), 0.3)
x = list(range(len(mean_contribs[0, :])))
plt.plot(mean_contribs[0, :], color=mean_color)
plt.fill_between(x, (mean_contribs[1, :]), (mean_contribs[2, :]), color=mean_color, edgecolor=None)
plt.title('P0: Median contribution over time (quart. percentiles), n='+str(n_players))
plt.xlabel('Round number')
plt.ylabel('Average Contribution')
plot_mean_contribs.show()
if save_plots:
    plot_mean_contribs.savefig('fig/P0_median-'+str(n_players)+'.png')

# --- Individuals plot ---
plot_contribs = plt.figure(1)
plt.plot(contribs, '.-')
plt.title('P0: Individual contribution over time, n='+str(n_players))
plt.xlabel('Round number')
plt.ylabel('Contributions')
plt.grid()
plot_contribs.show()
if save_plots:
    plot_contribs.savefig('fig/P0_individual-'+str(n_players)+'.png')

input()

