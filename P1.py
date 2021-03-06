import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from plot_utils import LinkedPlotter, avgPlotter


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)
else:
    seed = None

# Hyperparameters for the simulation
n_players = 10
starting_money = 100
mult_factor = 1.5
n_rounds = 30
connectivity = 6
prob_new_edge = 0.3
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = True
circle = True

network = 'BA'

# Initializations
players_money = np.array([starting_money]*n_players)
player_strategies = np.random.random(size=n_players)*starting_money
contribs = np.zeros((n_rounds+1, n_players))
contribs[0, :] = player_strategies.copy()
mean_contribs = np.zeros((3, n_rounds+1)) # data structure for the mean plot
mean_contribs[:, 0] = [np.median(player_strategies),
                       np.percentile(player_strategies, 25),
                       np.percentile(player_strategies, 75)]
graph = nx.barabasi_albert_graph(n_players, connectivity, seed=seed)
# graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)


for i_round in range(n_rounds):
    # Play one round
    payoffs = compute_pgg_payoffs(players_money, player_strategies, mult_factor)

    # Update the players strategies
    new_player_strategies = np.zeros(shape=n_players)
    for i_player in range(len(player_strategies)):
        neighbor_idxs = list(graph.adj[i_player])
        neighbor_strats = [player_strategies[i] for i in neighbor_idxs]
        neighbor_payoffs = [payoffs[i] for i in neighbor_idxs]
        new_player_strategies[i_player] = update_strategy(players_money[i_player],
                                                          player_strategies[i_player],
                                                          payoffs[i_player],
                                                          neighbor_strats,
                                                          neighbor_payoffs,
                                                          alpha,
                                                          noise_intensity)

    player_strategies = new_player_strategies.copy()
    mean_contribs[:, i_round+1] = [np.median(player_strategies),
                                   np.percentile(player_strategies, 25),
                                   np.percentile(player_strategies, 75)] # for mean plot
    contribs[i_round+1, :] = player_strategies.copy() # Save contributions made this round


# Change the format of the saved contributions for plotting
xs = [i for i in range(n_rounds+1)]
contribution_curves = []
for i_player in range(n_players):
    contribution_curves.append([xs, contribs[:, i_player]])

# Create plotting window
fig, ax = plt.subplots(ncols=2, figsize=(15, 6))
ax[0].set_title('P1: Graph (hover a node to outline its contribution)')
ax[1].set_title('P1: Contributions over time, n='+str(n_players))
ax[1].set_xlabel('Round number')
ax[1].set_ylabel('Contributions')
plt.grid()

# Plot graph and curves
linked_plotter = LinkedPlotter(graph, contribution_curves, ax[0], ax[1], fig, circle=circle)
if save_plots:
    fig.savefig('fig/P1_individuals_graph-'+str(n_players)+'.png')

# Plot scatter of contributions and avg. in a different figure
fig2, ax2 = plt.subplots(ncols=2, figsize=(15, 6))
ax2[0].set_title('P1: Contribution vs connectivity')
ax2[0].set_xlabel('Degree')
ax2[0].set_ylabel('Average contribution')
ax2[1].set_title('P1: Median contribution over time (quart. percentiles), r='+str(mult_factor)+', n='+str(n_players))
ax2[1].set_xlabel('Round number')

# Plot average contribution vs degree and average contribution level
avgPlotter(graph, contribution_curves, mean_contribs, ax2[0], ax2[1], network=network)
if save_plots:
    fig2.savefig('fig/P1_median-'+str(n_players)+'.png')

plt.show()