import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_neighborhood_wise_payoffs as compute_pgg
from update_strategies import soft_noisy_update_according_to_best_neighbor
from plot_utils import LinkedPlotter, avgPlotter
from statistics import stdev
from read_file_net import read_file_net


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)
else:
    seed = None

# Hyperparameters for the simulation
n_players = 30
starting_money = 100
mult_factor = 3
n_rounds = 10
connectivity = 4
prob_new_edge = 0.3
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False

# Initializations
# graph, n_players = read_file_net('facebook_net.txt')
# graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
graph = nx.barabasi_albert_graph(n_players, m = 3, seed=seed)
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
    payoffs = compute_pgg(graph, players_money, player_strategies, mult_factor)

    # Update the players strategies
    for i_player in range(len(player_strategies)):
        neighbor_idxs = list(graph.adj[i_player])
        neighbor_strats = [player_strategies[i] for i in neighbor_idxs]
        neighbor_payoffs = [payoffs[i] for i in neighbor_idxs]
        player_strategies[i_player] = update_strategy(players_money[i_player],
                                                      player_strategies[i_player],
                                                      payoffs[i_player],
                                                      neighbor_strats,
                                                      neighbor_payoffs,
                                                      alpha,
                                                      noise_intensity)

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

ax[0].set_title('P2: Graph (hover a node to outline its contribution)')
ax[1].set_title('P2: Contributions over time, n='+str(n_players)+', stoch.='+str(noise_intensity))
ax[1].set_xlabel('Round number')
ax[1].set_ylabel('Contributions')

# Plot graph and curves
linked_plotter = LinkedPlotter(graph, contribution_curves, ax[0], ax[1], fig, circle=False)
if save_plots:
    fig.savefig('fig/P2_individuals_graph-'+str(n_players)+'_'+str(noise_intensity)+'.png')

# Plot scatter of contributions and avg. in a different figure
fig2, ax2 = plt.subplots(ncols=2, figsize=(15, 6))
ax2[0].set_title('P2: Contribution vs connectivity')
ax2[0].set_xlabel('Degree')
ax2[0].set_ylabel('Average contribution')
ax2[1].set_title('P2: Median contribution over time (quart. percentiles), n='+str(n_players)+', stoch.='+str(noise_intensity))
ax2[1].set_xlabel('Round number')

# Plot average contribution vs degree and average contribution level
avgPlotter(graph, contribution_curves, mean_contribs, ax2[0], ax2[1])
if save_plots:
    fig2.savefig('fig/P2_median-'+str(n_players)+'_'+str(noise_intensity)+'.png')

plt.show()
