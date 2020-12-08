import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from plot_utils import LinkedPlotter
from statistics import stdev


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)
else:
    seed = None

# Hyperparameters for the simulation
n_players = 100
starting_money = 100
mult_factor = 1.5
n_rounds = 30
connectivity = 4
prob_new_edge = 0.3
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor

# Initializations
players_money = np.array([starting_money]*n_players)
player_strategies = np.random.random(size=n_players)*starting_money
contribs = np.zeros((n_rounds+1, n_players))
contribs[0, :] = player_strategies.copy()
mean_contribs = np.zeros((2, n_rounds+1)) # data structure for the mean plot
graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

for i_round in range(n_rounds):
    # Play one round
    payoffs = compute_pgg_payoffs(players_money, player_strategies, mult_factor)

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

    mean_contribs[:, i_round + 1] = [sum(player_strategies) / n_players, stdev(player_strategies)]  # for mean plot
    contribs[i_round + 1, :] = player_strategies.copy()  # Save contributions made this round


# --- Mean plot ---
plot_mean_contribs = plt.figure(1)
mean_color = (np.random.rand(), np.random.rand(), np.random.rand(), 0.5)
x = list(range(0, n_rounds+1))
plt.plot(mean_contribs[0, :], color=mean_color)
plt.fill_between(x,
                 (mean_contribs[0, :]+2*mean_contribs[1, :]),
                 (mean_contribs[0, :]-2*mean_contribs[1, :]),
                 color=mean_color)
plt.title('Mean contribution over time')
plt.suptitle('(Mean +/- 2SD)')
plt.xlabel('Round number')
plt.ylabel('Average Contribution')
plt.grid()
plot_mean_contribs.show()

# Change the format of the saved contributions for plotting
xs = [i for i in range(n_rounds+1)]
contribution_curves = []
for i_player in range(n_players):
    contribution_curves.append([xs, contribs[:, i_player]])

# Create plotting window
fig, ax = plt.subplots(ncols=2, figsize=(15, 6))
ax[0].set_title('Graph (hover a node to outline its contribution)')
ax[1].set_title('Contributions over time')
ax[1].set_xlabel('Round number')
ax[1].set_ylabel('Contributions')
plt.grid()

# Plot
linked_plotter = LinkedPlotter(graph, contribution_curves, ax[0], ax[1], fig)


# Plot scatter of contributions and avg. in a different figure
fig2, ax2 = plt.subplots(ncols=2, figsize=(15, 6))
ax2[0].set_title('Contribution vs connectivity')
ax2[0].set_xlabel('Degree')
ax2[0].set_ylabel('Contributions')
# This is for you John
ax2[1].set_title('Avg. contribution over time')
ax2[1].set_xlabel('Round number')
plt.grid()

# Plot scatter
contributions = [y[len(y)-1] for _, y in contribution_curves]
degree = [graph.degree(i) for i in range(graph.order())]
min_degree = min(degree)
max_degree = max(degree)
ordered_contribs = [[] for i in range(min_degree, max_degree+1)]
for idx in range(len(degree)):
    ordered_contribs[degree[idx]-min_degree].append(contributions[idx])
ax2[0].boxplot(ordered_contribs, positions=range(min_degree, max_degree+1))


# Plot avg. contribution


plt.show()
