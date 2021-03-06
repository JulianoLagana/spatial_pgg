import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_layered_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from plot_utils import LinkedPlotter


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
mult_factor = 3
n_rounds = 100
connectivity = 4
prob_new_edge = 0.3
alpha = 0.5
noise_intensity = 1
gamma = 1
update_strategy = soft_noisy_update_according_to_best_neighbor

# Initializations
players_money = np.array([starting_money]*n_players)
player_strategies = np.random.uniform(0.4, 1, size=n_players)*starting_money
contribs = np.zeros((n_rounds+1, n_players))
contribs[0, :] = player_strategies.copy()
graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

players = np.array(list(graph.nodes))
# Change number of countries
countries = np.array_split(players, 1)


for i_round in range(n_rounds):
    # Play one round
    payoffs = compute_pgg_layered_payoffs(graph, players_money, player_strategies, mult_factor, countries)
    # Change utility of players to incorporate average friends pay-off
    payoffs = [payoffs[i] + (gamma * np.min(payoffs[list(graph.adj[i])])) for i in range(len(player_strategies))]

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

    # Save contributions made this round
    contribs[i_round+1, :] = player_strategies.copy()

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
linked_plotter = LinkedPlotter(graph, contribution_curves, ax[0], ax[1], fig, circle=False, country=countries)
plt.show()
