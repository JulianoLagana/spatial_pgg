import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor


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

    # Save contributions made this round
    contribs[i_round+1, :] = player_strategies.copy()

plt.subplot(121)
plt.title('Graph structure')
nx.draw(graph, with_labels=True)

plt.subplot(122)
plt.title('Contributions over time')
plt.plot(contribs, '.-')
plt.xlabel('Round number')
plt.ylabel('Contributions')
plt.grid()
plt.show()

