import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_neighborhood_wise_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from plot_utils import LinkedPlotter, avgPlotter
from read_file_net import read_file_net
import multiprocessing
from joblib import Parallel, delayed




def parallel_function(i_player, neighbor_idxs, player_strategies, payoffs, players_money, alpha, noise_intensity):
    neighbor_strats = [player_strategies[i] for i in neighbor_idxs]
    neighbor_payoffs = [payoffs[i] for i in neighbor_idxs]
    new_player_strategy = update_strategy(players_money[i_player],
                                          player_strategies[i_player],
                                          payoffs[i_player],
                                          neighbor_strats,
                                          neighbor_payoffs,
                                          alpha,
                                          noise_intensity)
    return new_player_strategy


# Configurations
reproducible = True

# Optionally set seed for reproducibility
if reproducible:
    seed = 0
    np.random.seed(seed)
else:
    seed = None

num_cores = multiprocessing.cpu_count()

# Hyperparameters for the simulation
n_players = 100
starting_money = 100
n_rounds_trans = 20
n_rounds_avg = 5
connectivity = 4
prob_new_edge = 0.3
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False
plot_graph = False
circle = True
log_scale = True # For the scatter plot
size_marker = 0.5
network = 'BA' # 'FB', 'BA' or 'WS'
n_inits = 2
mult_factor = 5


# Initializations
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')
elif network == 'BA':
    graph = nx.barabasi_albert_graph(n_players, m=3, seed=seed)
else:
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

mean_degree = sum([graph.degree(i) for i in range(graph.order())])/n_players
print('Mean degree = {:d}'.format(int(mean_degree)))


avg_median_contribs = np.zeros(n_inits)

mean_contribs = np.zeros((n_inits,3, n_rounds_trans + n_rounds_avg + 1)) # data structure for the mean plot

# Plot scatter of contributions and avg. in a different figure
plt.figure(figsize=(7, 6))
plt.ylabel('Average contribution')
plt.xlabel('Round number')

for index in range(n_inits):
    print(index)
    players_money = np.array([starting_money] * n_players)
    player_strategies = np.random.random(size=n_players) * starting_money


    mean_contribs[index, :, 0] = [np.median(player_strategies),
                           np.percentile(player_strategies, 25),
                           np.percentile(player_strategies, 75)]

    for i_round in range(n_rounds_trans):
        # Play one round
        payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)

        # Update the players strategies
        new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
        player_strategies = np.array(new_player_strategies)
        mean_contribs[index, :, i_round+1] = [np.median(player_strategies),
                                     np.percentile(player_strategies, 25),
                                     np.percentile(player_strategies, 75)] # for mean plot

    for i_round in range(n_rounds_avg):
        # Play one round
        payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)

        # Update the players strategies
        new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
        player_strategies = np.array(new_player_strategies)
        median_aux = np.median(player_strategies)
        mean_contribs[index, :, i_round+n_rounds_trans+1] = [median_aux,
                                     np.percentile(player_strategies, 25),
                                     np.percentile(player_strategies, 75)] # for mean plot
        avg_median_contribs[index] += median_aux
    avg_median_contribs[index] /= n_rounds_avg

    # Plot avg. contribution
    mean_color = (np.random.rand(), np.random.rand(), np.random.rand(), 1)
    x = list(range(len(mean_contribs[0, :])))
    # plt.plot(mean_contribs[0, :], color=mean_color, label='r = {:.2f}'.format(mult_factor))
    plt.plot(mean_contribs[index, 0, :], label='init = {:d}'.format(int(index)))
    # plt.fill_between(x, (mean_contribs[1, :]), (mean_contribs[2, :]), color=mean_color, edgecolor=None)
    plt.ylim(0, 100)


plt.legend()

# Plot scatter avg contrib vs egree
plt.figure(figsize=(6, 6))
degree = [graph.degree(i) for i in range(graph.order())]
existing_degrees = [d for d in sorted(set(degree))]

mean_contribs_player = np.median(mean_contribs[:, 0, :], axis=0) ####
print(np.size(mean_contribs))
ordered_contribs = [[] for i in range(len(existing_degrees))]
for idx in range(len(degree)):
    ordered_contribs[existing_degrees.index(degree[idx])].append(mean_contribs_player[idx])

median_contribs_degree = [np.median(ordered_contribs[i]) for i in range(len(existing_degrees))]
error_bars = np.zeros((2, len(existing_degrees)))
error_bars[0, :] = [median_contribs_degree[i] - np.percentile(ordered_contribs[i], 25) for i in range(len(existing_degrees))]
error_bars[1, :] = [np.percentile(ordered_contribs[i], 75) - median_contribs_degree[i] for i in range(len(existing_degrees))]

size_marker = [len(ordered_contribs[i]) * size_marker for i in range(len(existing_degrees))]
plt.scatter(existing_degrees, median_contribs_degree, s=size_marker)
plt.errorbar(existing_degrees, median_contribs_degree, error_bars,
                   alpha=0.5, linestyle='--')

plt.show()