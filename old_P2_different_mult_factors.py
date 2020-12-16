import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_neighborhood_wise_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
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
starting_money = 100
n_rounds_trans = 100
n_rounds_avg = 10
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False
plot_graph = False
circle = True
log_scale = True # For the scatter plot
size_marker = 0.5
network = 'WS' # 'FB', 'BA' or 'WS'
n_points = 10


# Initializations
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')
elif network == 'BA':
    n_players = 4039
    m = 4
    graph = nx.barabasi_albert_graph(n_players, m=m, seed=seed)
else:
    n_players = 4039
    connectivity = 44
    prob_new_edge = 0.025
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

mean_degree = sum([graph.degree(i) for i in range(graph.order())])/n_players
print('Mean degree = {:d}'.format(int(mean_degree)))

mult_factors = np.arange(1, mean_degree + 1.01, (mean_degree+1) / n_points)

players_money = np.array([starting_money]*n_players)
initial_player_strategies = np.random.random(size=n_players)*starting_money
avg_median_contribs = np.zeros((len(mult_factors)))

# Plot scatter of contributions and avg. in a different figure
plt.figure(figsize=(7, 6))
plt.ylabel('Average contribution')
plt.xlabel('Round number')

index = 0
for mult_factor in list(mult_factors):
    print(mult_factor)
    player_strategies = np.copy(initial_player_strategies)

    mean_contribs = np.zeros((3, n_rounds_trans + n_rounds_avg + 1)) # data structure for the mean plot
    mean_contribs[:, 0] = [np.median(player_strategies),
                           np.percentile(player_strategies, 25),
                           np.percentile(player_strategies, 75)]

    for i_round in range(n_rounds_trans):
        # Play one round
        payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)

        # Update the players strategies
        new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
        player_strategies = np.array(new_player_strategies)
        mean_contribs[:, i_round+1] = [np.median(player_strategies),
                                       np.percentile(player_strategies, 25),
                                       np.percentile(player_strategies, 75)] # for mean plot


    for i_round in range(n_rounds_avg):
        # Play one round
        payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)

        # Update the players strategies
        new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
        player_strategies = np.array(new_player_strategies)
        median_aux = np.median(player_strategies)
        mean_contribs[:, i_round+n_rounds_trans+1] = [median_aux,
                                                      np.percentile(player_strategies, 25),
                                                      np.percentile(player_strategies, 75)] # for mean plot
        avg_median_contribs[index] += median_aux
    avg_median_contribs[index] /= n_rounds_avg

    # Plot avg. contribution
    mean_color = (np.random.rand(), np.random.rand(), np.random.rand(), 1)
    x = list(range(len(mean_contribs[0, :])))
    # plt.plot(mean_contribs[0, :], color=mean_color, label='r = {:.2f}'.format(mult_factor))
    plt.plot(mean_contribs[0, :], label='r = {:.2f}'.format(mult_factor))
    # plt.fill_between(x, (mean_contribs[1, :]), (mean_contribs[2, :]), color=mean_color, edgecolor=None)
    plt.ylim(0, 100)

    index += 1

plt.legend()

plt.figure(figsize=(7, 6))
plt.ylabel('Average contribution')
plt.xlabel('r / (<k> + 1)')
x = mult_factors/(mean_degree + 1)
plt.plot(x, avg_median_contribs)

with open('x-' + network + '.npy', 'wb') as f:
    np.save(f, x)
with open('y-' + network + '.npy', 'wb') as f:
    np.save(f, avg_median_contribs)

plt.show()