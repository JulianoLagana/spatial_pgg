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
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False
plot_graph = False
circle = True
log_scale = True # For the scatter plot
size_marker = 0.5
network = 'BA' # 'FB', 'BA' or 'WS'
n_points = 20
n_inits = 10
final_eta = 2
n_rounds_max = 2*10**3
n_rounds_check = 25
slope = 0.1


# Initializations for the different networks
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')

elif network == 'BA':
    n_players = 4039
    m = 4
    graph = nx.barabasi_albert_graph(n_players, m=m, seed=seed)

elif network == 'WS':
    n_players = 4039
    connectivity = 44
    prob_new_edge = 0.025
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

else:
    raise Exception('You have chosen a network type which is not defined.')

mean_degree = sum([graph.degree(i) for i in range(graph.order())])/n_players
print('Mean degree = {:d}'.format(int(mean_degree)))

mult_factors = np.arange(1, final_eta*(mean_degree + 1) + 0.01, (final_eta*(mean_degree+1)-1) / n_points)

players_money = np.array([starting_money]*n_players)
avg_median_contribs = np.zeros((len(mult_factors)))

for i_init in range(n_inits):

    initial_player_strategies = np.random.random(size=n_players) * starting_money
    index_mult_factor = 0
    print('Init. = {:d}'.format(i_init))

    for mult_factor in list(mult_factors):
        print('\t Mult. factor = {:.2f}'.format(mult_factor))
        player_strategies = np.copy(initial_player_strategies)

        medians = []

        for i_round in range(n_rounds_max):
            # Play one round
            payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)

            # Update the players strategies
            new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
            player_strategies = np.array(new_player_strategies)
            median_now = np.median(player_strategies)
            medians.append(median_now)

            if i_round % n_rounds_check == 0 and i_round > 0:
                fit = np.polyfit(np.arange(n_rounds_check), medians[-n_rounds_check:], deg=1)
                if np.abs(fit[0]) < slope:
                    avg_median_contribs[index_mult_factor] += np.mean(medians[-n_rounds_check:]) / n_rounds_check
                    print('\t Break after: {:d}'.format(int(i_round / n_rounds_check)))
                    break
                medians = []

        if i_round == (n_rounds_max - 1):
            print('\t Maximum number of rounds reached')
        index_mult_factor += 1

avg_median_contribs /= n_inits


plt.figure(figsize=(7, 6))
plt.ylabel('Average contribution')
plt.xlabel('r / (<k> + 1)')
x = mult_factors/(mean_degree + 1)
plt.plot(x, avg_median_contribs)

with open('fig/FinalCurves/x-' + network + '.npy', 'wb') as f:
    np.save(f, x)
with open('fig/FinalCurves/y-' + network + '.npy', 'wb') as f:
    np.save(f, avg_median_contribs)

plt.show()