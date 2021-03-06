import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgg import compute_pgg_neighborhood_wise_payoffs, compute_pgg_neighborhood_wise_payoffs_old
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
starting_money = 100
eta = 0.55
n_rounds = 500
alpha = 0.5
noise_intensity = 1
update_strategy = soft_noisy_update_according_to_best_neighbor
save_plots = False
plot_graph = False
circle = True
log_scale = True # For the scatter plot
size_marker = 0.5

network = 'BA' # 'FB', 'BA' or 'WS'


# Initializations
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')
elif network == 'BA':
    n_players = 4039
    graph = nx.barabasi_albert_graph(n_players, m=4, seed=seed)
else:
    n_players = 4039
    connectivity = 44
    prob_new_edge = 0.025
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)

mean_degree = sum([graph.degree(i) for i in range(graph.order())])/n_players
mult_factor = eta*(mean_degree + 1)

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

    if i_round % 10 == 0:
        print('Round: {:d}'.format(i_round))
    # Play one round
    payoffs = compute_pgg_neighborhood_wise_payoffs(graph, players_money, player_strategies, mult_factor)


    # Update the players strategies
    new_player_strategies = Parallel(n_jobs=num_cores)(delayed(parallel_function)(i_player, list(graph.adj[i_player]), player_strategies, payoffs, players_money, alpha, noise_intensity) for i_player in range(len(player_strategies)))
    player_strategies = np.array(new_player_strategies)
    mean_contribs[:, i_round+1] = [np.median(player_strategies),
                                 np.percentile(player_strategies, 25),
                                 np.percentile(player_strategies, 75)] # for mean plot
    contribs[i_round+1, :] = player_strategies.copy() # Save contributions made this round



# Change the format of the saved contributions for plotting
xs = [i for i in range(n_rounds+1)]
contribution_curves = []
for i_player in range(n_players):
    contribution_curves.append([xs, contribs[:, i_player]])

if plot_graph:
    # Create plotting window
    fig, ax = plt.subplots(ncols=2, figsize=(15, 6))

    ax[0].set_title('P2: Graph (hover a node to outline its contribution)')
    ax[1].set_title('P2: Contributions over time, n='+str(n_players)+', stoch.='+str(noise_intensity))
    ax[1].set_xlabel('Round number')
    ax[1].set_ylabel('Contributions')

    # Plot graph and curves
    linked_plotter = LinkedPlotter(graph, contribution_curves, ax[0], ax[1], fig, circle=circle)
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
avgPlotter(graph, contribution_curves, mean_contribs, ax2[0], ax2[1], log_scale=log_scale, size_marker=size_marker, network=network)
if save_plots:
    fig2.savefig('fig/P2_median-'+str(n_players)+'_'+str(noise_intensity)+'.png')

plt.show()
