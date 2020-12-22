import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from read_file_net import read_file_net

seed = 1
np.random.seed(seed)

network = 'WS-2' # 'FB', 'BA' or 'WS'
marker_size = 0.8
plot_graph = False
max_index_fit = 22

# Initializations
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')
    plot_graph = False

elif network == 'BA':
    n_players = 4039
    m = 4
    graph = nx.barabasi_albert_graph(n_players, m=m, seed=seed)
    circle = True

elif network == 'WS':
    n_players = 4039
    connectivity = 18 # 44
    prob_new_edge = 0.1 # 0.025
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
    circle = True

elif network == 'WS-1': # For example for the clustering coeff simulations
    n_players = 1000
    connectivity = 10
    prob_new_edge = 0.05
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
    circle = True

elif network == 'WS-2':
    n_players = 1000
    connectivity = 6
    prob_new_edge = 0.175
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
    circle = True

elif network == 'WS-3':
    n_players = 1000
    connectivity = 20
    prob_new_edge = 0.009
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
    circle = True

else:
    raise Exception('You have chosen a network type which is not defined.')


# Plot graph
sizes = [graph.degree(i)*marker_size for i in range(graph.order())]
if plot_graph:
    if circle:
        nx.draw_circular(graph, with_labels=False, node_size=sizes)
    else:
        nx.draw_kamada_kawai(graph, with_labels=False, node_size=sizes)

# Plot degree distrib
degree = [graph.degree(i) for i in range(graph.order())]
plt.figure(figsize=(6, 6))
hist, bin_edges = np.histogram(degree, bins = 100)
bin_centers = np.array([(bin_edges[i_bin]+bin_edges[i_bin+1])/2  for i_bin in range(len(hist))])


if network == 'BA' or network == 'FB':
    bin_centers_fit = []
    hist_fit = []
    for iIterate in range(len(hist)):
        if hist[iIterate] > 0:
            bin_centers_fit.append(bin_centers[iIterate])
            hist_fit.append(hist[iIterate])
    bin_centers_fit = np.array(bin_centers_fit)
    hist_fit = np.array(hist_fit)
    fit = np.polyfit(np.log(bin_centers_fit[:max_index_fit]), np.log(hist_fit[:max_index_fit]), deg = 1)
    plt.plot(bin_centers_fit[:max_index_fit], np.exp(fit[1] + fit[0]*np.log(bin_centers_fit[:max_index_fit])) )

    print('Scale free parameter: {:f}.'.format(fit[0]))

plt.scatter(bin_centers, hist)
plt.xlabel('Degree')
plt.ylabel('# Counts')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.5, 5*np.max(hist))
plt.title(network)

# Average shortest path length
print('Average shortest path length: {:f}'.format(nx.average_shortest_path_length(graph)))

# Clustering coeff
print('Average clustering coeff.: {:f}'.format(nx.average_clustering(graph)))


#Average degree
print('Average degree: {:f}'.format(np.average(  [graph.degree(i) for i in range(graph.order())] )))

plt.show()
