import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from read_file_net import read_file_net



seed = 0
np.random.seed(seed)

network = 'WS' # 'FB', 'BA' or 'WS'
marker_size = 0.5
plot_graph = False

# Initializations
if network == 'FB':
    graph, n_players = read_file_net('facebook_net.txt')
    plot_graph = False

elif network == 'BA':
    n_players = 1000
    m = 3
    graph = nx.barabasi_albert_graph(n_players, m=m, seed=seed)
    circle = True

else:
    n_players = 4000
    connectivity = 44
    prob_new_edge = 0.03
    graph = nx.watts_strogatz_graph(n_players, connectivity, prob_new_edge, seed=seed)
    circle = True


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
plt.hist(degree, bins=100)
plt.xlabel('Degree')
plt.ylabel('# Counts')
plt.xscale('log')
plt.yscale('log')
plt.title(network)

# Average shortest path length
print('Average shortest path length: {:f}'.format(nx.average_shortest_path_length(graph)))

# Clustering coeff
print('Average clustering coeff.: {:f}'.format(nx.average_clustering(graph)))
plt.show()
