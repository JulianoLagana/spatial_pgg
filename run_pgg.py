"""
This scipt is one possible implementation on how to use the NetworkPGG class to simulate results,
in a modular fashion. It uses a number of argparse arguments, that zou should make yourself familiar with.

To get some initial help type:
    'python3 run_pgg.py --help'
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from plot_utils import LinkedPlotter, avgPlotter, changePlotter
from modular_pgg import NetworkPGG, graph_constructor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=str, choices=["cluster", "spread"], help='Declare which simulation to run (cluster, spread).')
    parser.add_argument('--network', type=str, choices=["WS", "BA", "FA"], help='Declare wether to use WS, BA, or FA network.')
    parser.add_argument('--out_path', type=str, default=None, help='Possibilitz to give a specific out_path for the plots.')
    parser.add_argument('--player', default=100, type=int, help='Integer value for the number of players.')
    parser.add_argument('--plot', action='store_true', help='If added the plots will be plotted.')
    parser.add_argument('--r', default=3, type=float, help='Float value for the multiplication factor r.')
    parser.add_argument('--rounds', default=100, type=int, help='Integer value for the number of rounds played per simulation.')
    parser.add_argument('--save', action='store_true', help='If added the plots will be saved.')
    parser.add_argument('--seed', default=None, type=int, help='Integer value for random seed.')
    
    return parser


def cluster(args):
    # Optionally set seed for reproducibility
    if reproducible:
        seed = SEED
        np.random.seed(seed)
    else:
        seed = None

    soft_noisy_update_according_to_best_neighbor
    circle = True if args.network=="WS" else False
    log_scale = True # For the scatter plot
    size_marker = 0.5
    pass


def plot(graph, contribution_curves):
    circle = True if args.network=="WS" else False
    log_scale = True # For the scatter plot
    size_marker = 0.5

    

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
    avgPlotter(graph, contribution_curves, mean_contribs, ax2[0], ax2[1], log_scale=log_scale, size_marker=size_marker)
    if save_plots:
        fig2.savefig('fig/P2_median-'+str(n_players)+'_'+str(noise_intensity)+'.png')

    plt.show()


def spread(args):
    """This function has been implemented to investigate 
        how cooperation can spread from clusters.
    """
    if args.seed:
        np.random.seed(args.seed)
    else:
        seed = None

    graph = graph_constructor(args, connectivity=4, prob_new_edge=0.3)
    PGG = NetworkPGG(args, graph)
    PGG.simulate()
    changePlotter(PGG.graph, PGG.contribution_curves, [-1])

    # Intialize random cluster of players to 100% contribution and rest to zero.
    player_strategies = np.zeros(shape=PGG.n_players)
    start = np.random.randint(0, PGG.n_players)
    player_strategies[start] = PGG.starting_money
    start = list(graph.adj[start])
    for i in range(1): #set depth of cluster
        new = []
        for a in start:
            new += list(graph.adj[a])
            player_strategies[a] = PGG.starting_money
        start = new
    PGG.player_strategies = player_strategies

    PGG.simulate()

    changePlotter(PGG.graph, PGG.contribution_curves, [0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 99])


def main(args):
    if args.simulation == "cluster":
        cluster(args)
        print(args.simulation)

    if args.simulation == "spread":
        spread(args)
        print(args.simulation)



if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)


