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
    parser.add_argument('--simulation', type=str, choices=["cluster", "spread", "layered"], help='Declare which simulation to run (cluster, spread).')
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
    """This function has been implemented to investigate 
        how the clustering coefficients affects cooperation in WS.
    """

    circle = True if args.network=="WS" else False
    log_scale = True 
    size_marker = 0.5

    for i in [4, 5, 6, 10, 12, 20, 30]:
        graph = graph_constructor(args, connectivity=i, prob_new_edge=0.3)
        PGG = NetworkPGG(args, graph)
        PGG.simulate()

        if args.plot:
            # Create plotting window
            fig, ax = plt.subplots(ncols=2, figsize=(15, 6))

            ax[0].set_title('P2: Graph (hover a node to outline its contribution)')
            ax[1].set_title('P2: Contributions over time, n='+str(PGG.n_players)+', stoch.='+str(PGG.noise_intensity))
            ax[1].set_xlabel('Round number')
            ax[1].set_ylabel('Contributions')

            # Plot graph and curves
            linked_plotter = LinkedPlotter(PGG.graph, PGG.contribution_curves, ax[0], ax[1], fig, circle=circle)
            if args.save:
                fig.savefig('fig/P2_cluster_graph-'+str(PGG.n_players)+'_'+str(PGG.noise_intensity)+'.png')
            plt.show()

        # Plot scatter of contributions and avg. in a different figure
        fig2, ax2 = plt.subplots(ncols=2, figsize=(15, 6))
        ax2[0].set_title('P2: Contribution vs connectivity')
        ax2[0].set_xlabel('Degree')
        ax2[0].set_ylabel('Average contribution')
        ax2[1].set_title('P2: Median contribution over time (quart. percentiles), n='+str(PGG.n_players)+', stoch.='+str(PGG.noise_intensity))
        ax2[1].set_xlabel('Round number')

        # Plot average contribution vs degree and average contribution level
        avgPlotter(PGG.graph, PGG.contribution_curves, PGG.mean_contribs, ax2[0], ax2[1], log_scale=log_scale, size_marker=size_marker)
        if args.save:
            fig2.savefig('fig/P2_median-'+str(PGG.n_players)+'_'+str(PGG.noise_intensity)+'.png')

        plt.show()


def spread(args):
    """This function has been implemented to investigate 
        how cooperation can spread from a single clusters.
    """

    graph = graph_constructor(args, connectivity=4, prob_new_edge=0.3)
    PGG = NetworkPGG(args, graph)
    PGG.simulate()

    args.out_path += "random"
    changePlotter(PGG.graph, PGG.contribution_curves, [-1], args)
    args.out_path = args.out_path[:-len("random")]

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
    args.out_path += "spread"
    changePlotter(PGG.graph, PGG.contribution_curves, [0, 2, 5, 10, 15, 20, 40, 80, 99], args)
    args.out_path = args.out_path[:-len("spread")]


def PGG_03(args):
    """This function has been implemented to investigate 
        playing multiple subgames (country interpretation).
    """
    circle = True if args.network=="WS" else False
    log_scale = True 
    size_marker = 0.5

    graph = graph_constructor(args, connectivity=4, prob_new_edge=0.3)
    PGG = NetworkPGG(args, graph, countries=5)
    PGG.simulate()


    if args.plot:
        # Create plotting window
        fig, ax = plt.subplots(ncols=2, figsize=(15, 6))

        ax[0].set_title('P2: Graph (hover a node to outline its contribution)')
        ax[1].set_title('P2: Contributions over time, n='+str(PGG.n_players)+', stoch.='+str(PGG.noise_intensity))
        ax[1].set_xlabel('Round number')
        ax[1].set_ylabel('Contributions')

        # Plot graph and curves
        linked_plotter = LinkedPlotter(PGG.graph, PGG.contribution_curves, ax[0], ax[1], fig, circle=circle, country=PGG.countries)
        if args.save:
            fig.savefig('fig/P3_cluster_graph-'+str(PGG.n_players)+'_'+str(PGG.noise_intensity)+'.png')
        plt.show()

    # Plot scatter of contributions and avg. in a different figure
    fig2, ax2 = plt.subplots(ncols=2, figsize=(15, 6))
    ax2[0].set_title('P2: Contribution vs connectivity')
    ax2[0].set_xlabel('Degree')
    ax2[0].set_ylabel('Average contribution')
    ax2[1].set_title('P2: Median contribution over time (quart. percentiles), n='+str(PGG.n_players)+', stoch.='+str(PGG.noise_intensity))
    ax2[1].set_xlabel('Round number')

    # Plot average contribution vs degree and average contribution level
    avgPlotter(PGG.graph, PGG.contribution_curves, PGG.mean_contribs, ax2[0], ax2[1], log_scale=log_scale, size_marker=size_marker)
    if args.save:
        fig2.savefig('fig/P3_median-'+str(PGG.n_players)+'_'+str(PGG.noise_intensity)+'.png')

    plt.show()


def main(args):
    if args.seed:
        np.random.seed(args.seed)

    if not args.out_path:
        args.out_path = str(args.rounds) + "_" + str(args.player) + "_" + args.network + "_" + str(args.r)

    if args.simulation == "cluster":
        cluster(args)
        
    if args.simulation == "spread":
        spread(args)

    if args.simulation == "layered":
        PGG_03(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
