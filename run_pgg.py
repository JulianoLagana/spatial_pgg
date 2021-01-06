"""
This scipt is one possible implementation on how to use the NetworkPGG class to simulate results,
in a modular fashion. It uses a number of argparse arguments, that you should make yourself familiar with.

To get some initial help type:
    'python3 run_pgg.py --help'

Example runs for the three simulations:
    'python3 run_pgg.py --simulation spread --network BA --seed 100 --rounds 300 --player 100 --save '
    'python3 run_pgg.py --simulation spread --network WS --seed 42 --rounds 300 --player 30 --save'
    'python3 run_pgg.py --simulation layered --network WS --seed 42 --rounds 50 --player 1000'
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from plot_utils import LinkedPlotter, avgPlotter, changePlotter
from modular_pgg import NetworkPGG, graph_constructor


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=str, choices=["cluster", "spread", "layered", "test"], help='Declare which simulation to run (cluster, spread).')
    parser.add_argument('--network', type=str, choices=["WS", "BA", "FA"], help='Declare wether to use WS, BA, or FA network.')
    parser.add_argument('--out_path', type=str, default=None, help='Possibilitz to give a specific out_path for the plots.')
    parser.add_argument('--player', default=100, type=int, help='Integer value for the number of players.')
    parser.add_argument('--plot', action='store_true', help='If added the plots will be plotted.')
    parser.add_argument('--r', default=3, type=float, help='Float value for the multiplication factor r.')
    parser.add_argument('--rounds', default=100, type=int, help='Integer value for the number of rounds played per simulation.')
    parser.add_argument('--save', action='store_true', help='If added the plots will be saved.')
    parser.add_argument('--seed', default=0, type=int, help='Integer value for random seed.')
    
    return parser


def cluster(args):
    """This function has been implemented to investigate 
        how the clustering coefficients affects cooperation in WS.
    """

    # plot three graphs with different parameters kept fix
    fig, ax = plt.subplots(ncols=3, figsize=(21, 6))
    
    ax[0].title.set_text(f'Average shortest path lenght of ~{5}')
    ax[1].title.set_text(f'Constant probability of new edge of {100*0.15}%')
    ax[2].title.set_text(f'Constant average degree of {8}')
    ax[0].set_xlabel('Round number')
    ax[0].set_ylabel('Contributions')
    ax[1].set_xlabel('Round number')
    ax[2].set_xlabel('Round number')
    
    # Parameter settings for the WS graph
    paramters = [[(6, 0.175), (10, 0.05), (20, 0.009)], [(5, 0.15), (17, 0.15), (30, 0.15)], [(8, 0.3), (8, 0.15), (8, 0.009)]]
    x = np.arange(0, args.rounds + 1)

    for i in range(3):
        for a in range(3):
            connectivity, prob_new_edge = paramters[i][a]
            graph = graph_constructor(args, connectivity=connectivity, prob_new_edge=prob_new_edge)
            label = 'Clustering coeff.: {:4f}'.format(nx.average_clustering(graph))
            PGG = NetworkPGG(args, graph)
            PGG.simulate()

            y = PGG.contribs.mean(1)
            error = PGG.contribs.std(1)

            ax[i].errorbar(x, y, error, label=label, marker='o', markersize=3, linestyle=':')
            
        ax[i].legend()
    
    if args.save:
        fig.savefig(f'fig/P2_cluster_graph-{str(PGG.n_players)}_.png')
    
    plt.show()


def spread(args):
    """This function has been implemented to investigate 
        how cooperation can spread from a single clusters/neighbourhood.
    """
    if args.network == "WS":
        graph = graph_constructor(args, connectivity=4, prob_new_edge=0.3)
        PGG = NetworkPGG(args, graph)

        # Select center nodes for neighbourhood
        starts = [0, 10, 20, 30, 49]#[4, 22, 55, 80, 99]
        final_contributions = np.zeros((len(starts), args.player))
        start_contributions = np.zeros((len(starts), args.player))
        
        for i in range(0, len(starts)):
            # Intialize cluster of players to 100% contribution and rest to zero.
            player_strategies = np.zeros(shape=PGG.n_players)
            start = starts[i]
            player_strategies[start] = PGG.starting_money
            print("Hub degree: " + str(graph.degree(start)))
            start = list(graph.adj[start])
            for c in range(1): #set depth of cluster
                new = []
                for a in start:
                    new += list(graph.adj[a])
                    player_strategies[a] = PGG.starting_money
                start = new

            hub = np.argwhere(player_strategies)
            print("Average hub degree: " + str(np.array([graph.degree(i) for i in hub]).mean()))

            PGG.player_strategies = player_strategies
            
            PGG.simulate()
            final_contributions[i,:] = PGG.contribs[-1,:]
            start_contributions[i,:] = PGG.contribs[0,:]

            # Plot graphs
            plt = changePlotter(PGG.graph, PGG.contribs, [0, 5, 10, 15, 30, 60, 100, 200, 299], args)
            
            if args.save:
                plt.savefig('fig/spread/Spread_' + args.out_path + '_' + str(i) + '.png')
            plt.close()
        titles = [f"Init: {i}" for i in range(len(starts))]
        titles = titles + titles
        contributions = np.vstack((start_contributions, final_contributions))

        # Final graph of all initialisations
        plt = changePlotter(PGG.graph, contributions, np.arange(2*len(starts)), args, y_labels=["Start","End"], titles=titles)
        
        if args.save:
            plt.savefig('fig/spread/Spread_' + args.out_path + '_init' + '.png')
        plt.show()
        
        args.out_path += "random"
        args.out_path = args.out_path[:-len("random")]

    elif args.network == "BA":
        graph = graph_constructor(args, m=3)

        # Select center nodes for neighbourhood based on degree distribution.
        degrees = np.array([graph.degree(i) for i in range(graph.order())])
        starts = []
        starts.append(np.argmax(degrees))
        starts.append(np.argsort(degrees)[2*len(degrees)//3])
        starts.append(np.argsort(degrees)[len(degrees)//2])
        starts.append(np.argsort(degrees)[len(degrees)//3])
        starts.append(np.argmin(degrees))
        
        PGG = NetworkPGG(args, graph)
        final_contributions = np.zeros((len(starts), args.player))
        start_contributions = np.zeros((len(starts), args.player))
        
        for i in range(0, len(starts)):
            # Intialize random cluster of players to 100% contribution and rest to zero.
            player_strategies = np.zeros(shape=PGG.n_players)
            start = starts[i]
            player_strategies[start] = PGG.starting_money
            start = list(graph.adj[start])
            for c in range(1): #set depth of cluster
                new = []
                for a in start:
                    new += list(graph.adj[a])
                    player_strategies[a] = PGG.starting_money
                start = new
        
            PGG.player_strategies = player_strategies
            
            PGG.simulate()
            final_contributions[i,:] = PGG.contribs[-1,:]
            start_contributions[i,:] = PGG.contribs[0,:]

            # Plot Init
            plt = changePlotter(PGG.graph, PGG.contribs, [0, 5, 10, 15, 40, 60, 80, 100, 200, 299], args)
            
            if args.save: 
                plt.savefig('fig/spread/Spread_' + args.out_path + '_' + str(i) + '.png')
            plt.close()

        titles = [f"Init: {i}" for i in range(len(starts))]
        titles = titles + titles
        contributions = np.vstack((start_contributions, final_contributions))

        # Plot all initialisations
        plt = changePlotter(PGG.graph, contributions, np.arange(2*len(starts)), args, y_labels=["Start","End"], titles=titles)
        
        if args.save:
            plt.savefig(f'fig/spread/Spread_{args.network}_' + args.out_path + '_init' + '.png')
        plt.show()
        
        args.out_path += "random"
        args.out_path = args.out_path[:-len("random")]
    else:
        raise Warning("Simulation not implemented for the specified network. Feel free to add yourself.")


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

    if args.simulation == "test":
        # Test whatever you like.
        graph_constructor(args, connectivity=4, prob_new_edge=0.3)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
