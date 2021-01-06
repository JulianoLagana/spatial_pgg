import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np

from joblib import Parallel, delayed

from pgg import compute_pgg_neighborhood_wise_payoffs, compute_pgg_neighborhood_wise_payoffs_old, compute_pgg_layered_payoffs
from update_strategies import soft_noisy_update_according_to_best_neighbor
from read_file_net import read_file_net


def graph_constructor(args, m=None, connectivity=None, prob_new_edge=None):
    """This function constructs the graph given by the argsparse arguments and function parameters.
    """
    if args.network == 'FB':
        graph, _ = read_file_net('facebook_net.txt')

    elif args.network == 'BA':
        if m == None:
            raise Warning("Missing arguments for the choosen network.")
        graph = nx.barabasi_albert_graph(args.player, m=m, seed=args.seed)

    elif args.network == "WS":
        if connectivity == None or prob_new_edge == None:
            raise Warning("Missing arguments for the choosen network.")
        graph = nx.watts_strogatz_graph(args.player, connectivity, prob_new_edge, seed=args.seed)
    
    print(f'Constructed {args.network} graph:')
    # Average shortest path length
    print('Average shortest path length: {:f}'.format(nx.average_shortest_path_length(graph)))
    # Clustering coeff
    print('Average clustering coeff.: {:f}'.format(nx.average_clustering(graph)))
    #Average degree
    print('Average degree: {:f}'.format(np.average([graph.degree(i) for i in range(graph.order())])))

    return graph


class NetworkPGG():
    """
    Modular PGG class that works with different update and pay-off engines.

    Arguments:

        -args: dictionary with argparse arguments
        -graph: network of the networkx graph class

    """
    def __init__(self, args, graph, starting_money=100, alpha=0.5, noise_intensity=1, payoff_engines=compute_pgg_neighborhood_wise_payoffs, update_engine=soft_noisy_update_according_to_best_neighbor, countries=None):
        np.random.seed(seed=args.seed)
        self.graph = graph
        self.n_players = graph.number_of_nodes()
        self.update_engine = update_engine
        self.payoff_engines = payoff_engines
        self.num_cores = multiprocessing.cpu_count()

        # Hyperparameters for the simulation
        self.starting_money = starting_money
        self.n_rounds = args.rounds
        self.alpha = alpha
        self.noise_intensity = noise_intensity
        self.mult_factor = args.r
        
        # Simulation variables
        self.players_money = np.array([self.starting_money] * self.n_players)
        self.player_strategies = np.random.random(size=self.n_players)*self.starting_money
        if countries:
            # Change number of countries
            players = np.array(list(graph.nodes))
            # Change number of countries
            self.countries = np.array_split(players, countries)
        else:
            self.countries = None


    def simulate(self):
        """Simulate the game for self.n_rounds and calculate the contibutions for given settings.
        """
        self.contribs = np.zeros((self.n_rounds+1, self.n_players))
        self.contribs[0, :] = self.player_strategies.copy()

        # data structure for the mean plot
        self.mean_contribs = np.zeros((3, self.n_rounds+1)) 
        self.mean_contribs[:, 0] = [np.median(self.player_strategies),
                            np.percentile(self.player_strategies, 25),
                            np.percentile(self.player_strategies, 75)]
        self.payoffs = None

        for i_round in range(self.n_rounds):
            if i_round % 10 == 0:
                print('Round: {:d}'.format(i_round))
            # Play one round

            self.payoffs = self.payoff_engines(self.graph, self.players_money, self.player_strategies, self.mult_factor)

            if self.countries:
                self.payoffs = compute_pgg_layered_payoffs(self.graph, self.players_money, self.player_strategies, self.mult_factor, self.countries)
            
                # payoffs = self.payoffs.copy()
                # gamma = 0.5
                # Change utility of players to incorporate minumum empathy pay-off
                # self.payoffs = [payoffs[i] + (gamma * np.min(payoffs[list(self.graph.adj[i])])) for i in range(self.n_players)]
                # Change utility of players to incorporate avg empathy pay-off
                # self.payoffs = [payoffs[i] + (gamma * np.mean(payoffs[list(self.graph.adj[i])])) for i in range(self.n_players)]

            # Update the players strategies
            new_player_strategies = Parallel(n_jobs=self.num_cores)(delayed(self.parallel_function)(i_player, list(self.graph.adj[i_player]), self.player_strategies, self.payoffs, self.players_money, self.alpha, self.noise_intensity) for i_player in range(len(self.player_strategies)))
            self.player_strategies = np.array(new_player_strategies)
            self.mean_contribs[:, i_round+1] = [np.median(self.player_strategies),
                                        np.percentile(self.player_strategies, 25),
                                        np.percentile(self.player_strategies, 75)] # for mean plot
            self.contribs[i_round+1, :] = self.player_strategies.copy() # Save contributions made this round

        # Change the format of the saved contributions for plotting
        xs = [i for i in range(self.n_rounds+1)]
        self.contribution_curves = []
        for i_player in range(self.n_players):
            self.contribution_curves.append([xs, self.contribs[:, i_player]])
    
    
    def parallel_function(self, i_player, neighbor_idxs, player_strategies, payoffs, players_money, alpha, noise_intensity):
        neighbor_strats = [player_strategies[i] for i in neighbor_idxs]
        neighbor_payoffs = [payoffs[i] for i in neighbor_idxs]
        new_player_strategy = self.update_engine(players_money[i_player],
                                                player_strategies[i_player],
                                                payoffs[i_player],
                                                neighbor_strats,
                                                neighbor_payoffs,
                                                alpha,
                                                noise_intensity)
        return new_player_strategy


if __name__ == "__main__":
    pass
