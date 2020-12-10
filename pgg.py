import numpy as np
import networkx


def compute_pgg_payoffs(players_money: np.ndarray, contributions: np.ndarray, mult_factor: float) -> np.ndarray:
    assert len(players_money) == len(contributions), '`players_money` has to have the same number of elements as `contributions`'
    assert np.all(contributions <= players_money), f'At least one of the players is contributing more money than he has!' \
                                                   f'\nContributions: {contributions}' \
                                                   f'\nPlayers money: {players_money}'

    # Compute dividend (same for all players)
    divvy = np.sum(contributions)/len(contributions)*mult_factor

    # Return payoffs for each player
    return divvy - contributions


def compute_pgg_neighborhood_wise_payoffs_old(graph: networkx.classes.graph.Graph, players_money: np.ndarray,
                                              players_stratategies: np.ndarray, mult_factor: float) -> np.ndarray:
    """
    This function computes the payoffs for each player in our old formulation of the graph-PGG. Each player gets as
    final payoff only the payoff that it received when playing the PGG centered in it.
    """
    n_players = len(graph.nodes)
    payoffs = np.zeros((n_players,))
    for i_player in list(graph.nodes):
        neigh_idxs = list(graph.adj[i_player])
        subgame_players = neigh_idxs + [i_player]
        subgame_moneys = players_money[subgame_players]
        subgame_strategies = players_stratategies[subgame_players]
        payoffs[i_player] = compute_pgg_payoffs(subgame_moneys, subgame_strategies, mult_factor)[-1]
    return payoffs


def compute_pgg_neighborhood_wise_payoffs(graph: networkx.classes.graph.Graph, players_money: np.ndarray,
                                          players_strategies: np.ndarray, mult_factor: float) -> np.ndarray:
    """
    This function computes the payoffs for each player in the graph-PGG. Each player gets as final payoff the average
    payoff it received across all PGGs it played in the graph.
    """
    n_players = len(graph.nodes)
    payoffs = np.zeros(n_players)
    connectivities = np.zeros(n_players)

    for i_player in list(graph.nodes):
        neigh_idxs = list(graph.adj[i_player])
        connectivities[i_player] = len(neigh_idxs)
        subgame_players = neigh_idxs + [i_player]
        subgame_moneys = players_money[subgame_players]
        subgame_strategies = players_strategies[subgame_players]

        # Play the PGG game centered at this node and distribute the payoffs to all participants
        subgame_payoffs = compute_pgg_payoffs(subgame_moneys, subgame_strategies, mult_factor)
        payoffs[subgame_players] += subgame_payoffs

    return payoffs/connectivities


def compute_pgg_layered_payoffs(graph: networkx.classes.graph.Graph, players_money: np.ndarray,
                                          players_stratategies: np.ndarray, mult_factor: float, countries: np.ndarray) -> np.ndarray:
    n_players = len(graph.nodes)
    payoffs = np.zeros((n_players,))
    for subgame_players in countries:
        subgame_moneys = players_money[subgame_players]
        subgame_strategies = players_stratategies[subgame_players]
        payoffs[subgame_players] = compute_pgg_payoffs(subgame_moneys, subgame_strategies, mult_factor)
    return payoffs


# Example usage
if __name__ == '__main__':
    players_money = np.array([100, 100, 100])
    contributions = np.array([0, 100, 20])

    payoffs = compute_pgg_payoffs(players_money, contributions, 1.5)
    players_money = players_money + payoffs
    print(players_money)
