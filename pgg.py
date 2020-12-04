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


def compute_pgg_neighborhood_wise_payoffs(graph: networkx.classes.graph.Graph, players_money: np.ndarray,
                                          players_stratategies: np.ndarray, mult_factor: float) -> np.ndarray:
    n_players = len(graph.nodes)
    payoffs = np.zeros((n_players,))
    for i_player in list(graph.nodes):
        neigh_idxs = list(graph.adj[i_player])
        subgame_players = neigh_idxs + [i_player]
        subgame_moneys = players_money[subgame_players]
        subgame_strategies = players_stratategies[subgame_players]
        payoffs[i_player] = compute_pgg_payoffs(subgame_moneys, subgame_strategies, mult_factor)[-1]
    return payoffs


# Example usage
if __name__ == '__main__':
    players_money = np.array([100, 100, 100])
    contributions = np.array([0, 100, 20])

    payoffs = compute_pgg_payoffs(players_money, contributions, 1.5)
    players_money = players_money + payoffs
    print(players_money)
