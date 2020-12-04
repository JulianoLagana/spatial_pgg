import numpy as np


def soft_noisy_update_according_to_best_neighbor(player_money: float,
                                                 player_strategy: float,
                                                 player_payoff: float,
                                                 neighbors_strategy: np.array,
                                                 neighbors_payoff: np.array,
                                                 alpha: float,
                                                 noise_intensity: float) -> float:
    """
    Function used to update a player's strategy given knowledge about what other players played and their payoffs in the
     previous round. This essentially copies the strategy of the best performing neighbor, but in a soft way (hence
     alpha), and after the strategy is copied some noise is added to it.
    @param player_money: Maximum value this player can contribute in each round.
    @param player_strategy: How much money this player contributed in the previous round.
    @param player_payoff: Payoff received by this player from playing in the previous round.
    @param neighbors_strategy: Strategies used by this player's neighbors in the previous round.
    @param neighbors_payoff: Payoff received by each of this player's neighbors.
    @param alpha: Decides how soft the update will be. 0 corresponds to no update, 1 to hard update.
    @param noise_intensity: Intensity of the noise added to the resulting computed strategy.
    @return: Updated strategy for this player.
    """
    if player_payoff < np.max(neighbors_payoff):
        best_neighbor_strategy = neighbors_strategy[np.argmax(neighbors_payoff)]
        noise = np.random.normal(0, noise_intensity)
        new_strategy = np.clip(alpha*best_neighbor_strategy + (1-alpha)*player_strategy + noise, 0, player_money)
        return float(new_strategy)
    else:
        return player_strategy
