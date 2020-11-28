import numpy as np


def update_according_to_best_neighbor(player_strategy: float,
                                      player_payoff: float,
                                      neighbors_strategy: np.array,
                                      neighbors_payoff: np.array):
    return neighbors_strategy[np.argmax(neighbors_payoff)]


def get_soft_update_according_to_best_neighbor_fun(alpha):
    def soft_update_according_to_best_neighbor(player_strategy: float,
                                               player_payoff: float,
                                               neighbors_strategy: np.array,
                                               neighbors_payoff: np.array):
        best_neighbor_strategy = update_according_to_best_neighbor(player_strategy, player_payoff, neighbors_strategy, neighbors_payoff)
        return alpha*best_neighbor_strategy + (1-alpha)*player_strategy
    return soft_update_according_to_best_neighbor


def get_soft_noisy_update_according_to_best_neighbor_fun(alpha, noise_intensity):
    def soft_noisy_update_according_to_best_neighbor(player_money: float,
                                                     player_strategy: float,
                                                     player_payoff: float,
                                                     neighbors_strategy: np.array,
                                                     neighbors_payoff: np.array):
        best_neighbor_strategy = update_according_to_best_neighbor(player_strategy, player_payoff, neighbors_strategy,
                                                                   neighbors_payoff)
        noise = np.random.normal(0, noise_intensity)
        return np.clip(alpha*best_neighbor_strategy + (1-alpha)*player_strategy + noise, 0, player_money)
    return soft_noisy_update_according_to_best_neighbor
