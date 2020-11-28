import numpy as np


def soft_noisy_update_according_to_best_neighbor(player_money: float,
                                                 player_strategy: float,
                                                 player_payoff: float,
                                                 neighbors_strategy: np.array,
                                                 neighbors_payoff: np.array,
                                                 alpha: float,
                                                 noise_intensity: float):
    best_neighbor_strategy = neighbors_strategy[np.argmax(neighbors_payoff)]
    noise = np.random.normal(0, noise_intensity)
    return np.clip(alpha*best_neighbor_strategy + (1-alpha)*player_strategy + noise, 0, player_money)