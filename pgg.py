import numpy as np


def compute_pgg_payoffs(players_money: np.ndarray, contributions: np.ndarray, mult_factor: float) -> float:
    assert len(players_money) == len(contributions), '`players_money` has to have the same number of elements as `contributions`'
    assert np.all(contributions <= players_money), f'At least one of the players is contributing more money than he has!' \
                                                   f'\nContributions: {contributions}' \
                                                   f'\nPlayers money: {players_money}'

    # Compute dividend (same for all players)
    divvy = np.sum(contributions)/len(contributions)*mult_factor

    # Return resulting money for all players
    return players_money - contributions + divvy


players_money = np.array([100, 100, 100])
contributions = np.array([0, 100, 20])

players_money = compute_pgg_payoffs(players_money, contributions, 1.5)
print(players_money)
