import itertools
from math import isclose

try:
    from math import comb # N choose K, requires python 3.8
except:
    from scipy.special import comb

import numpy as np

def calc_naive_shapley_values(v: callable, N: int, players=None, verbose=False):
    """

    Calculate the Shapley value for a value function v, given the number of players N.
    Uses the naive formula, beware of combinatorial explosion.

    args:
    v - a characteristic function, a function that takes a subset of players (as a list) and returns a scalar
    N - number of players
    players - a list of players. Passed to v. If None, defaults to [0,1,...,N-1]
    verbose - if True, print some debugging information

    returns:
    shapley_values = a list of length N, containing the Shapley value for each player

    """

    if players is None:
        players = list(range(N))

    # enforce that v(empty set) = 0
    v_ = lambda S: v(S) if len(S)>0 else 0

    shapley_values = []
    for i in range(N): # iterating over the held-out player
        all_but_i = [j for j in range(N) if j != i]

        # list all possible subsets of players not containing the held-out component.
        all_subsets_excluding_i = list(
            itertools.chain.from_iterable(
                itertools.combinations(all_but_i,r) for r in range(N)
                )
            )

        expected_number_of_subsets = 2**N - 2**(N-1)  # all subsets minus all susbets without the held-out component, including the empty set
        assert len(all_subsets_excluding_i)==expected_number_of_subsets, f"expected {expected_number_of_subsets} subsets, got {len(all_subsets_excluding_i)}"

        if verbose:
            print("\n"+f"calculating shapley value for component {i}")
            print(f"all_subsets_excluding_{i}",all_subsets_excluding_i)

        shapley_value = 0
        for S in all_subsets_excluding_i:

            S_plus_i = sorted(list(S) + [i])

            v_without_i = v_([players[j] for j in S])
            v_with_i = v_([players[j] for j in S_plus_i])

            weight = 1 / (N * comb(N-1,len(S)))

            shapley_value += weight * (v_with_i - v_without_i)

            if verbose:
                print(f"v_without_{i} = v({[j for j in S]})={v_without_i}, v_with_{i} = v({[j for j in S_plus_i]})={v_with_i}, weight = 1 / ({N} * {comb(N-1,len(S))}) = {weight}")

        shapley_values.append(shapley_value)

    assert len(shapley_values) == N, f"expected {N} shapley values, got {len(shapley_values)}"

    # make sure the Efficiency property holds
    assert isclose(sum(shapley_values), v(players), abs_tol=1e-8), f"Efficiency property does not hold. sum(shapley_values)={sum(shapley_values)}, v(set(players))={v(set(players))}"

    return shapley_values


def test_calc_naive_shapley_values():
    # Test case 1: N = 3, v(S) = sum of players in S
    def v(S):
        return sum(S)

    N = 3
    players = [1, 1, 1]
    expected_shapley_values = [1, 1, 1]
    shapley_values = calc_naive_shapley_values(v, N, players=players, verbose=True)
    assert shapley_values == expected_shapley_values, f"Test case 1 failed. Expected {expected_shapley_values}, got {shapley_values}"

    # Test case 2: N = 5, v(S) = sum of players in S
    def v(S):
        return sum(S)
    N = 5
    players = np.random.randint(1, 100, size=N)
    expected_shapley_values = players
    shapley_values = calc_naive_shapley_values(v, N, players=players, verbose=False)
    assert np.isclose(shapley_values, expected_shapley_values).all(), f"Test case 2 failed. Expected {expected_shapley_values}, got {shapley_values}"

    # Test case 3: N = 5, number of players in S
    def v(S):
        return len(S)

    N = 5
    players = np.random.randint(1, 100, size=N)
    expected_shapley_values = [1, 1, 1, 1, 1]
    shapley_values = calc_naive_shapley_values(v, N, players=players, verbose=False)
    assert np.isclose(shapley_values, expected_shapley_values).all(), f"Test case 3 failed. Expected {expected_shapley_values}, got {shapley_values}"

    # Test case 4: N = 5, v(S) = 1 if player 3 is in S, 0 otherwise
    def v(S):
        return 1 if 3 in S else 0

    N = 5
    expected_shapley_values = [0, 0, 0, 1, 0]
    shapley_values = calc_naive_shapley_values(v, N, verbose=False)
    assert np.isclose(shapley_values, expected_shapley_values).all(), f"Test case 4 failed. Expected {expected_shapley_values}, got {shapley_values}"

    # Test case 5: N = 5, v(S) = sum of squares of players in S
    def v(S):
        return sum([x**2 for x in S])
    N = 5
    players = np.random.randint(1, 100, size=N)
    expected_shapley_values = players ** 2
    shapley_values = calc_naive_shapley_values(v, N, players=players, verbose=False)
    assert np.isclose(shapley_values, expected_shapley_values).all(), f"Test case 5 failed. Expected {expected_shapley_values}, got {shapley_values}"

    # Test case 5: N = 3, v(S) = 1 if all players are in S, 0 otherwise
    print("All tests passed!")

if __name__ == "__main__":
    test_calc_naive_shapley_values()