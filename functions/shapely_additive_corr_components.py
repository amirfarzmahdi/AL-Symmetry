import numpy as np
from scipy.stats import pearsonr

from shapley import calc_naive_shapley_values

def shapley_values_for_additive_components(y, X, corr_fun=None):
    """
    Compute the Shapley values of additive components of x for the value function corr(y,x).

    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        The target values.

    X : array-like, shape (n_samples, n_features)
        The additive components of x, such that x = X.sum(axis=1)

    corr_fun : callable, default None
        The function used to compute the correlation between y and x.
        If None, then scipy.stats.pearsonr is used.

    Returns:
    --------
    shapley_values : array-like, shape (n_features,)
        The Shapley values of the additive components of x.
    """
    import numpy as np

    if corr_fun is None:
        corr_fun = lambda x,y: pearsonr(x,y).statistic

    def v(S):
        if len(S)>0:
            x = X[:,list(S)].sum(axis=1)
            if np.isclose(np.var(x),0):
                return 0 # if x.sum() is close to zero, count the correlation as zero
            else:
                value = corr_fun(y,x)
                assert not np.isnan(value), f"corr_fun(y,x) returned nan for y={y}, x={x}"
                return value
        else:
            return 0

    N = X.shape[1]
    shapley_values = calc_naive_shapley_values(v, N)
    return shapley_values

def example1():
    y = np.array([1,
                  2,
                  3,
                  4,])

    X = np.array([[1, 0,  0],
                  [2, 0,  0],
                  [3, 0,  0],
                  [3, 1, -1],])

    print("Example 1:")
    # visualize X as a matrix in the terminal:

    print("y:")
    print(np.atleast_2d(y).T)

    print("X:")
    print(X)

    print("correlation between y and x:", pearsonr(y,X.sum(axis=1)).statistic)
    shapley_values = shapley_values_for_additive_components(y,X)
    print(f"shapley_values={shapley_values}")

def example2():
    y = np.array([1,
                  2,
                  3,
                  4,])

    X = np.array([[1, 1,  0],
                  [2, 2,  0],
                  [3, 3,  0],
                  [4, 4,  0],])

    print("Example 2:")
    # visualize X as a matrix in the terminal:

    print("y:")
    print(np.atleast_2d(y).T)

    print("X:")
    print(X)

    print("correlation between y and x:", pearsonr(y,X.sum(axis=1)).statistic)
    shapley_values = shapley_values_for_additive_components(y,X)
    print(f"shapley_values={shapley_values}")

if __name__ == "__main__":
    example1()
    example2()


# if corr_fun is None:
#         from scipy.stats import pearsonr
#         corr_fun = lambda y,x: pearsonr(y,x).statistic


#     Explanation:

#     X describes additive components of x, such that x = X.sum(axis=1)
#     We are interested in the Shapley values of each component for the value
#     function corr(y,x).

#     the resulting Shapley values are in the same order as the columns of X,
#     and sum to the correlation between y and x.

#     Example:
#     y = np.array([1,2,3])
#     X = np.array([[1,0],[2,-1],[3,0]])

#     here, the first component (1,2,3) is perfeclty correlated with y,
#     and the second component (0,-1,0) reduces the correlation.

#     Therefore, we expect the Shapley value for the first component to be high and positive,
#     and the Shapley value for the second component to be negative.

#     Their sum should be equal to the correlation between y and x.
#     corr(y,x) = corr([1,2,3],[1,1,3]) = 0.866

# if __name__ == "__main__":
#     y = np.array([1,2,3])
#     X = np.array([[1,0],[2,-1,],[3,0]])

#     shapley_values = naive_shapley(y,X, debug=True)
#     print(f"shapley_values={shapley_values}")


#        x_without_i = X[:,list(S)].sum(axis=1)
#             S_plus_i = sorted(list(S) + [i])
#             x_with_i = X[:,S_plus_i].sum(axis=1)
