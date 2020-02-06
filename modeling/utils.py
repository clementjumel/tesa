from math import log2
from scipy.stats import rankdata


# region Metrics

def dcg(y_pred, y_true, k):
    """
    Compute the DCG (Discounted Cumulative Gain) at k of the prediction.

    Args:
        y_pred: 1d np.array, labels predicted
        y_true: 1d np.array, true labels.
        k: int, number of ranks to take into account.

    Returns:
        float, score of the data.
    """

    def g(j):
        """ Gain function. """

        return 2.**y_true[j] - 1.

    def d(j):
        """ Position discount function. """

        return 1./log2(1. + y_pred[j])

    return sum([g(j)*d(j) for j in range(len(y_pred)) if y_pred[j] <= k])


def ndcg(y_pred, y_true, k):
    """
    Compute the NDCG (Normalized Discounted Cumulative Gain) at k of the prediction.

    Args:
        y_pred: 1d np.array, labels predicted
        y_true: 1d np.array, true labels.
        k: int, number of ranks to take into account.

    Returns:
        float, score of the data.
    """

    y_pred_perfect = rank(y_true)

    return dcg(y_pred, y_true, k)/dcg(y_pred_perfect, y_true, k)


def ap(y_pred, y_true):
    """
    Compute the AP (Averaged Precision).

    Args:
        y_pred: 1d np.array, labels predicted
        y_true: 1d np.array, true labels.

    Returns:
        float, score of the data.
    """

    n = len(y_true)

    def p(j):
        """ Precision until the position of d_ij for q_i. """

        return sum([y_true[k] for k in range(n) if y_pred[k] <= y_pred[j]])/y_pred[j]

    return sum([p(j)*y_true[j] for j in range(n)])/sum(y_true)

# endregion


# region Other methods

def rank(grades):
    """
    Rank according to the grades (rank is 1 for highest grade).

    Args:
        np.array, grades.

    Returns:
        np.array, rank predictions.
    """

    return rankdata([-grade for grade in grades], method='ordinal')

# endregion
