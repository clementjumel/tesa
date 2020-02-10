import torch


# region Metrics

def ap(y_pred, y_true):
    """
    Compute the AP (Averaged Precision).

    Args:
        y_pred: 1D torch.Tensor, labels predicted.
        y_true: 1D torch.Tensor, true labels.

    Returns:
        float, score of the data.
    """

    assert len(y_pred.shape) == 1 and y_pred.shape == y_true.shape

    n, s = len(y_pred), sum(y_true)
    if s == 0:
        return 0.

    p = torch.tensor([sum([y_true[k] for k in range(n) if y_pred[k] <= y_pred[j]])/y_pred[j] for j in range(n)])

    return float(torch.div(torch.dot(p, y_true), s))


def ap_at_k(y_pred, y_true, k):
    """
    Compute the AP (Averaged Precision) at k.

    Args:
        y_pred: 1D torch.Tensor, labels predicted
        y_true: 1D torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        float, score of the data.
    """

    assert len(y_pred.shape) == 1 and y_pred.shape == y_true.shape

    mask = torch.tensor(y_pred <= k)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return ap(y_pred=y_pred, y_true=y_true)


def dcg(y_pred, y_true, k):
    """
    Compute the DCG (Discounted Cumulative Gain) at k of the prediction.

    Args:
        y_pred: 1D torch.Tensor, labels predicted
        y_true: 1D torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        float, score of the data.
    """

    assert len(y_pred.shape) == 1 and y_pred.shape == y_true.shape

    mask = torch.tensor(y_pred <= k)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    g = 2**y_true - 1
    d = torch.div(1., torch.log2(y_pred + 1))

    return float(torch.dot(g, d))


def ndcg(y_pred, y_true, k):
    """
    Compute the NDCG (Normalized Discounted Cumulative Gain) at k of the prediction.

    Args:
        y_pred: 1D torch.Tensor, labels predicted
        y_true: 1D torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        float, score of the data.
    """

    assert len(y_pred.shape) == 1 and y_pred.shape == y_true.shape

    y_pred_perfect = rank(y_true)

    return dcg(y_pred, y_true, k)/dcg(y_pred_perfect, y_true, k)

# endregion


# region Other methods

def rank(grades):
    """
    Rank according to the grades (rank is 1 for highest grade). Deal with draws by assigning the best rank to the first
    grade encountered.

    Args:
        grades: 1D torch.Tensor, grades.

    Returns:
        1D torch.Tensor, rank predictions.
    """

    assert len(grades.shape) == 1
    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    inv = torch.zeros(n)
    inv[sorter] = torch.arange(1., n+1.)

    return inv


def progression(count, modulo, size, text):
    """
    Prints progression's updates and update the count.

    Args:
        count: int, current count.
        modulo: int, how often to print updates.
        size: int, size of the element to count.
        text: str, what to print at the beginning of the updates.

    Returns:
        int, incremented count of articles.
    """

    count += 1

    if count % modulo == 0:
        print("   " + text + " {}/{}...".format(count, size))

    return count

# endregion
