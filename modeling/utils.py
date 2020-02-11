import torch


# region Metrics

def ap(y_rank, y_true):
    """
    Compute the AP (Averaged Precision) for line torch.Tensors.

    Args:
        y_rank: torch.Tensor, rank predicted.
        y_true: torch.Tensor, true labels.

    Returns:
        torch.Tensor, score of the data.
    """

    n, s = len(y_rank), sum(y_true)
    if s == 0:
        return torch.tensor(0.)

    p = torch.tensor([sum([y_true[k] for k in range(n) if y_rank[k] <= y_rank[j]])/y_rank[j] for j in range(n)])

    return torch.div(torch.dot(p, y_true), s)


def ap_at_k(y_rank, y_true, k):
    """
    Compute the AP (Averaged Precision) at k for a line or column torch.Tensor.

    Args:
        y_rank: torch.Tensor, rank predicted.
        y_true: torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    y_rank, y_true = flatten_tensors(y_rank, y_true)

    mask = torch.tensor(y_rank <= k)

    y_rank = y_rank[mask]
    y_true = y_true[mask]

    return ap(y_rank=y_rank, y_true=y_true)


def dcg(y_rank, y_true, k):
    """
    Compute the DCG (Discounted Cumulative Gain) at k for line torch.Tensors.

    Args:
        y_rank: torch.Tensor, rank predicted
        y_true: torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    mask = torch.tensor(y_rank <= k)

    y_rank = y_rank[mask]
    y_true = y_true[mask]

    g = 2**y_true - 1
    d = torch.div(1., torch.log2(y_rank + 1))

    return torch.dot(g, d)


def ndcg(y_rank, y_true, k):
    """
    Compute the NDCG (Normalized Discounted Cumulative Gain) at k for line or column torch.Tensors.

    Args:
        y_rank: torch.Tensor, rank predicted
        y_true: torch.Tensor, true labels.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    y_rank, y_true = flatten_tensors(y_rank, y_true)

    y_rank_perfect = rank(y_true)

    return dcg(y_rank, y_true, k)/dcg(y_rank_perfect, y_true, k)

# endregion


# region Other methods

def rank(grades):
    """
    Rank according to the grades (rank is 1 for highest grade). Deal with draws by assigning the best rank to the first
    grade encountered.

    Args:
        grades: torch.Tensor, grades in line or column.

    Returns:
        torch.Tensor, rank predictions, with the same shape as grades.
    """

    shape = grades.shape

    if len(shape) == 2:
        grades = torch.reshape(grades, (-1,))

    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    inv = torch.zeros(n)
    inv[sorter] = torch.arange(1., n+1.)

    return torch.reshape(inv, shape=shape)


def flatten_tensors(y1, y2):
    """
    Check the coherence of the shapes between the inputs, that they are either column or line tensors, detach and
    flatten them if necessary.

    Args:
        y1: torch.Tensor, first tensor.
        y2: torch.Tensor, second tensor.

    Returns:
        y1: torch.Tensor, flattened first tensor.
        y2: torch.Tensor, flattened second tensor.
    """

    y1 = y1.detach()
    y2 = y2.detach()

    assert y1.shape == y2.shape
    assert len(y1.shape) == 1 or (len(y1.shape) == 2 and y1.shape[1] == 1)

    if len(y1.shape) == 2:
        y1 = torch.reshape(y1, (-1,))
        y2 = torch.reshape(y2, (-1,))

    return y1, y2

# endregion
