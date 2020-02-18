import torch


# region Metrics

def ap(ranks, targets):
    """
    Compute the Averaged Precision between the ranks and targets line Tensors.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.Tensor, score of the data.
    """

    n, s = len(targets), sum(targets)

    if s == 0:
        return torch.tensor(0.)

    p = torch.tensor([sum([targets[k] for k in range(n) if ranks[k] <= ranks[j]]) / ranks[j]
                      for j in range(n)])

    return torch.div(torch.dot(p, targets), s)


def ap_at_k(ranks, targets, k):
    """
    Compute the Averaged Precision at k between the ranks and targets line Tensors.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    mask = ranks <= k

    ranks = ranks[mask]
    targets = targets[mask]

    return ap(ranks=ranks, targets=targets)


def dcg(ranks, targets, k):
    """
    Compute the Discounted Cumulative Gain at k for the line Tensors ranks and targets.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    mask = torch.tensor(ranks <= k)

    ranks = ranks[mask]
    targets = targets[mask]

    g = 2 ** targets - 1
    d = torch.div(1., torch.log2(ranks + 1))

    return torch.dot(g, d)


def ndcg(ranks, targets, k):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.Tensor, score of the data.
    """

    perfect_ranks = rank(targets)

    return dcg(ranks, targets, k)/dcg(perfect_ranks, targets, k)

# endregion


# region Other methods

def rank(grades):
    """
    Rank according to the grades (rank is 1 for highest grade). Deal with draws by assigning the best rank to the first
    grade encountered.

    Args:
        grades: torch.Tensor, grades for the batch in a line Tensor.

    Returns:
        torch.Tensor, ranks corresponding to the grades in a line Tensor.
    """

    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    inv = torch.zeros(n)

    inv[sorter] = torch.arange(1., n + 1.)

    return inv

# endregion


def main():
    size, k = 10, 4

    targets = torch.tensor([0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])

    grades1 = torch.rand(size)
    grades2 = torch.zeros(size)
    grades3 = targets.clone()
    grades4 = torch.rand(size)
    grades4[2] = 1.
    grades4[6] = -0.4
    grades5 = torch.tensor([0.5, 0.5, 0.45, 0.5, 0, 0., -0.4, 0., 0., 0.])

    for grades in [grades1, grades2, grades3, grades4, grades5]:
        ranks = rank(grades)
        print(ranks[2], ranks[6])
        print(ap_at_k(ranks, targets, k))


if __name__ == '__main__':
    main()
