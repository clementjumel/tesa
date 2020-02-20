import torch


# region Metrics

def precision_at_k(ranks, targets, k):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    mask = ranks <= k
    targets = targets[mask]

    return targets.sum().type(dtype=torch.float)/float(k)


def recall_at_k(ranks, targets, k):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    n_relevant = targets.sum()

    if n_relevant == 0:
        return None

    mask = ranks <= k
    targets = targets[mask]

    return targets.sum().type(dtype=torch.float)/float(n_relevant)


def best_rank(ranks, targets, k):
    """
    Returns the best rank of the relevant answers. If k is specified, consider only the first k ranks retrieved. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    if k:
        mask = ranks <= k
        ranks = ranks[mask]
        targets = targets[mask]

    mask = targets > 0
    ranks = ranks[mask]

    return ranks.min().type(dtype=torch.float)


def average_rank(ranks, targets, k):
    """
    Returns the average rank of the relevant answers. If k is specified, consider only the first k ranks retrieved. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    if k:
        mask = ranks <= k
        ranks = ranks[mask]
        targets = targets[mask]

    mask = targets > 0
    ranks = ranks[mask]

    return ranks.type(dtype=torch.float).mean()


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

    else:
        targets = targets.type(dtype=torch.float)

        p = torch.tensor([sum([targets[k] for k in range(n) if ranks[k] <= ranks[j]]) / ranks[j]
                          for j in range(n)])

        return torch.div(torch.dot(p, targets), s)


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

# endregion


# region Other methods

def rank(outputs):
    """
    Rank according to the outputs (1 for highest grade). Deal with draws by assigning the best rank to the first
    output encountered.

    Args:
        outputs: torch.Tensor, (batch_size, 1 ou 2) tensors outputs.

    Returns:
        torch.Tensor, ranks corresponding to the grades in a line Tensor.
    """

    grades = outputs[:, -1]

    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    rank = torch.zeros(n)

    rank[sorter] = torch.arange(1, n + 1).type(dtype=torch.float)

    return rank

# endregion


def main():
    score = ap_at_k
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
        print(score(ranks, targets, k))


if __name__ == '__main__':
    main()
