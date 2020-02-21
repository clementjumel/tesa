import torch


# region Metrics

def precision_at_k(ranks, targets):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    k = 10

    if targets.sum() == 0:
        return None

    mask = ranks <= k
    targets = targets[mask]

    return targets.sum().type(dtype=torch.float)/float(k)


def recall_at_k(ranks, targets):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    k = 10

    n_relevant = targets.sum()

    if n_relevant == 0:
        return None

    mask = ranks <= k
    targets = targets[mask]

    return targets.sum().type(dtype=torch.float)/float(n_relevant)


def best_rank(ranks, targets):
    """
    Returns the best rank of the relevant answers. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    mask = targets > 0
    ranks = ranks[mask]

    return ranks.min().type(dtype=torch.float)


def average_rank(ranks, targets):
    """
    Returns the average rank of the relevant answers. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    mask = targets > 0
    ranks = ranks[mask]

    return ranks.type(dtype=torch.float).mean()


def reciprocal_best_rank(ranks, targets):
    """
    Returns the inverse of the best rank of the relevant answers. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    r = best_rank(ranks, targets)

    return 1./r if r is not None else None


def reciprocal_average_rank(ranks, targets):
    """
    Returns the inverse of the average rank of the relevant answers. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    r = average_rank(ranks, targets)

    return 1./r if r is not None else None


def average_precision(ranks, targets):
    """
    Compute the Average Precision the ranks and targets line Tensors. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    if targets.sum() == 0:
        return None

    mask = targets > 0
    ranks = ranks[mask].type(dtype=torch.float)
    targets = targets[mask].type(dtype=torch.float)

    p_j = torch.tensor([sum((ranks <= ranks[j]))/ranks[j] for j in range(len(targets))])

    return p_j.mean()


def ndcg(ranks, targets):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    k = 10

    def dcg(ranks, targets, k):
        """
        Compute the Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not relevant
        answer, returns None.

        Args:
            ranks: torch.Tensor, ranks predicted for the batch.
            targets: torch.Tensor, true labels for the batch.
            k: int, number of ranks to take into account.

        Returns:
            torch.tensor, score of the batch.
        """

        mask = ranks <= k
        ranks = ranks[mask]
        targets = targets[mask]

        mask = targets > 0
        ranks = ranks[mask].type(dtype=torch.float)

        return torch.div(1., torch.log2(ranks + 1.)).sum()

    perfect_ranks = rank(targets.reshape((-1, 1)))

    return dcg(ranks, targets, k)/dcg(perfect_ranks, targets, k)

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


def all_scores(ranks, targets):
    """
    Returns all the scores of the ranks and targets in a dictionary.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        dictionary, all the scores of the batch.
    """

    return {'precision_at_k': precision_at_k(ranks, targets),
            'recall_at_k': recall_at_k(ranks, targets),
            'best_rank': best_rank(ranks, targets),
            'average_rank': average_rank(ranks, targets),
            'reciprocal_best_rank': reciprocal_best_rank(ranks, targets),
            'reciprocal_average_rank': reciprocal_average_rank(ranks, targets),
            'average_precision': average_precision(ranks, targets),
            'ndcg': ndcg(ranks, targets)}

# endregion


def main():
    score = average_precision
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
        ranks = rank(grades.reshape(-1,1))
        print(ranks[2], ranks[6])
        print(score(ranks, targets))


if __name__ == '__main__':
    main()
