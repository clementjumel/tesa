from modeling.utils import ranking

import torch


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

    p_j = torch.tensor([torch.sum(torch.ge(-ranks, -ranks[j]))/ranks[j] for j in range(len(targets))])

    return p_j.mean()


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


def precision_at_10(ranks, targets):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return precision_at_k(ranks=ranks, targets=targets, k=10)


def precision_at_100(ranks, targets):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return precision_at_k(ranks=ranks, targets=targets, k=100)


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


def recall_at_10(ranks, targets):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return recall_at_k(ranks=ranks, targets=targets, k=10)


def recall_at_100(ranks, targets):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return recall_at_k(ranks=ranks, targets=targets, k=100)


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


def dcg_at_k(ranks, targets, k):
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


def ndcg_at_k(ranks, targets, k):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    perfect_ranks = ranking(targets.reshape((-1, 1)))

    return dcg_at_k(ranks, targets, k)/dcg_at_k(perfect_ranks, targets, k)


def ndcg_at_10(ranks, targets):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return ndcg_at_k(ranks=ranks, targets=targets, k=10)


def ndcg_at_100(ranks, targets):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.

    Returns:
        torch.tensor, score of the batch.
    """

    return ndcg_at_k(ranks=ranks, targets=targets, k=100)


def main():
    pass


if __name__ == '__main__':
    main()