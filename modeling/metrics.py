from modeling.utils import ranking

import torch


def average_precision(ranks, targets, relevance_level):
    """
    Compute the Average Precision the ranks and targets line Tensors. If there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    mask = targets >= relevance_level
    ranks = ranks[mask]
    n = len(ranks)

    if n == 0:
        return None

    else:
        ranks1 = ranks.expand((n, n))
        ranks2 = ranks.reshape((-1, 1)).expand((n, n))

        return torch.div(torch.ge(ranks2, ranks1).sum(dim=1).type(torch.float), ranks.type(torch.float)).mean()


def precision_at_k(ranks, targets, relevance_level, k):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    mask = targets < relevance_level
    targets[mask] = 0

    if targets.sum() == 0:
        return None

    else:
        mask = ranks <= k
        targets = targets[mask]

        mask = targets > 0
        targets[mask] = 1

        return torch.div(targets.sum().type(torch.float), k)


def precision_at_10(ranks, targets, relevance_level):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return precision_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=10)


def precision_at_100(ranks, targets, relevance_level):
    """
    Returns the precision at k, that is the fraction of answers retrieved by the first k ranks that are relevant. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return precision_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=100)


def recall_at_k(ranks, targets, relevance_level, k):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    mask = targets < relevance_level
    targets[mask] = 0
    targets[~mask] = 1
    n = targets.sum()

    if n == 0:
        return None

    else:
        mask = ranks <= k
        targets = targets[mask]

        return torch.div(targets.sum().type(torch.float), n)


def recall_at_10(ranks, targets, relevance_level):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return recall_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=10)


def recall_at_100(ranks, targets, relevance_level):
    """
    Returns the recall at k, that is the fraction of relevant answers that are retrieved by the first k ranks. If
    there is not relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return recall_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=100)


def reciprocal_best_rank(ranks, targets, relevance_level):
    """
    Returns the reciprocal best rank of the relevant answers. If there is no relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    mask = targets < relevance_level
    targets[mask] = 0

    if targets.sum() == 0:
        return None

    else:
        mask = targets > 0
        ranks = ranks[mask]

        return ranks.min().type(torch.float).reciprocal()


def reciprocal_average_rank(ranks, targets, relevance_level):
    """
    Returns the reciprocal average rank of the relevant answers. If there is no relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    mask = targets < relevance_level
    targets[mask] = 0

    if targets.sum() == 0:
        return None

    else:
        mask = targets > 0
        ranks = ranks[mask]

        return ranks.type(torch.float).mean().reciprocal()


def ndcg_at_k(ranks, targets, relevance_level, k):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.
        k: int, number of ranks to take into account.

    Returns:
        torch.tensor, score of the batch.
    """

    del relevance_level

    perfect_ranks = ranking(targets.reshape((-1, 1)))

    mask1, mask2 = ranks <= k, perfect_ranks <= k
    ranks, perfect_ranks = ranks[mask1], perfect_ranks[mask2]
    targets1, targets2 = targets[mask1], targets[mask2]

    g1 = torch.pow(2, targets1) - 1
    g2 = torch.pow(2, targets2) - 1

    d1 = torch.log2(ranks.type(torch.float) + 1).reciprocal()
    d2 = torch.log2(perfect_ranks.type(torch.float) + 1).reciprocal()

    return torch.div(torch.dot(g1.type(torch.float), d1), torch.dot(g2.type(torch.float), d2))


def ndcg_at_10(ranks, targets, relevance_level):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return ndcg_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=10)


def ndcg_at_100(ranks, targets, relevance_level):
    """
    Compute the Normalized Discounted Cumulative Gain at k for the line Tensors ranks and targets. If there is not
    relevant answer, returns None.

    Args:
        ranks: torch.Tensor, ranks predicted for the batch.
        targets: torch.Tensor, true labels for the batch.
        relevance_level: int, minimum label to consider a choice as relevant.

    Returns:
        torch.tensor, score of the batch.
    """

    return ndcg_at_k(ranks=ranks, targets=targets, relevance_level=relevance_level, k=100)
