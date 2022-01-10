"""
Metrics to evaluate the performance on the ranking task. The methods take two torch.Tensor as arguments (the ranks
predicted and the targets), and return the corresponding score in a torch.Tensor.
"""

from modeling.utils import get_ranks

import torch


def average_precision(ranks, targets):
    mask = targets > 0
    ranks = ranks[mask]
    n = len(ranks)

    if n == 0:
        return None

    else:
        ranks1 = ranks.expand((n, n))
        ranks2 = ranks.reshape((-1, 1)).expand((n, n))

        return torch.div(torch.ge(ranks2, ranks1).sum(dim=1).type(torch.float), ranks.type(torch.float)).mean()


def precision_at_k(ranks, targets, k):
    if targets.sum() == 0:
        return None

    else:
        mask = ranks <= k
        targets = targets[mask]

        mask = targets > 0
        targets[mask] = 1

        return torch.div(targets.sum().type(torch.float), k)


def precision_at_10(ranks, targets):
    return precision_at_k(ranks=ranks, targets=targets, k=10)


def recall_at_k(ranks, targets, k):
    n = targets.sum()
    if n == 0:
        return None

    else:
        mask = ranks <= k
        targets = targets[mask]

        return torch.div(targets.sum().type(torch.float), n)


def recall_at_10(ranks, targets):
    return recall_at_k(ranks=ranks, targets=targets, k=10)


def reciprocal_best_rank(ranks, targets):
    if targets.sum() == 0:
        return None

    else:
        mask = targets > 0
        ranks = ranks[mask]

        return ranks.min().type(torch.float).reciprocal()


def reciprocal_average_rank(ranks, targets):
    if targets.sum() == 0:
        return None

    else:
        mask = targets > 0
        ranks = ranks[mask]

        return ranks.type(torch.float).mean().reciprocal()


def ndcg_at_k(ranks, targets, k):
    perfect_ranks = get_ranks(targets.reshape((-1, 1)))

    mask1, mask2 = ranks <= k, perfect_ranks <= k
    ranks, perfect_ranks = ranks[mask1], perfect_ranks[mask2]
    targets1, targets2 = targets[mask1], targets[mask2]

    g1 = torch.pow(2, targets1) - 1
    g2 = torch.pow(2, targets2) - 1

    d1 = torch.log2(ranks.type(torch.float) + 1).reciprocal()
    d2 = torch.log2(perfect_ranks.type(torch.float) + 1).reciprocal()

    return torch.div(torch.dot(g1.type(torch.float), d1), torch.dot(g2.type(torch.float), d2))


def ndcg_at_10(ranks, targets):
    return ndcg_at_k(ranks=ranks, targets=targets, k=10)


if __name__ == '__main__':
    from sklearn.metrics import average_precision_score
    from numpy import zeros
    from numpy.random import rand, choice

    epsilon = 0.0000001

    for _ in range(100000):
        s = rand(24)
        r = get_ranks(torch.Tensor(s).reshape((-1, 1)))
        t = zeros(24)
        for i in choice(range(24), 3):
            t[i] = 1

        a1 = average_precision(r, torch.Tensor(t).type(torch.float)).item()
        a2 = average_precision_score(t, s)

        if abs(a1 - a2) > epsilon:
            print(a1, a2)
