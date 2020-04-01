from numpy import mean
import torch


def ranking(outputs):
    """
    Ranks according to the outputs (1 for highest grade). Deal with draws by assigning the best rank to the first
    output encountered.

    Args:
        outputs: torch.Tensor, (batch_size, 1 ou 2) tensors outputs.

    Returns:
        torch.Tensor, ranks corresponding to the grades in a line Tensor.
    """

    grades = outputs[:, -1]

    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    ranks = torch.zeros(n, dtype=torch.long)

    ranks[sorter] = torch.arange(1, n + 1)

    return ranks


def dict_mean(d):
    """
    Returns a dictionary with the mean of the lists of the dictionary d.

    Args:
        d: dict, input dictionary.

    Returns:
        dict, mean dictionary.
    """

    return {key: mean(item) for key, item in d.items()}


def dict_append(d1, d2):
    """
    Append the elements of d2 to the elements of d1.

    Args:
        d1: dict, main dictionary.
        d2: dict, secondary dictionary, appended to the main dictionary.
    """

    for key, item in d2.items():
        d1[key].append(item)


def list_remove_none(l):
    """
    Removes None from the list l.

    Args:
        l: list, initial list to process.

    Returns:
        list, final list, without None.
    """

    return [item for item in l if item is not None]


def dict_remove_none(d):
    """
    Removes None from the input dictionary.

    Args:
        d: dict, input dictionary.
    """

    for key in d:
        items = [item for item in d[key] if item is not None]
        d[key] = items
