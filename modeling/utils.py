from numpy import mean, std
import torch


def get_ranks(outputs):
    """
    Returns the ranks according to the outputs (1 for highest grade). Deal with draws by assigning the best rank to the
    first output encountered.

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


def inputs_to_context(inputs):
    """
    Returns the context of a ranking task as a string with the wikipedia information followed by the article's context.

    Args:
        inputs: dict, inputs of a batch.
    """

    context_elements = []

    for wikipedia in inputs['wikipedia']:
        if wikipedia != "No information found.":
            context_elements.append(wikipedia)

    context_elements.append(inputs['context'])

    return " ".join(context_elements)


# region Built-in objects method
def list_remove_none(l):
    """
    Removes None from the list l.

    Args:
        l: list, initial list to process.

    Returns:
        list, final list, without None.
    """

    return [item for item in l if item is not None]


def dict_append(d1, d2):
    """
    Append the elements of d2 to the elements of d1.

    Args:
        d1: dict, main dictionary.
        d2: dict, secondary dictionary, appended to the main dictionary.
    """

    for key, item in d2.items():
        d1[key].append(item)


def dict_mean(d):
    """
    Returns a dictionary with the mean of the lists of the dictionary d.

    Args:
        d: dict, input dictionary.

    Returns:
        dict, mean dictionary.
    """

    return {key: mean(item) for key, item in d.items()}


def dict_std(d):
    """
    Returns a dictionary with the standard deviation of the lists of the dictionary d.

    Args:
        d: dict, input dictionary.

    Returns:
        dict, standard deviation dictionary.
    """

    return {key: std(item) for key, item in d.items()}
# endregion
