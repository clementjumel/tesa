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


def format_context(ranking_or_inputs, context_format):
    """
    Return the context formated depending on context_format.

    Args:
        ranking_or_inputs: list of (inputs, targets) batches, or just an inputs batch.
        context_format: str, version of the context format to use.
    """

    if isinstance(ranking_or_inputs, list):  # ranking_or_inputs is a ranking
        inputs, _ = ranking_or_inputs[0]
    else:  # ranking_or_inputs is some inputs
        inputs = ranking_or_inputs

    assert isinstance(inputs, dict)

    if context_format == "v0":  # No separation token
        context_items = []

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.append(wiki_article)

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend([nyt_title + ':', nyt_context])

        return " ".join(context_items)

    elif context_format == "v1":  # [W] wiki1 [W] wiki2 [C] article1 [C] article2 [CLS]
        context_items = []

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.extend(["[W]", wiki_article])

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend(["[C]", nyt_title + ":", nyt_context])

        context_items.append("[CLS]")
        return " ".join(context_items)

    elif context_format == "v2":  # [W] wiki1 [W] wiki2 [T] title1 [C] context1 [T] title2 [C] context2 [CLS]
        context_items = []

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.extend(["[W]", wiki_article])

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend(["[T]", nyt_title, "[C]", nyt_context])

        context_items.append("[CLS]")
        return " ".join(context_items)

    elif context_format == "v3":  # [W] wiki1 [W] wiki2 [T] all titles [C] all contexts [CLS]
        context_items = []

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.extend(["[W]", wiki_article])

        context_items.append("[T]")
        context_items.extend(inputs['nyt_titles'])

        context_items.append("[C]")
        context_items.append(inputs['nyt_contexts'])

        context_items.append("[CLS]")
        return " ".join(context_items)

    else:
        raise Exception("Context format not implemented: %s." % context_format)


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
