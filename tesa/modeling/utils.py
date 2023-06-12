import torch
from numpy import mean, std


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


def format_context(ranking_or_inputs, context_format, context_max_size):
    """
    Return the context formated depending on context_format.

    Args:
        ranking_or_inputs: list of (inputs, targets) batches, or just an inputs batch.
        context_format: str, version of the context format to use.
        context_max_size: int, maximum number of tokens in the context.
    """

    if isinstance(ranking_or_inputs, list):  # ranking_or_inputs is a ranking
        inputs, _ = ranking_or_inputs[0]
    else:  # ranking_or_inputs is some inputs
        inputs = ranking_or_inputs

    assert isinstance(inputs, dict)

    context_items = []

    if context_format == "v0":  # (no separation token) wikis articles entities
        for wiki_article in inputs["wiki_articles"]:
            if wiki_article:
                context_items.append(wiki_article)

        for nyt_title, nyt_context in zip(inputs["nyt_titles"], inputs["nyt_contexts"]):
            context_items.extend([nyt_title + ":", nyt_context])

        context_items.append(", ".join(inputs["entities"]))

    elif context_format in ["v1", "v2", "v3", "v4"]:
        # wiki_sep wiki1 wiki_sep wiki2 article_sep article1 article_sep article2 entity_sep entity1 entity_sep ...
        if context_format == "v1":
            wiki_sep = "§"
            article_sep = "£"
            entity_sep = "µ"

        elif context_format == "v2":
            wiki_sep = "Information:"
            article_sep = "Article:"
            entity_sep = "Entity:"

        elif context_format == "v3":
            wiki_sep = "<w>"
            article_sep = "<a>"
            entity_sep = "<e>"

        else:
            wiki_sep = "W"
            article_sep = "A"
            entity_sep = "E"

        for wiki_article in inputs["wiki_articles"]:
            if wiki_article:
                context_items.extend([wiki_sep, wiki_article])

        for nyt_title, nyt_context in zip(inputs["nyt_titles"], inputs["nyt_contexts"]):
            context_items.extend([article_sep, nyt_title + ".", nyt_context])

        for entity in inputs["entities"]:
            context_items.extend([entity_sep, entity])

    else:  # ablation studies based on v0
        if context_format == "va":  # no wikis
            for nyt_title, nyt_context in zip(inputs["nyt_titles"], inputs["nyt_contexts"]):
                context_items.extend([nyt_title + ":", nyt_context])

            context_items.append(", ".join(inputs["entities"]))

        elif context_format == "vb":  # no article
            for wiki_article in inputs["wiki_articles"]:
                if wiki_article:
                    context_items.append(wiki_article)

            context_items.append(", ".join(inputs["entities"]))

        elif context_format == "vc":  # no wikis and no article
            context_items.append(", ".join(inputs["entities"]))

        else:
            raise NotImplementedError("Context format not implemented: %s." % context_format)

    context = " ".join(context_items)

    context_words = context.split()
    l1 = len(context_words)
    if l1 > context_max_size:
        context_words = context_words[-context_max_size:]
        l2 = len(context_words)
        context = " ".join(context_words)

        print("Removing %i tokens from the context." % (l1 - l2))

    return context


def format_targets(ranking, targets_format):
    """
    Returns the generation targets as a list of str, depending on targets_format.

    Args:
        ranking: list of (inputs, targets) batches
        targets_format: str, version of the targets format to use.
    """

    valid_choices = []
    for inputs, targets in ranking:
        for choice, target in zip(inputs["choices"], targets):
            if target:
                valid_choices.append(choice)

    if targets_format == "v0":  # one target per valid choice
        return valid_choices

    elif (
        targets_format == "v1"
    ):  # all valid choices in one target, separated with separation tokens.
        return ["∂ " + " ∂ ".join(valid_choices)]

    else:
        raise Exception("Targets format not implemented: %s." % targets_format)


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
