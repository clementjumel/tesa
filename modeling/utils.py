from numpy import mean
import torch


# region Miscellaneous

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


def dict_remove_none(d):
    """
    Remove None from the input dictionary. If a list item is empty, replace it with [0.].

    Args:
        d: dict, input dictionary.
    """

    for key in d:
        items = [item for item in d[key] if item is not None]
        items = items or [0.]
        d[key] = items

# endregion


# region Latex methods

LATEX_CODE = ''

SCORE_TO_LATEX_NAME = {
    'average_precision': r'\shortstack{av.\\prec.}',
    'precision_at_10': r'\shortstack{prec.\\@10}',
    'precision_at_100': r'\shortstack{prec.\\@100}',
    'recall_at_10': r'\shortstack{recall\\@10}',
    'recall_at_100': r'\shortstack{recall\\@100}',
    'reciprocal_best_rank': r'\shortstack{recip.\\best\\rank}',
    'reciprocal_average_rank': r'\shortstack{recip.\\av.\\rank}',
    'ndcg_at_10': r'\shortstack{ndcg\\@10}',
    'ndcg_at_100': r'\shortstack{ndcg\\@10à}'
}

MODEL_TO_LATEX_NAME = {
    'RandomBaseline': 'random',
    'FrequencyBaseline':  'frequency',
    'SummariesCountBaseline':  r'\shortstack{summ.\\count}',
    'SummariesUniqueCountBaseline':  r'\shortstack{summ.\\un. count}',
    'SummariesOverlapBaseline':  r'\shortstack{summ.\\overlap}',
    'SummariesAverageEmbeddingBaseline':  r'\shortstack{summ. av.\\embed.}',
    'SummariesOverlapAverageEmbeddingBaseline':  r'\shortstack{summ. overlap\\av. embed.}',
    'ActivatedSummariesBaseline':  r'\shortstack{activated\\summ.}',
    'ContextCountBaseline':  r'\shortstack{cont.\\count}',
    'ContextUniqueCountBaseline':  r'\shortstack{cont. un.\\ count}',
    'ContextAverageEmbeddingBaseline':  r'\shortstack{cont. av.\\embed.}',
    'SummariesContextCountBaseline':  r'\shortstack{summ. cont.\\count}',
    'SummariesContextUniqueCountBaseline':  r'\shortstack{summ. cont.\\un. count}',
    'SummariesOverlapContextBaseline':  r'\shortstack{summ.\\ overlap cont.}',
    'SummariesContextAverageEmbeddingBaseline':  r'\shortstack{summ. cont.\\av. embed.}',
    'SummariesOverlapContextAverageEmbeddingBaseline':  r'\shortstack{summ.\\overlap cont.\\av. embed.}'
}


def init_latex_code(scores_names):
    """
    Initializes the global variable LATEX_CODE.

    Args:
        scores_names: iterable, names of the scores to use, the first one being monitored during training.
    """

    global LATEX_CODE

    LATEX_CODE = ''

    line = r'\begin{tabular}{r|'
    line += '|'.join(['c' for _ in range(len(scores_names))])
    line += '}'
    LATEX_CODE += line + '\n'

    names = [SCORE_TO_LATEX_NAME[name] if name in SCORE_TO_LATEX_NAME else name for name in scores_names]
    line = 'Method & '
    line += ' & '.join(names)
    line += r' \\ \hline'
    LATEX_CODE += line + '\n'


def update_latex_code(model, valid=True, test=False):
    """
    Updates the global variable LATEX_CODE with the scores of display_metrics of the model.

    Args:
        model: models.Model, model to evaluate.
        valid: bool, whether or not to display the validation scores.
        test: bool, whether or not to display the test scores.
    """

    global LATEX_CODE

    assert not (valid and test) and (valid or test)

    name = type(model).__name__
    name = MODEL_TO_LATEX_NAME[name] if name in MODEL_TO_LATEX_NAME else name

    latex_code_valid, latex_code_test = model.display_metrics(valid=valid, test=test, silent=True)
    latex_code = latex_code_valid or latex_code_test

    line = name + latex_code + r'\\'
    LATEX_CODE += line + '\n'


def display_latex_code():
    """ Display the global variable LATEX_CODE. """

    print(LATEX_CODE)

# endregion


def main():
    pass


if __name__ == '__main__':
    main()
