"""
Script to run baselines on a task.

Usages:
    python run_baselines.py -t context_free_same_type -m random
    python run_baselines.py -t context_free_same_type -m frequency
    python run_baselines.py -t context_free_same_type -m summaries_average_embedding --word2vec
    python run_baselines.py -t context_free_same_type -rs 32 -bs 16 -m random
    python run_baselines.py -t context_free_same_type -rs 32 -bs 16 -m frequency
    python run_baselines.py -t context_free_same_type -rs 32 -bs 16 -m summaries_average_embedding --word2vec
"""

import modeling.models as models
from toolbox.utils import to_class_name, load_task, add_task_argument

from argparse import ArgumentParser
from gensim.models import KeyedVectors

from toolbox.parameters import SCORES_NAMES, BASELINES_RANDOM_SEED

from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_RESULTS_PATH, TENSORBOARD_LOGS_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    add_task_argument(ap)
    ap.add_argument("-m", "--model", required=True, type=str, help="Name of the model.")
    ap.add_argument("-w", "--word2vec", action="store_true", help="Load Word2Vec embedding.")
    ap.add_argument("-e", "--experiment", default=None, type=str, help="Name of the experiment.")

    args = vars(ap.parse_args())

    return args


def get_word2vec(folder_path):
    """
    Returns the pretrained word2vec embedding and its dimension.

    Args:
        folder_path: str, path to the pretrained_models folder.

    Returns:
        pretrained_model: gensim.KeyedVector, pretrained embedding.
        pretrained_model_dim: int, dimension of the embedding.
    """

    fname = folder_path + "GoogleNews-vectors-negative300.bin"

    pretrained_model = KeyedVectors.load_word2vec_format(fname=fname, binary=True)
    pretrained_model_dim = 300

    print("Word2Vec embedding loaded.\n")

    return pretrained_model, pretrained_model_dim


def play_baseline(task, model):
    """
    Performs the preview of the data and the evaluation of a baseline.

    Args:
        task: modeling_task.ModelingTask, task to evaluate the baseline upon.
        model: models.Baseline, baseline to evaluate.
    """

    model.preview(task.train_loader)
    model.preview(task.valid_loader)

    print("Evaluation on the train_loader...")
    model.valid(task.train_loader)

    print("Evaluation on the valid_loader...")
    model.valid(task.valid_loader)


def main():
    """ Makes a baseline run on a task. """

    args = parse_arguments()

    task_name = args['task']
    valid_proportion = args['valid_proportion']
    test_proportion = args['test_proportion']
    ranking_size = args['ranking_size']
    batch_size = args['batch_size']
    cross_validation = args['cross_validation']
    short = args['short']
    model_name = to_class_name(args['model'])
    word2vec = args['word2vec']
    experiment_name = args['experiment']

    task = load_task(task_name=task_name,
                     valid_proportion=valid_proportion,
                     test_proportion=test_proportion,
                     ranking_size=ranking_size,
                     batch_size=batch_size,
                     cross_validation=cross_validation,
                     short=short,
                     folder_path=MODELING_TASK_RESULTS_PATH)

    pretrained_model, pretrained_model_dim = get_word2vec(PRETRAINED_MODELS_PATH) if word2vec else (None, None)

    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_dim=pretrained_model_dim,
                                        tensorboard_logs_path=TENSORBOARD_LOGS_PATH,
                                        experiment_name=experiment_name,
                                        random_seed=BASELINES_RANDOM_SEED)

    play_baseline(task=task,
                  model=model)


if __name__ == '__main__':
    main()
