"""
Script to run Models on a ModelingTask.

Usages:
    python run_models.py -t context_free_same_type -m frequency
    python run_models.py -t context_free_same_type -rs 32 -bs 16 -m summaries_average_embedding --word2vec
    python run_models.py -t context_free_same_type -rs 32 -bs 16 --short -m classifier_bart --bart \
        -bp results/models/RTE-bin_vanilla -cf checkpoints/checkpoint1.pt
"""

import modeling.models as models
from toolbox.utils import to_class_name, add_task_argument, load_task, get_trained_model, play
from toolbox.parameters import SCORES_NAMES, MODELS_RANDOM_SEED
from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_RESULTS_PATH, TENSORBOARD_LOGS_PATH

from argparse import ArgumentParser


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()
    add_task_argument(ap)

    ap.add_argument("-m", "--model", required=True, type=str, help="Name of the model.")
    ap.add_argument("-e", "--experiment", default=None, type=str, help="Name of the experiment.")
    ap.add_argument("-w", "--word2vec", action="store_true", help="Load Word2Vec embedding.")
    ap.add_argument("-b", "--bart", action="store_true", help="Load a BART model.")
    ap.add_argument("-bp", "--bin_path", default=None, type=str, help="Path to bin folder (preprocessed data).")
    ap.add_argument("-cf", "--checkpoint_file", default=None, type=str, help="Name of BART's checkpoint file.")

    return vars(ap.parse_args())


def main():
    """ Makes a model run on a task. """

    args = parse_arguments()

    task = load_task(args=args, folder_path=MODELING_TASK_RESULTS_PATH)

    trained_model = get_trained_model(args=args, folder_path=PRETRAINED_MODELS_PATH)

    model = getattr(models, to_class_name(args['model']))(scores_names=SCORES_NAMES,
                                                          relevance_level=task.relevance_level,
                                                          trained_model=trained_model,
                                                          tensorboard_logs_path=TENSORBOARD_LOGS_PATH,
                                                          experiment_name=args['experiment'],
                                                          random_seed=MODELS_RANDOM_SEED)

    play(task=task, model=model)


if __name__ == '__main__':
    main()
