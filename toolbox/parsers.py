from toolbox.parameters import *
from toolbox.paths import *

from argparse import ArgumentParser


def standard_parser():
    """ Create an argument parser, add the parsing arguments relative to the options and return it. """

    ap = ArgumentParser()

    ap.add_argument("--random_seed",
                    type=int, default=RANDOM_SEED,
                    help="Random seed for numpy and torch.")
    ap.add_argument("--root",
                    type=str, default="",
                    help="Path to the root of the project.")
    ap.add_argument("--no_save",
                    action='store_true',
                    help="No save option.")
    ap.add_argument("--silent",
                    action='store_true',
                    help="Silence option.")

    return ap


def add_task_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the modeling task.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the arguments.
    """

    ap.add_argument("-t", "--task",
                    type=str, required=True,
                    help="Name of the modeling task version.")
    ap.add_argument("-vp", "--valid_proportion",
                    type=float, default=VALID_PROPORTION,
                    help="Proportion of the validation set.")
    ap.add_argument("-tp", "--test_proportion",
                    type=float, default=TEST_PROPORTION,
                    help="Proportion of the test set.")
    ap.add_argument("-rs", "--ranking_size",
                    type=int, default=RANKING_SIZE,
                    help="Size of the ranking tasks.")
    ap.add_argument("-bs", "--batch_size",
                    type=int, default=BATCH_SIZE,
                    help="Size of the batches of the task.")
    ap.add_argument("-cf", "--context_format",
                    type=str, default=CONTEXT_FORMAT,
                    help="Version of the context format.")
    ap.add_argument("-tf", "--targets_format",
                    type=str, default=TARGETS_FORMAT,
                    help="Version of the targets format.")
    ap.add_argument("--task_path",
                    type=str, default=MODELING_TASK_RESULTS_PATH,
                    help="Path to the task folder.")
    ap.add_argument("--cross_validation",
                    action='store_true',
                    help="Cross validation option.")
    ap.add_argument("--generation",
                    action='store_true',
                    help="Generation finetuning option.")
    ap.add_argument("--classification",
                    action='store_true',
                    help="Classification finetuning option.")


def add_model_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the model.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the arguments.
    """

    ap.add_argument("-m", "--model",
                    type=str, required=True,
                    help="Name of the model.")
    ap.add_argument("-sn", "--score_names",
                    type=list, default=SCORES_NAMES,
                    help="Names of the scores to use.")
    ap.add_argument("-e", "--experiment",
                    type=str, default=None,
                    help="Name of the experiment.")
    ap.add_argument("--tensorboard_logs_path",
                    type=str, default=TENSORBOARD_LOGS_PATH,
                    help="Path of the tensorboard logs folder.")
    ap.add_argument("--pretrained_path",
                    type=str, default=PRETRAINED_MODELS_PATH,
                    help="Path to the pretrained model folder.")
    ap.add_argument("--checkpoint_file",
                    type=str, default=None,
                    help="Name of BART's checkpoint file.")
    ap.add_argument("--model_random_seed",
                    type=int, default=RANDOM_SEED,
                    help="Random seed of the model.")
    ap.add_argument("--show_rankings",
                    type=int, default=SHOW_RANKINGS,
                    help="Number of rankings to show.")
    ap.add_argument("--show_choices",
                    type=int, default=SHOW_CHOICES,
                    help="Number of choices to show per ranking.")
    ap.add_argument("--word2vec",
                    action="store_true",
                    help="Load Word2Vec embedding.")
    ap.add_argument("--bart",
                    action="store_true",
                    help="Load a BART model.")
    ap.add_argument("--show",
                    action="store_true",
                    help="Option to show some results.")
