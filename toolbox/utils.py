from toolbox.parameters import *
from toolbox.paths import *

from pickle import load
from argparse import ArgumentParser
import torch


def to_class_name(name):
    """ For a name of the format 'abc_efg', returns the corresponding Class name, of the format 'AbcEfg'. """

    return "".join([word.capitalize() for word in name.split("_")])


def standard_parser():
    """ Create an argument parser, add the parsing arguments relative to the options and return it. """

    ap = ArgumentParser()
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

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
    ap.add_argument("--pretrained_path",
                    type=str, default=PRETRAINED_MODELS_PATH,
                    help="Path to the pretrained model folder.")
    ap.add_argument("--checkpoint_file",
                    type=str, default=None,
                    help="Name of BART's checkpoint file.")
    ap.add_argument("--word2vec",
                    action="store_true",
                    help="Load Word2Vec embedding.")
    ap.add_argument("--bart",
                    action="store_true",
                    help="Load a BART model.")
    ap.add_argument("--show",
                    action="store_true",
                    help="Option to show some results.")


def load_task(args):
    """
    Load a Task using pickle, depending on the arguments passed in args.

    Args:
        args: argparse.ArgumentParser, arguments passed to the script.

    Returns:
        database_creation.modeling_task.Task, loaded object.
    """

    task_name = args.task
    valid_proportion = args.valid_proportion
    test_proportion = args.test_proportion
    ranking_size = args.ranking_size
    batch_size = args.batch_size
    context_format = args.context_format
    targets_format = args.targets_format
    folder_path = args.task_path
    cross_validation = args.cross_validation

    train_proportion = ("%.2f" % (1 - valid_proportion - test_proportion)).split(".")[1]
    valid_proportion = ("%.2f" % valid_proportion).split(".")[1]
    test_proportion = ("%.2f" % test_proportion).split(".")[1]

    suffix = "_" + "-".join([train_proportion, valid_proportion, test_proportion])
    suffix += "_rs" + str(ranking_size) if ranking_size is not None else ""
    suffix += "_bs" + str(batch_size)
    suffix += "_cf-" + context_format if context_format is not None else ""
    suffix += "_tf-" + targets_format if targets_format is not None else ""
    suffix += "_cv" if cross_validation else ""

    task_name = "".join(task_name.split("_"))
    file_name = folder_path + task_name + suffix + '.pkl'

    with open(file_name, 'rb') as file:
        task = load(file)

    print("Task loaded from %s.\n" % file_name)

    return task


def get_trained_model(args):
    """ Returns the trained model (word2vec embedding or bart). """

    use_word2vec = args.word2vec
    use_bart = args.bart
    folder_path = args.pretrained_path
    checkpoint_file = args.checkpoint_file

    if not use_word2vec and not use_bart:
        return None

    elif use_word2vec and not use_bart:
        return get_word2vec(folder_path)

    elif not use_word2vec and use_bart:
        return get_bart(folder_path=folder_path, checkpoint_file=checkpoint_file)

    else:
        raise Exception("Must chose between word2vec and BART.")


def get_word2vec(folder_path):
    """
    Returns the word2vec embedding.

    Args:
        folder_path: str, path to the pretrained models.
    """

    from gensim.models import KeyedVectors

    fname = folder_path + "GoogleNews-vectors-negative300.bin"
    word2vec = KeyedVectors.load_word2vec_format(fname=fname, binary=True)

    print("Word2Vec embedding loaded.\n")

    return word2vec


def get_bart(folder_path, checkpoint_file):
    """
    Returns a pretrained BART model.

    Args:
        folder_path: str, path to BART's model, containing the checkpoint.
        checkpoint_file: str, name of BART's checkpoint file (starting from BART's folder).
    """

    from fairseq.models.bart import BARTModel

    bart = BARTModel.from_pretrained(model_name_or_path=folder_path,
                                     checkpoint_file=checkpoint_file)

    if torch.cuda.is_available():
        bart.cuda()
        print("Using BART on GPU...")

    bart.eval()

    print("BART loaded (in evaluation mode).\n")

    return bart
