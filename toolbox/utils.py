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
        ap: argparse.ArgumentParser, argument parser to update with the modeling task relative arguments.
    """

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-vp", "--valid_proportion", default=0.25, type=float, help="Proportion of the validation set.")
    ap.add_argument("-tp", "--test_proportion", default=0.25, type=float, help="Proportion of the test set.")
    ap.add_argument("-rs", "--ranking_size", default=None, type=int, help="Size of the ranking tasks.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int, help="Size of the batches of the task.")
    ap.add_argument("--task_path", default=None, type=str, help="Path to the task folder.")
    ap.add_argument("--cross_validation", action='store_true', help="Cross validation option.")
    ap.add_argument("--short", action='store_true', help="Shorten modeling task option.")


def load_task(args, folder_path):
    """
    Load a Task using pickle from the folder_path, depending on the arguments passed in args.

    Args:
        args: dict, arguments passed to the script.
        folder_path: str, path of the folder to load from.

    Returns:
        database_creation.modeling_task.Task, loaded object.
    """

    task_name = args['task']
    valid_proportion = args['valid_proportion']
    test_proportion = args['test_proportion']
    ranking_size = args['ranking_size']
    batch_size = args['batch_size']
    folder_path = args['task_path'] or folder_path
    cross_validation = args['cross_validation']
    short = args['short']

    train_proportion = ("%.2f" % (1 - valid_proportion - test_proportion)).split(".")[1]
    valid_proportion = ("%.2f" % valid_proportion).split(".")[1]
    test_proportion = ("%.2f" % test_proportion).split(".")[1]
    suffix = "_" + "-".join([train_proportion, valid_proportion, test_proportion])

    suffix += "_rs" + str(ranking_size) if ranking_size is not None else ""
    suffix += "_bs" + str(batch_size)
    suffix += "_cv" if cross_validation else ""
    suffix += "_short" if short else ""

    task_name = "".join(task_name.split("_"))

    file_name = folder_path + task_name + suffix + '.pkl'

    with open(file_name, 'rb') as file:
        task = load(file)

    print("Task loaded from %s.\n" % file_name)

    return task


def get_trained_model(args, folder_path):
    """
    Returns the trained model (word2vec embedding or bart).

    Args:
        args: dict, arguments passed to the script.
        folder_path: str, path of the folder to load from (if not in the arguments).
    """

    use_word2vec = args['word2vec']
    use_bart = args['bart']
    folder_path = args['trained_path'] or folder_path
    checkpoint_file = args['checkpoint_file']

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
