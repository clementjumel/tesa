from pickle import load
from gensim.models import KeyedVectors
from fairseq.models.bart import BARTModel
import torch


def to_class_name(name):
    """
    For a name of the format 'abc_efg', returns the corresponding Class name, of the format 'AbcEfg'.

    Args:
        name: str, name of the class in format 'abc_efg'.

    Returns:
        str, name of the class in format 'AbcEfg'.
    """

    return "".join([word.capitalize() for word in name.split("_")])


def load_task(task_name, batch_size, cross_validation, short, folder_path):
    """
    Load a Task using pickle from [folder_path][task_name].pkl

    Args:
        task_name: str, name of the Task to load (eg 'context_free').
        batch_size: int, size of the batches of the modeling task.
        cross_validation: bool, cross-validation option.
        short: bool, whether to load the shorten task or not.
        folder_path: str, path of the folder to load from.

    Returns:
        database_creation.modeling_task.Task, loaded object.
    """

    suffix = "_bs" + str(batch_size)
    suffix += "_cv" if cross_validation else ""
    suffix += "_short" if short else ""

    task_name = "".join(task_name.split("_"))

    file_name = folder_path + task_name + suffix + '.pkl'

    with open(file_name, 'rb') as file:
        task = load(file)

    print("Task loaded from %s.\n" % file_name)

    return task


def get_pretrained_model(pretrained_model_name, folder_path):
    """
    Returns the pretrained model and its dimension, if relevant.

    Args:
        pretrained_model_name: str, name of the model.
        folder_path: str, path to the pretrained_models folder.

    Returns:
        pretrained_model: unknown type, various pretrained model or embedding.
        pretrained_model_dim: int, dimension of the pretrained model.
    """

    if pretrained_model_name is None:
        pretrained_model, pretrained_model_dim = None, None

    else:
        pretrained_model_names = ["word2vec", "bart_mnli"]

        if pretrained_model_name == pretrained_model_names[0]:
            fname = folder_path + "GoogleNews-vectors-negative300.bin"

            pretrained_model = KeyedVectors.load_word2vec_format(fname=fname, binary=True)
            pretrained_model_dim = 300

            print("Word2Vec embedding loaded.\n")

        elif pretrained_model_name == pretrained_model_names[1]:
            file_name = folder_path + "bart.large.mnli"

            pretrained_model = BARTModel.from_pretrained(file_name, checkpoint_file='model.pt')
            pretrained_model.eval()

            if torch.cuda.is_available():
                pretrained_model.cuda()

            pretrained_model_dim = None

            print("Pretrained BART.mnli loaded.\n")

        else:
            raise Exception("Wrong pretrained model name: %s (valid names are %s)." % (pretrained_model_name,
                                                                                       str(pretrained_model_names)))

    return pretrained_model, pretrained_model_dim
