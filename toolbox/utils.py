from argparse import ArgumentParser
from gensim.models import KeyedVectors
from torch.hub import load as torch_hub_load


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task_name", required=True, help="Name of the modeling task (e.g. 'ContextFree').")
    ap.add_argument("-m", "--model_name", required=True, help="Name of the baseline (e.g. 'Random').")
    ap.add_argument("-e", "--experiment_name", help="Name of the experiment (e.g. 'test_0').")
    ap.add_argument("-p", "--pretrained_model_name", help="Name of the pretrained model (e.g. 'word2vec').")

    args = vars(ap.parse_args())

    return args


def get_pretrained_model(args, silent, folder_path, root=''):
    """
    Returns the pretrained model and its dimension, if relevant, given the arguments of the script.

    Args:
        args: dict, arguments passed to the script.
        silent: bool, silence option.
        folder_path: str, path to the pretrained_models folder.
        root: str, path to the root of the project.

    Returns:
        pretrained_model: unknown type, various pretrained model or embedding.
        pretrained_model_dim: int, dimension of the pretrained model.
    """

    if "pretrained_model_name" not in args:
        pretrained_model, pretrained_model_dim = None, None

    else:
        pretrained_model_name = args['pretrained_model_name']
        pretrained_model_names = ["word2vec", "bart_mnli"]

        if pretrained_model_name == pretrained_model_names[0]:
            fname = root + folder_path + "GoogleNews-vectors-negative300.bin"

            pretrained_model = KeyedVectors.load_word2vec_format(fname=fname, binary=True)
            pretrained_model_dim = 300

            print("Word2Vec embedding loaded.") if not silent else None

        elif pretrained_model_name == pretrained_model_names[1]:
            pretrained_model = torch_hub_load("pytorch/fairseq", "bart.large.mnli")
            pretrained_model_dim = None

            print("Pretrained BART mnli loaded.") if not silent else None

        else:
            raise Exception("Wrong pretrained model name: %s (valid names are %s)." % (pretrained_model_name,
                                                                                       str(pretrained_model_names)))

    return pretrained_model, pretrained_model_dim
