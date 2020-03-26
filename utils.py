from argparse import ArgumentParser


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task_name", required=True, help="Name of the modeling task (e.g. 'ContextFree').")
    ap.add_argument("-m", "--model_name", required=True, help="Name of the baseline (e.g. 'Random').")
    ap.add_argument("-p", "--pretrained_model", help="Name of the pretrained model (e.g. 'word2vec').")
    ap.add_argument("-e", "--experiment_name", help="Name of the experiment (e.g. 'test_0').")

    args = vars(ap.parse_args())

    return args
