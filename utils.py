from argparse import ArgumentParser


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task_name", required=True, help="name of the modeling task")
    ap.add_argument("-m", "--model_name", required=True, help="name of the baseline")

    args = vars(ap.parse_args())

    return args
