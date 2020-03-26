from argparse import ArgumentParser


def parse_arguments():
    """
    Use arparse to parse the input arguments and retrieve the task_name and the model_name.

    Returns:
        task_name: str, name of the task to use.
        model_name: str, name of the model to use.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task_name", required=True, help="name of the modeling task")
    ap.add_argument("-m", "--model_name", required=True, help="name of the baseline")

    args = vars(ap.parse_args())

    return args['task_name'], args['model_name']
