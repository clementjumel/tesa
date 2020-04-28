"""
Script to run a Model on a task.

Usages:
    python run_model.py -t context_free_same_type -m random
    python run_model.py -t context_free_same_type -m frequency --show
    python run_model.py -t context_free_same_type -m summaries_average_embedding --word2vec
"""

from toolbox.parsers import standard_parser, add_task_arguments, add_model_arguments
from toolbox.utils import load_task, get_trained_model, to_class_name
import modeling.models as models


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_task_arguments(ap)
    add_model_arguments(ap)

    return ap.parse_args()


def main():
    """ Makes a model run on a task. """

    args = parse_arguments()

    task = load_task(args)
    trained_model = get_trained_model(args)

    model = getattr(models, to_class_name(args.model))(trained_model=trained_model,
                                                       relevance_level=task.relevance_level,
                                                       args=args)

    model.play(task)


if __name__ == '__main__':
    main()
