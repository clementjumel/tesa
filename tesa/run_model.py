"""
Script to run a Model on a task.

Usages:
    python run_model.py -m random
    python run_model.py -m frequency --show
    python run_model.py -m summaries_average_embedding --word2vec
"""

import tesa.modeling.models as models
from tesa.toolbox.parsers import add_model_arguments, add_task_arguments, standard_parser
from tesa.toolbox.utils import get_pretrained_model, load_task, to_class_name


def parse_arguments():
    """Use arparse to parse the input arguments and return it as a argparse.ArgumentParser."""

    ap = standard_parser()
    add_task_arguments(ap)
    add_model_arguments(ap)

    return ap.parse_args()


def main():
    """Makes a model run on a task."""

    args = parse_arguments()

    task = load_task(args)
    pretrained_model = get_pretrained_model(args)

    model = getattr(models, to_class_name(args.model))(args=args, pretrained_model=pretrained_model)

    model.play(task=task, args=args)


if __name__ == "__main__":
    main()
