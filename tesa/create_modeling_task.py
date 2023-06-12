"""
Script to create the modeling task to be solved by models.

Usages:
    tests:
        python create_modeling_task.py -t context-free-same-type --classification --generation --no_save
    regular usages:
        python create_modeling_task.py -t context-free-same-type --classification --generation
        python create_modeling_task.py -t context-dependent-same-type --generation --generation
"""

import tesa.modeling.modeling_task as modeling_task
from tesa.toolbox.parsers import add_annotations_arguments, add_task_arguments, standard_parser
from tesa.toolbox.utils import to_class_name


def parse_arguments():
    """Use arparse to parse the input arguments and return it as a argparse.ArgumentParser."""

    ap = standard_parser()
    add_annotations_arguments(ap)
    add_task_arguments(ap)

    return ap.parse_args()


def main():
    """Creates and saves the modeling tasks."""

    args = parse_arguments()

    task_name = to_class_name(args.task)
    task = getattr(modeling_task, task_name)(
        ranking_size=args.ranking_size,
        batch_size=args.batch_size,
        context_format=args.context_format,
        targets_format=args.targets_format,
        context_max_size=args.context_max_size,
        k_cross_validation=int(args.cross_validation) * args.k_cross_validation,
        valid_proportion=args.valid_proportion,
        test_proportion=args.test_proportion,
        random_seed=args.random_seed,
        save=not args.no_save,
        silent=args.silent,
        results_path=args.task_path,
        annotation_task_results_path=args.annotations_path,
    )

    task.process_data_loaders()

    if args.classification:
        task.process_classification_task(args.finetuning_data_path)

    if args.generation:
        task.process_generation_task(args.finetuning_data_path)


if __name__ == "__main__":
    main()
