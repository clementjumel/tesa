"""
Script to create the modeling task to be solved by models.

Usages:
    tests:
        python create_modeling_task.py -t context_free_same_type --no_save
        python create_modeling_task.py -t context_free_same_type --classification -cf v0 --no_save
        python create_modeling_task.py -t context_free_same_type --generation -cf v0 -tf v0 --no_save
    regular usages:
        python create_modeling_task.py -t context_free_same_type --classification -cf v0
        python create_modeling_task.py -t context_free_same_type --generation -cf v0 -tf v0
"""

import modeling.modeling_task as modeling_task
from toolbox.parsers import standard_parser, add_task_arguments
from toolbox.utils import to_class_name
from toolbox.parameters import K_CROSS_VALIDATION, RANDOM_SEED
from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH, MODELING_TASK_RESULTS_PATH, FINETUNING_DATA_PATH


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_task_arguments(ap)

    return ap.parse_args()


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task_name = to_class_name(args.task)
    task = getattr(modeling_task, task_name)(ranking_size=args.ranking_size,
                                             batch_size=args.batch_size,
                                             context_format=args.context_format,
                                             targets_format=args.targets_format,
                                             context_max_size=args.context_max_size,
                                             k_cross_validation=int(args.cross_validation) * K_CROSS_VALIDATION,
                                             valid_proportion=args.valid_proportion,
                                             test_proportion=args.test_proportion,
                                             random_seed=RANDOM_SEED,
                                             save=not args.no_save,
                                             silent=args.silent,
                                             results_path=MODELING_TASK_RESULTS_PATH,
                                             annotation_task_results_path=ANNOTATION_TASK_RESULTS_PATH)

    task.process_data_loaders()

    if args.classification:
        task.process_classification_task(FINETUNING_DATA_PATH)

    if args.generation:
        task.process_generation_task(FINETUNING_DATA_PATH)


if __name__ == '__main__':
    main()
