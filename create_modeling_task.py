"""
Script to create the modeling task to be solved by models.

Usages:
    tests:
        python create_modeling_task.py -t context_free -vp 0.5 -tp 0.5 --no_save
        python create_modeling_task.py -t context_free_same_type -rs 0 --no_save
        python create_modeling_task.py -t hybrid -rs 64 -bs 32 --no_save
        python create_modeling_task.py -t hybrid_same_type --short --no_save
        python create_modeling_task.py -t full_hybrid --cross_validation --no_save
        python create_modeling_task.py -t context_dependent --rte_like --cnndm_like --no_save
        python create_modeling_task.py -t context_dependent_same_type -vp 0.1 -tp 0.1 -rs 32 -bs 16 --short \
            --cross_validation --rte_like --cnndm_like --no_save
        python create_modeling_task.py -t context_dependent_same_type -vp 0.1 -tp 0.1 -rs 32 -bs 16 --short \
            --cross_validation --rte_like --cnndm_like --silent --no_save
    regular usages:
        python create_modeling_task.py -t context_free_same_type --short
        python create_modeling_task.py -t context_free_same_type -rs 32 -bs 16 --short --rte_like --cnndm_like
"""

import database_creation.modeling_task as modeling_task
from toolbox.utils import to_class_name, add_task_argument
from toolbox.parameters import MIN_ASSIGNMENTS, MIN_ANSWERS, EXCLUDE_PILOT, DROP_LAST, K_CROSS_VALIDATION, \
    MODELING_TASK_SHORT_SIZE, MODELING_TASK_SEED
from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH, MODELING_TASK_RESULTS_PATH

from argparse import ArgumentParser


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()
    add_task_argument(ap)

    ap.add_argument("--rte_like", action='store_true', help="RTE-like saving option.")
    ap.add_argument("--cnndm_like", action='store_true', help="CNN/DM-like saving option.")
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

    return vars(ap.parse_args())


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task = getattr(modeling_task, to_class_name(args['task']))(ranking_size=args['ranking_size'],
                                                               min_assignments=MIN_ASSIGNMENTS,
                                                               min_answers=MIN_ANSWERS,
                                                               exclude_pilot=EXCLUDE_PILOT,
                                                               annotation_results_path=ANNOTATION_TASK_RESULTS_PATH,
                                                               batch_size=args['batch_size'],
                                                               drop_last=DROP_LAST,
                                                               k_cross_validation=
                                                               int(args['cross_validation']) * K_CROSS_VALIDATION,
                                                               valid_proportion=args['valid_proportion'],
                                                               test_proportion=args['test_proportion'],
                                                               random_seed=MODELING_TASK_SEED,
                                                               save=not args['no_save'],
                                                               silent=args['silent'],
                                                               results_path=MODELING_TASK_RESULTS_PATH)

    task.process_data_loaders()

    if args['rte_like']:
        task.process_dataset_like("rte")

    if args['cnndm_like']:
        task.process_dataset_like("cnndm")

    if args['short']:
        task.process_short_task(size=MODELING_TASK_SHORT_SIZE)


if __name__ == '__main__':
    main()
