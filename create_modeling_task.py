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
from toolbox.utils import to_class_name

from argparse import ArgumentParser

from toolbox.parameters import \
    MIN_ASSIGNMENTS, MIN_ANSWERS, EXCLUDE_PILOT, DROP_LAST, K_CROSS_VALIDATION, \
    MODELING_TASK_SHORT_SIZE, MODELING_TASK_SEED

from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH, MODELING_TASK_RESULTS_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-vp", "--valid_proportion", default=0.25, type=float, help="Proportion of the validation set.")
    ap.add_argument("-tp", "--test_proportion", default=0.25, type=float, help="Proportion of the test set.")
    ap.add_argument("-rs", "--ranking_size", default=None, type=int, help="Size of the ranking tasks.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int, help="Size of the batches to generate.")
    ap.add_argument("--short", action='store_true', help="Shorten modeling task option.")
    ap.add_argument("--cross_validation", action='store_true', help="Cross validation option.")
    ap.add_argument("--rte_like", action='store_true', help="RTE-like saving option.")
    ap.add_argument("--cnndm_like", action='store_true', help="CNN/DM-like saving option.")
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

    args = vars(ap.parse_args())

    return args


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task_name = to_class_name(args['task'])
    valid_proportion = args['valid_proportion']
    test_proportion = args['test_proportion']
    ranking_size = args['ranking_size']
    batch_size = args['batch_size']
    short = args['short']
    k_cross_validation = int(args['cross_validation']) * K_CROSS_VALIDATION
    rte_like = args['rte_like']
    cnndm_like = args['cnndm_like']
    save = not args['no_save']
    silent = args['silent']

    task = getattr(modeling_task, task_name)(ranking_size=ranking_size,
                                             min_assignments=MIN_ASSIGNMENTS,
                                             min_answers=MIN_ANSWERS,
                                             exclude_pilot=EXCLUDE_PILOT,
                                             annotation_results_path=ANNOTATION_TASK_RESULTS_PATH,
                                             batch_size=batch_size,
                                             drop_last=DROP_LAST,
                                             k_cross_validation=k_cross_validation,
                                             valid_proportion=valid_proportion,
                                             test_proportion=test_proportion,
                                             random_seed=MODELING_TASK_SEED,
                                             save=save,
                                             silent=silent,
                                             results_path=MODELING_TASK_RESULTS_PATH)

    task.process_data_loaders()

    if rte_like:
        task.process_dataset_like("rte")

    if cnndm_like:
        task.process_dataset_like("cnndm")

    if short:
        task.process_short_task(size=MODELING_TASK_SHORT_SIZE)


if __name__ == '__main__':
    main()
