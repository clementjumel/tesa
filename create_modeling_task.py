import database_creation.modeling_task as modeling_task
from toolbox.utils import to_class_name

from argparse import ArgumentParser

from toolbox.parameters import \
    MIN_ASSIGNMENTS, MIN_ANSWERS, EXCLUDE_PILOT, DROP_LAST, K_CROSS_VALIDATION, \
    MODELING_TASK_SHORT_SIZE, MODELING_TASK_SEED, \
    BASELINES_SPLIT_VALID_PROPORTION, BASELINES_SPLIT_TEST_PROPORTION, \
    MODELS_SPLIT_VALID_PROPORTION, MODELS_SPLIT_TEST_PROPORTION

from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH, MODELING_TASK_FOR_BASELINES_PATH, MODELING_TASK_FOR_MODELS_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-s", "--size", default=64, type=int, help="Size of the batches to generate.")
    ap.add_argument("--short", action='store_true', help="Shorten modeling task option.")
    ap.add_argument("--cross_validation", action='store_true', help="Cross validation option.")
    ap.add_argument("--all_batches", action='store_true', help="Batches for all loaders option.")
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

    args = vars(ap.parse_args())

    return args


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task_name = to_class_name(args['task'])
    batch_size = args['size']
    short = args['short']
    k_cross_validation = int(args['cross_validation']) * K_CROSS_VALIDATION
    all_batches = args['all_batches']
    save = not args['no_save']
    silent = args['silent']

    # Saves with only validation and test split (for baseline evaluations)
    task = getattr(modeling_task, task_name)(min_assignments=MIN_ASSIGNMENTS,
                                             min_answers=MIN_ANSWERS,
                                             exclude_pilot=EXCLUDE_PILOT,
                                             annotation_results_path=ANNOTATION_TASK_RESULTS_PATH,
                                             all_batches=all_batches,
                                             batch_size=batch_size,
                                             drop_last=DROP_LAST,
                                             k_cross_validation=None,
                                             valid_proportion=BASELINES_SPLIT_VALID_PROPORTION,
                                             test_proportion=BASELINES_SPLIT_TEST_PROPORTION,
                                             random_seed=MODELING_TASK_SEED,
                                             save=save,
                                             silent=silent,
                                             results_path=MODELING_TASK_FOR_BASELINES_PATH)

    task.process_data_loaders()

    if short:
        task.process_short_task(size=MODELING_TASK_SHORT_SIZE)

    # Saves with train, validation and test split (for model training)
    task = getattr(modeling_task, task_name)(min_assignments=MIN_ASSIGNMENTS,
                                             min_answers=MIN_ANSWERS,
                                             exclude_pilot=EXCLUDE_PILOT,
                                             annotation_results_path=ANNOTATION_TASK_RESULTS_PATH,
                                             all_batches=all_batches,
                                             batch_size=batch_size,
                                             drop_last=DROP_LAST,
                                             k_cross_validation=k_cross_validation,
                                             valid_proportion=MODELS_SPLIT_VALID_PROPORTION,
                                             test_proportion=MODELS_SPLIT_TEST_PROPORTION,
                                             random_seed=MODELING_TASK_SEED,
                                             save=save,
                                             silent=silent,
                                             results_path=MODELING_TASK_FOR_MODELS_PATH)

    task.process_data_loaders()

    if short:
        task.process_short_task(size=MODELING_TASK_SHORT_SIZE)


if __name__ == '__main__':
    main()
