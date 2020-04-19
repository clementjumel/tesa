"""
modeling_task script to create the modeling task to be solved by models.

Usages:
    tests:
        python mt_task.py -t context_free_same_type --no_save
        python mt_task.py -t context_free_same_type -rs 0 -cf v0 -tf v0 --generation --no_save
        python mt_task.py -t context_free_same_type -rs 32 -bs 16 -cf v0 --classification --no_save
    regular usages:
        python mt_task.py -t context_free_same_type
        python mt_task.py -t context_free_same_type -rs 0 -cf v0 -tf v0 --generation
        python mt_task.py -t context_free_same_type -rs 32 -bs 16 -cf v0 --classification
"""

import modeling.modeling_task as modeling_task
from toolbox.utils import standard_parser, add_task_arguments, to_class_name
from toolbox.parameters import K_CROSS_VALIDATION, MODELING_TASK_SEED
from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH, MODELING_TASK_RESULTS_PATH, FINETUNING_DATA_PATH


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_task_arguments(ap)

    ap.add_argument("--generation", action='store_true', help="Generation finetuning option.")
    ap.add_argument("--classification", action='store_true', help="Classification finetuning option.")
    ap.add_argument("-cf", "--context_format", default=None, type=str, help="Version of the context_format.")
    ap.add_argument("-tf", "--targets_format", default=None, type=str, help="Version of the targets_format.")

    return ap.parse_args()


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task_name = to_class_name(args.task)
    task = getattr(modeling_task, task_name)(ranking_size=args.ranking_size,
                                             batch_size=args.batch_size,
                                             context_format=args.context_format,
                                             targets_format=args.targets_format,
                                             k_cross_validation=int(args.cross_validation) * K_CROSS_VALIDATION,
                                             valid_proportion=args.valid_proportion,
                                             test_proportion=args.test_proportion,
                                             random_seed=MODELING_TASK_SEED,
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
