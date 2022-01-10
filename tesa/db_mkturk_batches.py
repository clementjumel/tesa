"""
database_creation script to create the annotation batches (for Amazon mkturk) from the Queries saved.

Usages:
    tests:
        python db_mkturk_batches.py -b 1 -bs 10 --short --no_save
        python db_mkturk_batches.py -b 2 -bs 10 --no_save
    regular usages:
        python db_mkturk_batches.py -b 10 -bs 400
        python db_mkturk_batches.py -b 10 -bs 400 --silent
"""

from database_creation.annotation_task import AnnotationTask
from toolbox.utils import standard_parser
from toolbox.parameters import YEARS, MAX_TUPLE_SIZE, RANDOM, ANNOTATION_TASK_SEED, EXCLUDE_PILOT
from toolbox.paths import NYT_ANNOTATED_CORPUS_PATH, ANNOTATION_TASK_RESULTS_PATH


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()

    ap.add_argument("-b", "--batches", required=True, type=int, help="Number of batches to generate.")
    ap.add_argument("-bs", "--batch_size", required=True, type=int, help="Size of the batches to generate.")
    ap.add_argument("--short", action='store_true', help="Shorten corpus option.")

    return ap.parse_args()


def main():
    """ Creates and saves annotation batches. """

    args = parse_arguments()

    annotation_task = AnnotationTask(years=YEARS,
                                     max_tuple_size=MAX_TUPLE_SIZE,
                                     short=args.short,
                                     short_size=None,
                                     random=RANDOM,
                                     debug=None,
                                     random_seed=ANNOTATION_TASK_SEED,
                                     save=not args.no_save,
                                     silent=args.silent,
                                     corpus_path=NYT_ANNOTATED_CORPUS_PATH,
                                     results_path=ANNOTATION_TASK_RESULTS_PATH)

    annotation_task.process_queries(load=True)

    annotation_task.process_annotation_batches(batches=args.batches,
                                               batch_size=args.batch_size,
                                               exclude_pilot=EXCLUDE_PILOT)


if __name__ == '__main__':
    main()
