from database_creation.annotation_task import AnnotationTask

from argparse import ArgumentParser

from toolbox.parameters import \
    YEARS, MAX_TUPLE_SIZE, RANDOM, ANNOTATION_TASK_SEED, EXCLUDE_PILOT

from toolbox.paths import NYT_ANNOTATED_CORPUS_PATH, ANNOTATION_TASK_RESULTS_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-b", "--batches", required=True, type=int, help="Number of batches to generate.")
    ap.add_argument("-bs", "--batch_size", required=True, type=int, help="Size of the batches to generate.")
    ap.add_argument("--short", action='store_true', help="Shorten corpus option.")
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

    args = vars(ap.parse_args())

    return args


def main():
    """ Creates and saves annotation batches. """

    args = parse_arguments()

    batches = args['batches']
    batch_size = args['batch_size']
    short = args['short']
    save = not args['no_save']
    silent = args['silent']

    annotation_task = AnnotationTask(years=YEARS,
                                     max_tuple_size=MAX_TUPLE_SIZE,
                                     short=short,
                                     random=RANDOM,
                                     debug=None,
                                     random_seed=ANNOTATION_TASK_SEED,
                                     save=save,
                                     silent=silent,
                                     corpus_path=NYT_ANNOTATED_CORPUS_PATH,
                                     results_path=ANNOTATION_TASK_RESULTS_PATH)

    annotation_task.process_queries(load=True)
    annotation_task.process_annotation_batches(batches=batches,
                                               batch_size=batch_size,
                                               exclude_pilot=EXCLUDE_PILOT)


if __name__ == '__main__':
    main()
