from database_creation.annotation_task import AnnotationTask

from argparse import ArgumentParser

from toolbox.parameters import \
    YEARS, MAX_TUPLE_SIZE, RANDOM, ANNOTATION_TASK_SHORT_SIZE, ANNOTATION_TASK_SEED, \
    LOAD_WIKI, WIKIPEDIA_FILE_NAME, CORRECT_WIKI

from toolbox.paths import NYT_ANNOTATED_CORPUS_PATH, ANNOTATION_TASK_RESULTS_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("--short", action='store_true', help="Shorten corpus option.")
    ap.add_argument("--debug", action='store_true', help="Corpus' articles debugging option.")
    ap.add_argument("--no_save", action='store_true', help="No save option.")
    ap.add_argument("--silent", action='store_true', help="Silence option.")

    args = vars(ap.parse_args())

    return args


def main():
    """ Creates and saves the queries of the annotation task. """

    args = parse_arguments()

    short = args['short']
    debug = args['debug']
    save = not args['no_save']
    silent = args['silent']

    annotation_task = AnnotationTask(years=YEARS,
                                     max_tuple_size=MAX_TUPLE_SIZE,
                                     short=short,
                                     short_size=ANNOTATION_TASK_SHORT_SIZE,
                                     random=RANDOM,
                                     debug=debug,
                                     random_seed=ANNOTATION_TASK_SEED,
                                     save=save,
                                     silent=silent,
                                     corpus_path=NYT_ANNOTATED_CORPUS_PATH,
                                     results_path=ANNOTATION_TASK_RESULTS_PATH)

    annotation_task.preprocess_database()
    annotation_task.process_articles()
    annotation_task.process_wikipedia(load=LOAD_WIKI, file_name=WIKIPEDIA_FILE_NAME)

    if CORRECT_WIKI:
        annotation_task.correct_wiki(step=None, out_name=WIKIPEDIA_FILE_NAME)

    annotation_task.process_queries(load=False)


if __name__ == '__main__':
    main()
