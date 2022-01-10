"""
database_creation script to create the Queries for the annotations.

Usages:
    tests:
        python db_mkturk_queries.py --short --debug --no_save
        python db_mkturk_queries.py --short --debug
    regular usages:
        python db_mkturk_queries.py
        python db_mkturk_queries.py --silent
"""

from database_creation.annotation_task import AnnotationTask
from toolbox.utils import standard_parser
from toolbox.parameters import YEARS, MAX_TUPLE_SIZE, RANDOM, ANNOTATION_TASK_SHORT_SIZE, ANNOTATION_TASK_SEED, \
    LOAD_WIKI, WIKIPEDIA_FILE_NAME, CORRECT_WIKI
from toolbox.paths import NYT_ANNOTATED_CORPUS_PATH, ANNOTATION_TASK_RESULTS_PATH


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()

    ap.add_argument("--short", action='store_true', help="Shorten corpus option.")
    ap.add_argument("--debug", action='store_true', help="Corpus' articles debugging option.")

    return ap.parse_args()


def main():
    """ Creates and saves the queries of the annotation task. """

    args = parse_arguments()

    annotation_task = AnnotationTask(years=YEARS,
                                     max_tuple_size=MAX_TUPLE_SIZE,
                                     short=args.short,
                                     short_size=ANNOTATION_TASK_SHORT_SIZE,
                                     random=RANDOM,
                                     debug=args.debug,
                                     random_seed=ANNOTATION_TASK_SEED,
                                     save=not args.no_save,
                                     silent=args.silent,
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
