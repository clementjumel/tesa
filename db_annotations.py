"""
Script to save the annotated queries and the annotations.

Usages:
    tests:
        python db_annotations.py --no_save
    regular usage:
        python db_annotations.py --silent
"""

from database_creation.annotation_task import AnnotationTask
from toolbox.utils import standard_parser
from toolbox.parameters import MIN_ASSIGNMENTS, MIN_ANSWERS, EXCLUDE_PILOT
from toolbox.paths import ANNOTATION_TASK_RESULTS_PATH

from collections import defaultdict
from pickle import dump
from os import makedirs
from os.path import exists


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    return standard_parser().parse_args()


def filter_annotations(annotations, min_assignments, min_answers, args):
    """
    Remove the annotations which don't meet the two criteria (annotations with not enough answers and answers from
    workers that didn't do enough assignments) and return them.

    Args:
        annotations: dict of list of Annotations, Annotations from the MT workers.
        min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
        min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.
        args: argparse.ArgumentParser, parser object that contains the options of a script.
    """

    length = sum([len(annotation_list) for _, annotation_list in annotations.items()])

    if not args.silent:
        print("Filtering the annotations (initial number of annotations: %i)..." % length)

    workers_count = defaultdict(list)

    for annotation_id_, annotation_list in annotations.items():
        for annotation in annotation_list:
            workers_count[annotation.worker_id].append(annotation_id_)

    for worker_id, annotation_ids in workers_count.items():
        if len(annotation_ids) < min_assignments:
            for annotation_id_ in annotation_ids:
                annotations[annotation_id_] = [annotation for annotation in annotations[annotation_id_]
                                               if annotation.worker_id != worker_id]

    length = sum([len(annotation_list) for _, annotation_list in annotations.items()])

    if not args.silent:
        print("First filter done (number of assignments): %i remaining..." % length)

    annotations = {id_: annotation_list for id_, annotation_list in annotations.items()
                   if len([annotation for annotation in annotation_list if not annotation.bug]) >= min_answers}

    length = sum([len(annotation_list) for _, annotation_list in annotations.items()])

    if not args.silent:
        print("Second filter done (number of answers): %i remaining.\n" % length)

    return annotations


def save_pkl(annotations, queries, path, args):
    """
    Saves the annotations and the queries using pickle.

    Args:
        annotations: dict of list of Annotations, Annotations from the MT workers.
        queries: dict of Queries, Queries of the annotations.
        path: str, path of the folder to save in.
        args: argparse.ArgumentParser, parser object that contains the options of a script.
    """

    annotations_fname = path + "annotations.pkl"
    queries_fname = path + "queries.pkl"

    if not args.no_save:
        if not exists(path):
            makedirs(path)
            if not args.silent:
                print("Folder(s) created at %s." % path)

        with open(annotations_fname, 'wb') as annotations_file, open(queries_fname, 'wb') as queries_file:
            dump(obj=annotations, file=annotations_file, protocol=-1)
            dump(obj=queries, file=queries_file, protocol=-1)

            if not args.silent:
                print("Files annotations.pkl & queries.pkl saved at %s." % path)

    elif not args.silent:
        print("Files annotations.pkl & queries.pkl not saved at %s (not in save mode)." % path)


def main():
    """ Save in a .pkl the annotated queries and the annotations. """

    args = parse_arguments()

    annotation_task = AnnotationTask(silent=args.silent,
                                     results_path=ANNOTATION_TASK_RESULTS_PATH,
                                     years=None,
                                     max_tuple_size=None,
                                     short=None,
                                     short_size=None,
                                     random=None,
                                     debug=None,
                                     random_seed=None,
                                     save=None,
                                     corpus_path=None)

    annotation_task.process_task(exclude_pilot=EXCLUDE_PILOT)

    queries = annotation_task.queries
    annotations = annotation_task.annotations

    annotations = filter_annotations(annotations=annotations,
                                     min_assignments=MIN_ASSIGNMENTS,
                                     min_answers=MIN_ANSWERS,
                                     args=args)

    save_pkl(queries=queries, annotations=annotations, path=ANNOTATION_TASK_RESULTS_PATH + "annotations/", args=args)


if __name__ == '__main__':
    main()
