"""
Script to preprocess and save the annotated queries and the annotations.

Usages:
    tests:
        python preprocess_annotations.py --no_save
    regular usage:
        python preprocess_annotations.py
"""

from database_creation.annotation_task import AnnotationTask
from toolbox.parsers import standard_parser, add_annotations_arguments

from collections import defaultdict
from pickle import dump
from os import makedirs
from os.path import exists


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_annotations_arguments(ap)

    return ap.parse_args()


def filter_annotations(annotations, args):
    """
    Remove the annotations which don't meet the two criteria (annotations with not enough answers and answers from
    workers that didn't do enough assignments) and return them.

    Args:
        annotations: dict of list of Annotations, Annotations from the MT workers.
        args: argparse.ArgumentParser, parser object that contains the options of a script.
    """

    min_assignments = args.min_assignments
    min_answers = args.min_answers

    length1 = sum([len([annotation for annotation in annotation_list if annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])
    length2 = sum([len([annotation for annotation in annotation_list if not annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])

    if not args.silent:
        print("Filtering the annotations; annotations answered: %i, n/a: %i..." % (length1, length2))

    workers_count = defaultdict(list)

    for annotation_id_, annotation_list in annotations.items():
        for annotation in annotation_list:
            workers_count[annotation.worker_id].append(annotation_id_)

    worker_cmpt = 0
    for worker_id, annotation_ids in workers_count.items():
        if len(annotation_ids) < min_assignments:
            worker_cmpt += 1
            for annotation_id_ in annotation_ids:
                annotations[annotation_id_] = [annotation for annotation in annotations[annotation_id_]
                                               if annotation.worker_id != worker_id]

    length1 = sum([len([annotation for annotation in annotation_list if annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])
    length2 = sum([len([annotation for annotation in annotation_list if not annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])

    if not args.silent:
        print("Number of workers discarded: %i" % worker_cmpt)
        print("First filter done (number of assignments); annotations answered: %i, n/a: %i..." % (length1, length2))

    annotations = {id_: annotation_list for id_, annotation_list in annotations.items()
                   if len([annotation for annotation in annotation_list if not annotation.bug]) >= min_answers}

    length1 = sum([len([annotation for annotation in annotation_list if annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])
    length2 = sum([len([annotation for annotation in annotation_list if not annotation.preprocessed_answers])
                   for _, annotation_list in annotations.items()])

    if not args.silent:
        print("Second filter done (number of answers); annotations answered: %i, n/a %i.\n" % (length1, length2))

    return annotations


def save_pkl(annotations, queries, args):
    """
    Saves the annotations and the queries using pickle.

    Args:
        annotations: dict of list of Annotations, Annotations from the MT workers.
        queries: dict of Queries, Queries of the annotations.
        args: argparse.ArgumentParser, parser object that contains the options of a script.
    """

    path = args.annotations_path + "annotations/"
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
                                     results_path=args.annotations_path,
                                     years=None,
                                     max_tuple_size=None,
                                     short=None,
                                     short_size=None,
                                     random=None,
                                     debug=None,
                                     random_seed=None,
                                     save=None,
                                     corpus_path=None)

    annotation_task.process_task(exclude_pilot=args.exclude_pilot)

    queries = annotation_task.queries
    annotations = annotation_task.annotations

    annotations = filter_annotations(annotations, args=args)

    save_pkl(queries=queries, annotations=annotations, args=args)


if __name__ == '__main__':
    main()
