from database_creation.annotation_task import AnnotationTask

# TODO
# from parameters import


def main():
    """ Creates and saves the queries of an annotation task. """

    # TODO: check if runs
    annotation_task = AnnotationTask()

    annotation_task.preprocess_database()
    annotation_task.process_articles()

    annotation_task.process_wikipedia(load=True)
    annotation_task.correct_wiki()

    annotation_task.process_queries(check_changes=True, csv_seed=1)


if __name__ == '__main__':
    main()
