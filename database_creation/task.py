from database_creation.database import Database
from database_creation.utils import Sample

from collections import defaultdict
from numpy import split, concatenate, asarray
from numpy.random import shuffle


class Task:

    # region Class initialization

    def __init__(self, min_assignments=5, min_answers=2, test_proportion=0.25, valid_proportion=0.25, batch_size=32,
                 drop_last=True, k_cross_validation=0):
        """
        Initializes an instance of the base Task.

        Args:
            min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
            min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            batch_size: int, number of samples in each batch.
            drop_last: bool, whether or not to delete the last batch if incomplete.
            k_cross_validation: int, number of folds to use in k-fold cross validation (if 0, doesn't use k-fold).
        """

        self.use_cross_validation = bool(k_cross_validation)

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.compute_data(min_assignments=min_assignments,
                          min_answers=min_answers,
                          batch_size=batch_size,
                          drop_last=drop_last,
                          test_proportion=test_proportion,
                          valid_proportion=valid_proportion,
                          k_cross_validation=k_cross_validation)

    # endregion

    # region Main methods

    def preview_data(self, model, include_train=True, include_valid=False):
        """
        Preview the data for the model.

        Args:
            model: models.Model, model to train.
            include_train: bool, whether to include the train loader for the preview.
            include_valid: bool, whether to include the valid loader for the preview.
        """

        assert include_train or include_valid

        if include_train and include_valid:
            data_loader = concatenate((self.train_loader, self.valid_loader), axis=0)
        else:
            data_loader = self.train_loader if include_train else self.valid_loader

        model.preview_data(data_loader=data_loader)

    def train_model(self, model, n_epochs=1, n_updates=100, is_regression=False):
        """
        Train a model on the training set and compute the metrics on the validation sets.

        Args:
            model: models.Model, model to train.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
        """

        if not self.use_cross_validation:
            model.train(train_loader=self.train_loader,
                        valid_loader=self.valid_loader,
                        n_epochs=n_epochs,
                        n_updates=n_updates,
                        is_regression=is_regression)

        else:
            for train_loader, valid_loader in self.train_loader:
                model.reset()
                model.train(train_loader=train_loader,
                            valid_loader=valid_loader,
                            n_epochs=n_epochs,
                            n_updates=n_updates,
                            is_regression=is_regression)

    def valid_model(self, model, is_regression=False):
        """
        Evaluate a baseline model on the validation set.

        Args:
            model: models.Model, model to validate.
            is_regression: bool, whether to use the regression set up for the task.
        """

        model.valid(data_loader=self.valid_loader, is_regression=is_regression)

    def test_model(self, model, is_regression=False):
        """
        Evaluate the model on the test set.

        Args:
            model: models.Model, model to test.
            is_regression: bool, whether to use the regression set up for the task.
        """

        model.test(data_loader=self.test_loader, is_regression=is_regression)

    def explain_model(self, model, scores_names=None, n_samples=5, n_answers=10):
        """
        Explain the answers of the model on the valid_set.

        Args:
            model: models.Model, model to test.
            scores_names: iterable, names of the scores to plot, if None, display all of them.
            n_samples: int, number of samples to explain.
            n_answers: int, number of best answers to look at.
        """

        model.explain(data_loader=self.valid_loader, scores_names=scores_names, n_samples=n_samples,
                      n_answers=n_answers)

    # endregion

    # region Methods compute_

    def compute_data(self, min_assignments, min_answers, batch_size, drop_last, test_proportion, valid_proportion,
                     k_cross_validation):
        """
        Compute the learnable data loaders by processing the Dataset.

        Args:
            min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
            min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.
            batch_size: int, number of samples in each batch.
            drop_last: bool, whether or not to delete the last batch if incomplete.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            k_cross_validation: int, number of folds to create.
        """

        raw_data = self.get_raw_data(min_assignments=min_assignments, min_answers=min_answers)

        n = len(raw_data)
        n_test = round(test_proportion * n)

        if self.use_cross_validation:
            n_test += (n - n_test) % k_cross_validation

            test_set, complete_train_set = split(raw_data, [n_test])

            k_splits = split(complete_train_set, k_cross_validation)
            train_sets, valid_sets = [], []

            for i in range(k_cross_validation):
                valid_set = k_splits[i]
                train_set = concatenate([k_splits[j] for j in range(k_cross_validation) if j != i])

                train_sets.append(train_set)
                valid_sets.append(valid_set)

            s = list(set([(train_sets[i].shape, valid_sets[i].shape) for i in range(len(train_sets))]))
            assert len(s) == 1

            self.train_loader = [(self.to_loader(data=train_sets[i], batch_size=batch_size, drop_last=drop_last),
                                  valid_sets[i]) for i in range(len(train_sets))]
            self.test_loader = test_set

            print("Split into k-fold cross validation sets (train: %d, %d percents," % (s[0][0][0], 100*s[0][0][0]/n) +
                  " valid: %d, %d percents)" % (s[0][1][0], 100*s[0][1][0]/n) +
                  " and a test set (%d, %d percents)." % (test_set.shape[0], 100*test_set.shape[0]/n))

        else:
            n_valid = round(valid_proportion * n)
            n_train = n - n_test - n_valid
            assert n_train >= 0 and n_valid >= 0 and n_test >= 0

            train_set, valid_set, test_set = split(raw_data, [n_train, n_train + n_valid])

            self.train_loader = self.to_loader(data=train_set, batch_size=batch_size, drop_last=drop_last)
            self.valid_loader = valid_set
            self.test_loader = test_set

            print("Split into train (%d, %d percents)," % (train_set.shape[0], 100*train_set.shape[0]/n) +
                  " valid (%d, %d percents)" % (valid_set.shape[0], 100*valid_set.shape[0]/n) +
                  " and test (%d, %d) percents." % (test_set.shape[0], 100*test_set.shape[0]/n))

    # endregion

    # region Methods get_

    def get_raw_data(self, min_assignments, min_answers):
        """
        Returns the raw data using the methods from Dataset.

        Args:
            min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
            min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.

        Returns:
            2d-array, raw sample from the task, each line corresponding to (inputs, targets)
        """

        database = Database()
        database.process_task()

        queries, annotations = database.queries, database.annotations

        annotations = self.filter_annotations(annotations=annotations,
                                              min_assignments=min_assignments,
                                              min_answers=min_answers)

        samples = self.get_samples(queries=queries, annotations=annotations)

        raw_data = asarray([(sample.get_x(), sample.get_y()) for sample in samples])

        print("Raw data imported ({} samples).".format(raw_data.shape[0]))

        return raw_data

    def get_samples(self, queries, annotations):
        """
        Returns the samples of the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            list, shuffled list of Samples.
        """

        samples = []

        for _, sample_annotations in annotations.items():
            sample_queries_ids = sorted(set([annotation.id_ for annotation in sample_annotations]))
            sample_queries = [queries[query_id_] for query_id_ in sample_queries_ids]

            labelled_answers = self.get_labelled_answers(sample_queries=sample_queries,
                                                         sample_annotations=sample_annotations,
                                                         queries=queries,
                                                         annotations=annotations)

            samples.append(Sample(queries=sample_queries, labelled_answers=labelled_answers))

        shuffle(samples)

        return samples

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        return dict()

    @staticmethod
    def get_answers_all(annotations):
        """
        Returns a set of all the answers in the annotations.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            set, preprocessed answers of the annotations.
        """

        answers = set()

        for _, annotation_list in annotations.items():
            for annotation in annotation_list:
                for answer in annotation.preprocessed_answers:
                    answers.add(answer)

        return answers

    @staticmethod
    def get_answers_same_type(annotations, sample_queries, queries):
        """
        Returns a set of the answers in the annotations with the same entities type than sample_queries.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.
            sample_queries: list, Annotations of the Sample.
            queries: dict of Query, Queries of the annotations.

        Returns:
            set, preprocessed answers of the annotations.
        """

        assert len(set([query.entities_type_ for query in sample_queries])) == 1
        entities_type_ = sample_queries[0].entities_type_

        answers = set()

        for _, annotation_list in annotations.items():
            for annotation in annotation_list:
                if entities_type_ == queries[annotation.id_].entities_type_:
                    for answer in annotation.preprocessed_answers:
                        answers.add(answer)

        return answers

    @staticmethod
    def get_answers_same_tuple(annotations, sample_queries, queries):
        """
        Returns a set of the answers in the annotations with the same entities tuple than sample_queries.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.
            sample_queries: list, Annotations of the Sample.
            queries: dict of Query, Queries of the annotations.

        Returns:
            set, preprocessed answers of the annotations.
        """

        assert len(set([tuple(sorted(query.entities)) for query in sample_queries])) == 1
        entities_tuple_ = tuple(sorted(sample_queries[0].entities))

        answers = set()

        for _, annotation_list in annotations.items():
            for annotation in annotation_list:
                if entities_tuple_ == tuple(sorted(queries[annotation.id_].entities)):
                    for answer in annotation.preprocessed_answers:
                        answers.add(answer)

        return answers

    @staticmethod
    def get_answers_sample(sample_annotations):
        """
        Returns a set of all the answers in the sample annotations.

        Args:
            sample_annotations: list of annotations, annotations of the Sample.

        Returns:
            set, preprocessed answers of the annotations.
        """

        answers = set()

        for annotation in sample_annotations:
            for answer in annotation.preprocessed_answers:
                answers.add(answer)

        return answers

    # endregion

    # region Other methods

    @staticmethod
    def filter_annotations(annotations, min_assignments, min_answers):
        """
        Returns the Annotations filtered from those with not enough answers or with workers having not done enough
        assignments.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.
            min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
            min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.

        Returns:
            dict of list of Annotations, filtered Annotations from the MT workers.
        """

        length = sum([len(annotation_list) for _, annotation_list in annotations.items()])
        print("Filtering the annotations (initial number of annotations: %d)...".format(length))

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
        print("First filter done (number of assignments): %d remaining...".format(length))

        annotations = {id_: annotation_list for id_, annotation_list in annotations.items()
                       if len([annotation for annotation in annotation_list if not annotation.bug]) >= min_answers}

        length = sum([len(annotation_list) for _, annotation_list in annotations.items()])
        print("Second filter done (number of answers): %d remaining.".format(length))

        return annotations

    @staticmethod
    def to_loader(data, batch_size, drop_last):
        """
        Returns the loader of the data.

        Args:
            data: 2d-array, data samples, each line corresponding to (inputs, targets).
            batch_size: int, number of samples in each batch.
            drop_last: bool, whether or not to delete the last batch if incomplete.

        Returns:
            data_loader: list, pairs of (batch_inputs, batch_targets), batched data samples.
        """

        data_loader = []

        for inputs, targets in data:

            n = len(inputs['choices'])
            cmpt = 0

            while (drop_last and cmpt + batch_size <= n) or (not drop_last and cmpt + 1 <= n):
                batch_inputs = dict([(key, item) for key, item in inputs.items() if key != 'choices'])
                batch_inputs['choices'] = inputs['choices'][cmpt:cmpt + batch_size]

                batch_targets = targets[cmpt:cmpt + batch_size]

                data_loader.append([batch_inputs, batch_targets])

                cmpt += len(batch_targets)

        shuffle(data_loader)
        data_loader = asarray(data_loader)

        return data_loader

    # endregion


class ContextFreeTask(Task):

    # region Methods get_

    @staticmethod
    def get_annotations(annotations):
        """
        Rework the annotations for the specificity of the Task.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            dict of list of Annotations, Annotations from the MT workers.
        """

        new_annotations = defaultdict(list)

        for id_, annotation_list in annotations.items():
            for annotation in annotation_list:
                entities_tuple = tuple(sorted(annotation.entities))
                new_annotations[entities_tuple].append(annotation)

        return new_annotations

    def get_samples(self, queries, annotations):
        """
        Returns the samples of the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            list, shuffled list of Samples.
        """

        return super().get_samples(queries=queries, annotations=self.get_annotations(annotations))

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers

    # endregion


class ContextFreeSameTypeTask(ContextFreeTask):

    # region Methods get_

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers

    # endregion


class ContextDependentTask(Task):

    # region Methods get_

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers

    # endregion


class ContextDependentSameTypeTask(ContextDependentTask):

    # region Methods get_

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers

    # endregion


class HybridTask(ContextDependentTask):

    # region Methods get_

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        for answer in answers:
            labelled_answers[answer] = 1

        answers = self.get_answers_same_tuple(annotations=annotations, sample_queries=sample_queries, queries=queries)
        for answer in answers:
            labelled_answers[answer] = 2

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 3

        return labelled_answers

    # endregion


class HybridSameTypeTask(HybridTask):

    # region Methods get_

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        """
        Returns the answers and their labels as a list of tuples.

        Args:
            sample_queries: list of queries, queries of the Sample.
            sample_annotations: list of annotations, annotations of the Sample.
            queries: dict of Query, Queries of the annotations.
            annotations: dict of Annotations, all the annotations.

        Returns:
            dict, answers and their labels (0 for negative answers).
        """

        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_same_tuple(annotations=annotations, sample_queries=sample_queries, queries=queries)
        for answer in answers:
            labelled_answers[answer] = 1

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 2

        return labelled_answers

    # endregion


def main():
    pass


if __name__ == '__main__':
    main()
