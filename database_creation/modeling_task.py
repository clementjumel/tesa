from database_creation.annotation_task import AnnotationTask
from database_creation.utils import Sample

from collections import defaultdict
from numpy import asarray, split, concatenate
from numpy.random import seed, shuffle, choice
from pickle import dump
from re import findall
from csv import writer
import os


class ModelingTask:
    relevance_level = None

    def __init__(self, ranking_size, min_assignments, min_answers, exclude_pilot, annotation_results_path,
                 batch_size, drop_last, k_cross_validation, valid_proportion, test_proportion, random_seed, save,
                 silent, results_path):
        """
        Initializes an instance of the base ModelingTask.

        Args:
            ranking_size: int, total number of choices to compute; if None, computes all of them; if 0, computes
                only the positive examples.
            min_assignments: int, minimum number of assignments a worker has to have done to be taken into account.
            min_answers: int, minimum number of annotators that answers an annotation for it to be taken into account.
            exclude_pilot: bool, whether or not to exclude the data from the pilot.
            annotation_results_path: str, path to the annotation results folder.
            batch_size: int, number of samples in each batch.
            drop_last: bool, whether or not to delete the last batch if incomplete.
            k_cross_validation: int, number of folds to use in k-fold cross validation (if 0, doesn't use k-fold).
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            random_seed: int, the seed to use for the random processes.
            save: bool, saving option.
            silent: bool, silent option.
            results_path: str, path to the folder to save the task in.
        """

        self.ranking_size = ranking_size
        self.min_assignments = min_assignments
        self.min_answers = min_answers
        self.exclude_pilot = exclude_pilot
        self.annotation_results_path = annotation_results_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.k_cross_validation = k_cross_validation
        self.valid_proportion = valid_proportion
        self.test_proportion = test_proportion
        self.save = save
        self.silent = silent
        self.results_path = results_path

        self.unprocessed_data = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.short = False

        seed(random_seed)

    # region Main methods

    def process_data_loaders(self):
        """ Process the data of the annotations to create the data loaders. """

        self.compute_unprocessed_data()
        self.compute_data_loaders()
        self.save_pkl()

    def process_rte_like(self):
        """ Saves the data_loaders in .tsv files with a similar format as the RTE task of GLUE. """

        self.save_rte_like()

    # TODO
    def process_summaries_like(self):
        """  """

        self.save_summaries_like()

    def process_short_task(self, size):
        """
        Shorten the data_loaders by keeping only the [size] first batches for each.

        Args:
            size: int, number of batches to keep for each loader.
        """

        self.compute_shorten_loaders(size=size)
        self.save_pkl()

    def preview_data(self, model, include_train, include_valid):
        """
        Preview the data for the model.

        Args:
            model: models.Model, model to train.
            include_train: bool, whether to include the train loader for the preview.
            include_valid: bool, whether to include the valid loader for the preview.
        """

        assert include_train or include_valid

        if include_train and include_valid:
            data_loader = concatenate((self.train_loader, self.valid_loader))
        elif include_train:
            data_loader = self.train_loader
        else:
            data_loader = self.valid_loader

        shuffle(data_loader)
        model.preview_data(data_loader)

    def train_model(self, model):
        """
        Train a model on the training set and compute the metrics on the validation sets.

        Args:
            model: models.Model, model to train.
        """

        if not self.k_cross_validation:
            model.train(train_loader=self.train_loader,
                        valid_loader=self.valid_loader)

        else:
            for train_loader, valid_loader in zip(self.train_loader, self.valid_loader):
                model.reset()
                model.train(train_loader=train_loader,
                            valid_loader=valid_loader)

    def valid_model(self, model):
        """
        Evaluate a baseline model on the validation set.

        Args:
            model: models.Model, model to validate.
        """

        model.valid(self.valid_loader)

    def test_model(self, model):
        """
        Evaluate the model on the test set.

        Args:
            model: models.Model, model to test.
        """

        model.test(self.test_loader)

    def show_model(self, model):
        """
        Show the answers of the model on the valid_set.

        Args:
            model: models.Model, model to test.
        """

        model.show(self.valid_loader)

    # endregion

    # region Methods compute_

    def compute_unprocessed_data(self):
        """ Computes the unprocessed annotations data as a 2d-array. """

        annotation_task = AnnotationTask(silent=self.silent,
                                         results_path=self.annotation_results_path,
                                         years=None,
                                         max_tuple_size=None,
                                         short=None,
                                         short_size=None,
                                         random=None,
                                         debug=None,
                                         random_seed=None,
                                         save=None,
                                         corpus_path=None)

        annotation_task.process_task(exclude_pilot=self.exclude_pilot)

        queries = annotation_task.queries
        annotations = self.filter_annotations(annotation_task.annotations)

        samples = self.get_samples(queries=queries, annotations=annotations)

        unprocessed_data = asarray([(sample.get_x(), sample.get_y()) for sample in samples])
        n = unprocessed_data.shape[0]

        if not n:
            raise Exception("No data imported.")
        else:
            self.print("Unprocessed data imported (%i ranking task).\n" % n)
            self.unprocessed_data = unprocessed_data

    def compute_data_loaders(self):
        """ Compute the data loaders. """

        k = self.k_cross_validation
        n = self.unprocessed_data.shape[0]
        n_test = round(self.test_proportion * n)

        if not k:
            n_valid = round(self.valid_proportion * n)
            n_train = n - n_test - n_valid

            assert 0 <= n_train <= n and 0 <= n_valid <= n and 0 <= n_test <= n

            train_set, valid_set, test_set = split(self.unprocessed_data, [n_train, n_train + n_valid])

            assert n_train == train_set.shape[0] and n_valid == valid_set.shape[0] and n_test == test_set.shape[0]

            self.train_loader = self.to_loader(train_set)
            self.valid_loader = self.to_loader(valid_set)
            self.test_loader = self.to_loader(test_set)

            train_loader, valid_loader, test_loader = self.train_loader, self.valid_loader,  self.test_loader

            self.print("Data loaders computed:")

        else:
            n_test += (n - n_test) % k

            test_set, cross_validation_set = split(self.unprocessed_data, [n_test])
            cross_validation_split = split(cross_validation_set, k)

            train_sets, valid_sets = [], []
            n_trains, n_valids = set(), set()

            for i in range(k):
                train_set = concatenate([cross_validation_split[j] for j in range(k) if j != i])
                valid_set = cross_validation_split[i]

                train_sets.append(train_set)
                valid_sets.append(valid_set)

                n_trains.add(train_set.shape[0])
                n_valids.add(valid_set.shape[0])

            assert len(n_trains) == 1 and len(n_valids) == 1

            n_train, n_valid = n_trains.pop(), n_valids.pop()

            self.train_loader = [self.to_loader(train_set) for train_set in train_sets]
            self.valid_loader = [self.to_loader(valid_set) for valid_set in valid_sets]
            self.test_loader = self.to_loader(test_set)

            train_loader, valid_loader, test_loader = self.train_loader[0], self.valid_loader[0], self.test_loader

            self.print("Data loaders for %i-fold cross validation computed:" % k)

        m_train = sum([len(ranking_task) for ranking_task in train_loader])
        m_valid = sum([len(ranking_task) for ranking_task in valid_loader])
        m_test = sum([len(ranking_task) for ranking_task in test_loader])

        self.print("   train: %i ranking tasks (%i%%), %i batches,\n" % (n_train, 100 * n_train / n, m_train),
                   "  valid: %i ranking tasks (%i%%), %i batches,\n" % (n_valid, 100 * n_valid / n, m_valid),
                   "  test: %i ranking tasks (%i%%), %i batches.\n" % (n_test, 100 * n_test / n, m_test))

        self.unprocessed_data = None

    def compute_shorten_loaders(self, size):
        """
        Shorten the data loaders by keeping only their [size] first batches of the first ranking task.

        Args:
            size: int, number of batches to keep for each loader.
        """

        if not self.k_cross_validation:
            self.train_loader = self.train_loader[0:1][0:size]
            self.valid_loader = self.valid_loader[0:1][0:size]

        else:
            self.train_loader = [train_loader[0:1][0:size] for train_loader in self.train_loader]
            self.valid_loader = [valid_loader[0:1][0:size] for valid_loader in self.valid_loader]

        self.test_loader = self.test_loader[0:1][0:size]

        self.short = True

        self.print("Data_loaders shorten to keep only the first %i batch(es) of the first ranking task.\n" % size)

    # endregion

    # region Methods get_

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

            labelled_answers = self.filter_ranking_size(labelled_answers)

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

    def filter_ranking_size(self, labelled_answers):
        """
        Returns a filtered version of labelled answers with the same format, depending of self.ranking_size: if it is
        None: don't do anything, if it is 0, remove all the negative answers, otherwise, remove negative answers to keep
        only self.ranking_size total answers.

        Args:
            labelled_answers: dict, answers and their labels (0 for negative answers).
        """

        if self.ranking_size is None:
            pass

        else:
            negative_answers = sorted([key for key, value in labelled_answers.items() if value == 0])
            labelled_answers = {key: value for key, value in labelled_answers.items() if value > 0}

            if self.ranking_size:
                n = len(labelled_answers)

                if n > self.ranking_size:
                    raise Exception("Too small ranking size, some gold standard answers will be lost.")

                else:
                    negative_answers = {answer: 0 for answer in choice(a=negative_answers, size=self.ranking_size - n)}
                    labelled_answers.update(negative_answers)

        return labelled_answers

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

    # region File methods

    def class_name(self):
        """ Returns the standardized name of the class. """

        return "".join([word.lower() for word in findall(r'[A-Z][^A-Z]*', self.__class__.__name__)])

    def suffix(self, full):
        """
        Returns the standard suffix of a file_name, either short or full.

        Args:
            full: bool, whether or not to compute the full prefix.
        """

        train_proportion = ("%.2f" % (1 - self.valid_proportion - self.test_proportion)).split(".")[1]
        valid_proportion =("%.2f" % self.valid_proportion).split(".")[1]
        test_proportion = ("%.2f" % self.test_proportion).split(".")[1]
        suffix = "_" + "-".join([train_proportion, valid_proportion, test_proportion])

        if full:
            suffix += "_rs" + str(self.ranking_size) if self.ranking_size is not None else ""
            suffix += "_bs" + str(self.batch_size)
            suffix += "_cv" if self.k_cross_validation else ""
            suffix += "_short" if self.short else ""

        return suffix

    def save_pkl(self):
        """ Save the Task using pickle in [self.results_path]. """

        file_name = self.results_path + self.class_name() + self.suffix(full=True) + '.pkl'

        if self.save:
            with open(file_name, 'wb') as file:
                dump(obj=self, file=file, protocol=-1)

            self.print("Task saved at %s.\n" % file_name)

        else:
            self.print("Not saving %s (not in save mode).\n" % file_name)

    def save_rte_like(self):
        """ Saves the data loaders similarly to GLUE's RTE task. """

        def to_tsv(data_loader, file_name):
            """
            Saves a data_loader in a .tsv file similar to RTE's ones.

            Args:
                data_loader: list, list of ranking task, which are lists of batches (batch_inputs, batch_targets).
                file_name: str, name of the file to save in.
            """

            def to_context_sentence(inputs):
                """
                Returns a string representing the context of the inputs.

                Args:
                    inputs: dict, inputs of the data.

                Returns:
                    str, sentence representing the context of the input.
                """

                sentence = ""

                for wikipedia in inputs['wikipedia']:
                    if wikipedia != "No information found.":
                        sentence += wikipedia

                sentence += " "
                sentence += inputs['context']

                return sentence

            with open(file_name, 'wt') as file:
                tsv_writer = writer(file, delimiter='\t')
                tsv_writer.writerow(['index', 'sentence1', 'sentence2', 'label'])
                index = 0

                for ranking_task in data_loader:
                    for inputs, targets in ranking_task:
                        sentence1 = to_context_sentence(inputs)

                        for choice, target in zip(inputs['choices'], targets):
                            sentence2 = choice
                            label = "entailment" if target else "not_entailment"

                            tsv_writer.writerow([str(index), sentence1, sentence2, label])
                            index += 1

        folder_name = self.results_path + self.class_name() + self.suffix(full=False) + "/"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = folder_name + "train.tsv"
        if self.save:
            to_tsv(data_loader=self.train_loader, file_name=file_name)
            self.print("Data loader saved in %s." % file_name)
        else:
            self.print("Not saving %s (not in save mode)." % file_name)

        file_name = folder_name + "dev.tsv"
        if self.save:
            to_tsv(data_loader=self.valid_loader, file_name=file_name)
            self.print("Data loader saved in %s." % file_name)
        else:
            self.print("Not saving %s (not in save mode)." % file_name)

        file_name = folder_name + "test.tsv"
        if self.save:
            to_tsv(data_loader=self.test_loader, file_name=file_name)
            self.print("Data loader saved in %s.\n" % file_name)
        else:
            self.print("Not saving %s (not in save mode).\n" % file_name)

    # TODO
    def save_summaries_like(self):
        """  """

        pass

    # endregion

    # region Other methods

    def print(self, *args):
        """ Prints only if not in silent mode. """

        if not self.silent:
            print(*args)

    def filter_annotations(self, annotations):
        """
        Returns the Annotations filtered from those with not enough answers or with workers having not done enough
        assignments.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            dict of list of Annotations, filtered Annotations from the MT workers.
        """

        length = sum([len(annotation_list) for _, annotation_list in annotations.items()])
        self.print("Filtering the annotations (initial number of annotations: %i)..." % length)

        workers_count = defaultdict(list)

        for annotation_id_, annotation_list in annotations.items():
            for annotation in annotation_list:
                workers_count[annotation.worker_id].append(annotation_id_)

        for worker_id, annotation_ids in workers_count.items():
            if len(annotation_ids) < self.min_assignments:
                for annotation_id_ in annotation_ids:
                    annotations[annotation_id_] = [annotation for annotation in annotations[annotation_id_]
                                                   if annotation.worker_id != worker_id]

        length = sum([len(annotation_list) for _, annotation_list in annotations.items()])
        self.print("First filter done (number of assignments): %i remaining..." % length)

        annotations = {id_: annotation_list for id_, annotation_list in annotations.items()
                       if len([annotation for annotation in annotation_list if not annotation.bug]) >= self.min_answers}

        length = sum([len(annotation_list) for _, annotation_list in annotations.items()])
        self.print("Second filter done (number of answers): %i remaining.\n" % length)

        return annotations

    def to_loader(self, data):
        """
        Returns the loader of the data.

        Args:
            data: 2d-array, data samples, each line corresponding to (inputs, targets).

        Returns:
            list, list of ranking task, which are lists of batches (batch_inputs, batch_targets).
        """

        data_loader = []

        for inputs, targets in data:
            n = len(inputs['choices'])
            cmpt = 0
            ranking_task = []

            while (self.drop_last and cmpt + self.batch_size <= n) or (not self.drop_last and cmpt + 1 <= n):
                batch_inputs = dict([(key, item) for key, item in inputs.items() if key != 'choices'])
                batch_inputs['choices'] = inputs['choices'][cmpt:cmpt + self.batch_size]
                batch_targets = targets[cmpt:cmpt + self.batch_size]

                ranking_task.append((batch_inputs, batch_targets))
                cmpt += len(batch_targets)

            shuffle(ranking_task)
            data_loader.append(ranking_task)

        shuffle(data_loader)

        return data_loader

    # endregion


class ContextFree(ModelingTask):
    relevance_level = 1

    # region Methods get_

    @staticmethod
    def rework_annotations(queries, annotations):
        """
        Rework the annotations for the specificity of the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            dict of list of Annotations, Annotations from the MT workers.
        """

        new_annotations = defaultdict(list)

        for id_, annotation_list in annotations.items():
            for annotation in annotation_list:
                entities_tuple = tuple(sorted(queries[annotation.id_].entities))
                new_annotations[entities_tuple].append(annotation)

        return new_annotations

    def get_samples(self, queries, annotations):
        annotations = self.rework_annotations(queries=queries, annotations=annotations)

        return super().get_samples(queries=queries, annotations=annotations)

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers

    # endregion


class ContextFreeSameType(ContextFree):
    relevance_level = 1

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers


class ContextDependent(ModelingTask):
    relevance_level = 1

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers


class ContextDependentSameType(ModelingTask):
    relevance_level = 1

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers


class FullHybrid(ModelingTask):
    relevance_level = 2

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
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


class Hybrid(ModelingTask):
    relevance_level = 1

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_same_tuple(annotations=annotations, sample_queries=sample_queries, queries=queries)
        for answer in answers:
            labelled_answers[answer] = 1

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 2

        return labelled_answers


class HybridSameType(ModelingTask):
    relevance_level = 1

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_same_type(annotations=annotations, sample_queries=sample_queries, queries=queries)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_same_tuple(annotations=annotations, sample_queries=sample_queries, queries=queries)
        for answer in answers:
            labelled_answers[answer] = 1

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 2

        return labelled_answers
