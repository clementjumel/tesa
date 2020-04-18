from modeling.ranking_task import RankingTask
from modeling.utils import format_context, format_choices

from collections import defaultdict
from numpy import asarray, split, concatenate
from numpy.random import seed, shuffle
from pickle import dump, load
from re import findall
from csv import writer
from os import makedirs
from os.path import exists


class ModelingTask:
    relevance_level = None

    def __init__(self, ranking_size, batch_size, context_format, choices_format, k_cross_validation, valid_proportion,
                 test_proportion, random_seed, save, silent, results_path, annotation_task_results_path):
        """
        Initializes an instance of the base ModelingTask.

        Args:
            ranking_size: int, number of choices to compute for each ranking task.
            batch_size: int, number of samples in each batch.
            context_format: str, version of the context format to encode the inputs in a string.
            choices_format: str, version of the choices format to encode the choices in strings.
            k_cross_validation: int, number of folds to use in k-fold cross validation (if 0, doesn't use k-fold).
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            random_seed: int, the seed to use for the random processes.
            save: bool, saving option.
            silent: bool, silent option.
            results_path: str, path to the folder to save the modeling task in.
            annotation_task_results_path: str, path to the annotation results folder.
        """

        self.ranking_size = ranking_size
        self.batch_size = batch_size
        self.context_format = context_format
        self.choices_format = choices_format
        self.k_cross_validation = k_cross_validation
        self.valid_proportion = valid_proportion
        self.test_proportion = test_proportion
        self.save = save
        self.silent = silent
        self.results_path = results_path
        self.annotation_task_results_path = annotation_task_results_path

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        seed(random_seed)

    # region Main methods

    def process_data_loaders(self):
        """ Process the data of the annotations to create the data loaders. """

        self.compute_ranking_tasks()

        self.makedirs('')
        self.save_pkl()

    def process_classification_task(self, folder_path):
        """
        Save the task set up for classification for BART finetuning.

        Args:
            folder_path: str, path of the folder to create, starting from <self.results_path>.
        """

        folder_path = folder_path + self.class_name() + self.suffix() + "/classification_task/"

        self.makedirs(folder_path)
        self.save_classification_task(folder_path)

    def process_generation_task(self, folder_path):
        """
        Save the task set up for generation (summarization) for BART finetuning.

        Args:
            folder_path: str, path of the folder to create, starting from <self.results_path>.

        """

        folder_path = folder_path + self.class_name() + self.suffix() + "/generation_task/"

        self.makedirs(folder_path)
        self.save_generation_task(folder_path)

    # endregion

    def compute_ranking_tasks(self):
        """ Compute the RankingTasks of the ModelingTask, and split them into the data_loaders. """

        annotations = load(self.annotation_task_results_path + "annotations/annotations.pkl")
        queries = load(self.annotation_task_results_path + "annotations/queries.pkl")

        self.print("Annotations and queries loaded from %s/annotations/." % self.annotation_task_results_path)

        annotations = self.get_reordered_annotations(queries=queries, annotations=annotations)

        ranking_tasks = []
        for _, sample_annotations in annotations.items():
            sample_queries_ids = sorted(set([annotation.id_ for annotation in sample_annotations]))
            sample_queries = [queries[query_id_] for query_id_ in sample_queries_ids]

            labelled_answers = self.get_labelled_answers(sample_queries=sample_queries,
                                                         sample_annotations=sample_annotations,
                                                         queries=queries,
                                                         annotations=annotations)

            ranking_tasks.append(RankingTask(queries=sample_queries,
                                             labelled_answers=labelled_answers,
                                             ranking_size=self.ranking_size,
                                             batch_size=self.batch_size))

        shuffle(ranking_tasks)

        n = len(ranking_tasks)
        if not n:
            raise Exception("No data imported.")

        k = self.k_cross_validation
        n_test = round(self.test_proportion * n)

        if not k:
            n_valid = round(self.valid_proportion * n)
            n_train = n - n_test - n_valid

            assert 0 <= n_train <= n and 0 <= n_valid <= n and 0 <= n_test <= n
            train_set, valid_set, test_set = split(asarray(ranking_tasks), [n_train, n_train + n_valid])
            assert n_train == train_set.shape[0] and n_valid == valid_set.shape[0] and n_test == test_set.shape[0]

            self.train_loader = [ranking_task.to_loader() for ranking_task in train_set]
            self.valid_loader = [ranking_task.to_loader() for ranking_task in valid_set]
            self.test_loader = [ranking_task.to_loader() for ranking_task in test_set]

            train_loader, valid_loader, test_loader = self.train_loader, self.valid_loader, self.test_loader
            self.print("Data loaders computed:")

        else:
            n_test += (n - n_test) % k

            test_set, cross_validation_set = split(asarray(ranking_tasks), [n_test])
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

            self.train_loader = [[ranking_task.to_loader() for ranking_task in train_set] for train_set in train_sets]
            self.valid_loader = [[ranking_task.to_loader() for ranking_task in valid_set] for valid_set in valid_sets]
            self.test_loader = [ranking_task.to_loader() for ranking_task in test_set]

            train_loader, valid_loader, test_loader = self.train_loader[0], self.valid_loader[0], self.test_loader

            self.print("Data loaders for %i-fold cross validation computed:" % k)

        m_train = sum([len(ranking_task) for ranking_task in train_loader])
        m_valid = sum([len(ranking_task) for ranking_task in valid_loader])
        m_test = sum([len(ranking_task) for ranking_task in test_loader])

        self.print("   train: %i ranking tasks (%i%%), %i batches,\n" % (n_train, 100 * n_train / n, m_train),
                   "  valid: %i ranking tasks (%i%%), %i batches,\n" % (n_valid, 100 * n_valid / n, m_valid),
                   "  test: %i ranking tasks (%i%%), %i batches.\n" % (n_test, 100 * n_test / n, m_test))

    # region Methods get_

    @staticmethod
    def get_reordered_annotations(queries, annotations):
        """
        Rework the annotations for the specificity of the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            dict of list of Annotations, Annotations from the MT workers.
        """

        return annotations

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

    def get_classification_rows(self, ranking_task):
        """
        Returns a list of rows [sentence1, sentence2, label] for the ranking_task.

        Args:
            ranking_task: list of (inputs, targets) batches.
        """

        rows = []
        sentence1 = format_context(ranking_task, context_format=self.context_format)

        for inputs, targets in ranking_task:
            for choice, target in zip(inputs['choices'], targets):
                sentence2 = choice
                label = "aggregation" if target else "not_aggregation"

                rows.append([sentence1, sentence2, label])

        return rows

    def get_generation_rows(self, ranking_task):
        """
        Return two lists, for the sources and the targets, respectively, of the ranking_task.

        Args:
            ranking_task: list of (inputs, targets) batches.
        """

        source_rows, target_rows = [], []
        source = format_context(ranking_task, context_format=self.context_format)
        choices = format_choices(ranking_task, choices_format=self.choices_format)

        for choice in choices:
            source_rows.append(source), target_rows.append(choice)

        return source_rows, target_rows

    # endregion

    # region Other methods

    def class_name(self):
        """ Returns the standardized name of the class. """

        return "".join([word.lower() for word in findall(r'[A-Z][^A-Z]*', self.__class__.__name__)])

    def suffix(self):
        """ Returns the standard suffix of a file_name as a string. """

        train_proportion = ("%.2f" % (1 - self.valid_proportion - self.test_proportion)).split(".")[1]
        valid_proportion = ("%.2f" % self.valid_proportion).split(".")[1]
        test_proportion = ("%.2f" % self.test_proportion).split(".")[1]
        suffix = "_" + "-".join([train_proportion, valid_proportion, test_proportion])

        suffix += "_rs" + str(self.ranking_size) if self.ranking_size is not None else ""
        suffix += "_bs" + str(self.batch_size)
        suffix += "_co-" + self.context_format
        suffix += "_ch-" + self.choices_format
        suffix += "_cv" if self.k_cross_validation else ""

        return suffix

    def makedirs(self, folder_name):
        """
        If necessary, creates folders to save outputs, starting from self.results_path.

        Args:
            folder_name: str, path of the folder to create.
        """

        folder_name = self.results_path + folder_name

        if self.save:
            if not exists(folder_name):
                makedirs(folder_name)
                self.print("Creating folder(s) %s." % folder_name)

    def save_pkl(self):
        """ Save the Task using pickle in self.results_path. """

        file_name = self.results_path + self.class_name() + self.suffix() + '.pkl'

        if self.save:
            with open(file_name, 'wb') as file:
                dump(obj=self, file=file, protocol=-1)

            self.print("Task saved at %s.\n" % file_name)

        else:
            self.print("Not saving %s (not in save mode).\n" % file_name)

    def save_classification_task(self, path):
        """
        Saves the task as a classification task.

        Args:
            path: str, full path to the folder to save in.
        """

        data_loader_names = ["train", "valid", "test"]
        file_names = [path + file_name + ".tsv" for file_name in ["train", "dev", "test"]]

        for i, data_loader_name in enumerate(data_loader_names):
            data_loader = getattr(self, data_loader_name + "_loader")
            file_name = file_names[i]

            all_rows = []
            for ranking_task in data_loader:
                all_rows.extend(self.get_classification_rows(ranking_task))

            shuffle(all_rows)
            all_rows = [[str(j)] + row for j, row in enumerate(all_rows)]

            if self.save:
                with open(file_name, 'wt') as file:
                    tsv_writer = writer(file, delimiter='\t')
                    tsv_writer.writerow(['index', 'sentence1', 'sentence2', 'label'])

                    for row in all_rows:
                        tsv_writer.writerow(row)

                self.print("File %s saved." % file_name)

            else:
                self.print("File %s not saved (not in save mode)." % file_name)

    def save_generation_task(self, path):
        """
        Saves the task as a generation task.

        Args:
            path: str, full path to the folder to save in.
        """

        data_loader_names = ["train", "valid", "test"]
        file_name_pairs = [[path + file_name + suffix for suffix in [".source", ".target"]]
                           for file_name in ["train", "val", "test"]]

        for i, data_loader_name in enumerate(data_loader_names):
            data_loader = getattr(self, data_loader_name + "_loader")
            file_name_pair = file_name_pairs[i]

            all_source_rows, all_target_rows = [], []
            for ranking_task in data_loader:
                source_rows, target_rows = self.get_generation_rows(ranking_task)
                all_source_rows.extend(source_rows), all_target_rows.extend(target_rows)

            shuffle(all_source_rows), shuffle(all_target_rows)

            if self.save:
                with open(file_name_pair[0], 'wt') as source_file, open(file_name_pair[1], 'wt') as target_file:
                    source_file.writelines(all_source_rows), target_file.writelines(all_target_rows)

                self.print("File %s and %s saved." % (file_name_pair[0], file_name_pair[1]))

            else:
                self.print("File %s and %s not saved (not in save mode)." % (file_name_pair[0], file_name_pair[1]))

    def print(self, *args):
        """ Prints only if not in silent mode. """

        if not self.silent:
            print(*args)

    # endregion


class ContextFree(ModelingTask):
    relevance_level = 1

    @staticmethod
    def get_reordered_annotations(queries, annotations):
        new_annotations = defaultdict(list)

        for id_, annotation_list in annotations.items():
            for annotation in annotation_list:
                entities_tuple = tuple(sorted(queries[annotation.id_].entities))
                new_annotations[entities_tuple].append(annotation)

        return new_annotations

    def get_labelled_answers(self, sample_queries, sample_annotations, queries, annotations):
        answers = self.get_answers_all(annotations=annotations)
        labelled_answers = {answer: 0 for answer in answers}

        answers = self.get_answers_sample(sample_annotations=sample_annotations)
        for answer in answers:
            labelled_answers[answer] = 1

        return labelled_answers


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
