from database_creation.database import Database

from numpy import split, concatenate, asarray
from numpy.random import shuffle


class Pipeline:
    # region Class initialization

    def __init__(self, use_k_fold=False):
        """
        Initializes an instance of Pipeline.

        Args:
            use_k_fold: bool, whether or not to use k-fold cross validation.
        """

        self.use_k_fold = use_k_fold

        self.train_loader = None
        self.valid_loader = None
        self.k_fold_loader = None
        self.test_loader = None

    # endregion

    # region Main methods

    def process_data(self, batch_size=32, drop_last=True, test_proportion=0.25, valid_proportion=0.25, k=10):
        """
        Process the pipeline to generate learnable data loaders.

        Args:
            batch_size: int, number of samples in each batch.
            drop_last: bool, whether or not to delete the last batch if incomplete.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            k: int, number of folds to create.
        """

        raw_data = self.get_raw_data()

        if not self.use_k_fold:
            train_set, valid_set, test_set = self.get_train_valid_split(raw_data=raw_data,
                                                                        valid_proportion=valid_proportion,
                                                                        test_proportion=test_proportion)

            self.train_loader = self.get_loader(data=train_set, batch_size=batch_size, drop_last=drop_last)
            self.valid_loader = valid_set
            self.test_loader = test_set

        else:
            train_sets, valid_sets, test_set, complete_train_set = \
                self.get_k_fold_split(raw_data=raw_data, k=k, test_proportion=test_proportion)

            self.k_fold_loader = [(self.get_loader(data=train_sets[i], batch_size=batch_size, drop_last=drop_last),
                                   valid_sets[i]) for i in range(len(train_sets))]

            self.train_loader = self.get_loader(data=complete_train_set, batch_size=batch_size, drop_last=drop_last)
            self.valid_loader = []
            self.test_loader = test_set

    def preview_data(self, model, include_valid=False):
        """
        Preview the data for the model.

        Args:
            model: models.Model, model to train.
            include_valid: bool, whether to include the valid loader during the preview.
        """

        data_loader = self.train_loader if not include_valid \
            else concatenate((self.train_loader, self.valid_loader), axis=0)

        model.preview_data(data_loader=data_loader)

    def evaluate_baseline(self, model, n_updates=100):
        """
        Evaluate a baseline model on the training and validation sets.

        Args:
            model: models.Model, model to evaluate.
            n_updates: int, number of batches between the updates.
        """

        model.test(test_loader=self.train_loader, n_updates=n_updates, is_regression=None, is_test=False)
        model.test(test_loader=self.valid_loader, n_updates=n_updates, is_regression=None, is_test=False)

    def train_model(self, model, n_epochs=1, n_updates=100, is_regression=False):
        """
        Train a model on the training set and compute the metrics on the validation sets.

        Args:
            model: models.Model, model to train.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
        """

        if not self.use_k_fold:
            model.train(train_loader=self.train_loader,
                        valid_loader=self.valid_loader,
                        n_epochs=n_epochs,
                        n_updates=n_updates,
                        is_regression=is_regression)

        else:
            for train_loader, valid_loader in self.k_fold_loader:
                model.reset()
                model.train(train_loader=train_loader,
                            valid_loader=valid_loader,
                            n_epochs=n_epochs,
                            n_updates=n_updates,
                            is_regression=is_regression)

    def test_model(self, model, n_updates=100, is_regression=False):
        """
        Evaluate the model on the test set.

        Args:
            model: models.Model, model to test.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
        """

        model.test(test_loader=self.test_loader, n_updates=n_updates, is_regression=is_regression, is_test=True)

    # endregion

    # region Methods get_

    @staticmethod
    def get_raw_data(assignment_threshold=5):
        """
        Returns the raw data using the methods from Database.

        Args:
            int, minimum number a worker has to have worked on the task to be taken into account.

        Returns:
            2d-array, raw sample from the task, each line corresponding to (inputs, targets)
        """

        database = Database()
        raw_data = database.process_task(assignment_threshold=assignment_threshold)

        print("Raw data imported ({} samples).".format(raw_data.shape[0]))

        return raw_data

    @staticmethod
    def get_train_valid_split(raw_data, valid_proportion, test_proportion):
        """
        Returns the data by splitting the raw data into train, valid and test sets.

        Args:
            raw_data: 2d-array, raw sample from the task, each line corresponding to (inputs, targets).
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            test_proportion: float, fraction (between 0 and 1) of the data, to keep in the test set.

        Returns:
            train_set: 2d-array, training samples, each line corresponding to (inputs, targets).
            valid_set: 2d-array, validation samples, each line corresponding to (inputs, targets).
            test_set: 2d-array, testing samples, each line corresponding to (inputs, targets).
        """

        assert valid_proportion is not None and test_proportion is not None

        n = len(raw_data)
        n_valid, n_test = round(valid_proportion * n), round(test_proportion * n)
        n_train = n - n_test - n_valid

        assert n_train >= 0 and n_valid >= 0 and n_test >= 0

        train_set, valid_set, test_set = split(raw_data, [n_train, n_train + n_valid])

        print("Split into train ({}, {}%),".format(train_set.shape[0], round(100 * train_set.shape[0] / n)) +
              " valid ({}, {}%)".format(valid_set.shape[0], round(100 * valid_set.shape[0] / n)) +
              " and test ({}, {}%) sets.".format(test_set.shape[0], round(100 * test_set.shape[0] / n)))

        return train_set, valid_set, test_set

    @staticmethod
    def get_k_fold_split(raw_data, k, test_proportion):
        """
        Returns the data by splitting the raw data into a k-fold cross validation and a test set.

        Args:
            raw_data: 2d-array, raw sample from the task, each line corresponding to (inputs, targets).
            k: int, number of folds to create.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.

        Returns:
            train_sets: list of 2d-arrays, training samples, each line corresponding to (inputs, targets).
            valid_sets: list of 2d-arrays, validation samples, each line corresponding to (inputs, targets).
            test_set: 2d-array, testing samples, each line corresponding to (inputs, targets).
            complete_train_set: 2d-array, all training samples, each line corresponding to (inputs, targets).
        """

        assert k is not None and test_proportion is not None

        n = len(raw_data)
        n_test = round(test_proportion * n)
        n_test += (n - n_test) % k

        test_set, complete_train_set = split(raw_data, [n_test])

        k_splits = split(complete_train_set, k)
        train_sets, valid_sets = [], []

        for i in range(k):
            valid_set = k_splits[i]
            train_set = concatenate([k_splits[j] for j in range(k) if j != i])

            train_sets.append(train_set)
            valid_sets.append(valid_set)

        s = list(set([(train_sets[i].shape, valid_sets[i].shape) for i in range(len(train_sets))]))
        assert len(s) == 1

        print("Split into k-fold cross validation sets (train: {}, {}%,".format(s[0][0][0], round(100*s[0][0][0]/n)) +
              " valid: {}, {}%)".format(s[0][1][0], round(100*s[0][1][0]/n)) +
              " and a test set ({}, {}%).".format(test_set.shape[0], round(100*test_set.shape[0]/n)))

        return train_sets, valid_sets, test_set, complete_train_set

    @staticmethod
    def get_loader(data, batch_size, drop_last):
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
