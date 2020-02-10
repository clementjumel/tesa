from database_creation.database import Database

from numpy import split, concatenate, mean


class Pipeline:
    # region Class initialization

    def __init__(self, use_k_fold=False):
        """
        Initializes an instance of Pipeline.

        Args:
            use_k_fold: bool, whether or not to use k-fold cross validation.
        """

        self.use_k_fold = use_k_fold

        self.raw_data = None

        self.train_set = None
        self.valid_set = None
        self.k_fold = None

        self.test_set = None

    # endregion

    # region Main methods

    def process_data(self, test_proportion, valid_proportion=None, k=None):
        """
        Process the pipeline to generate learnable data.

        Args:
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            k: int, number of folds to create.
        """

        self.compute_raw_data()

        if not self.use_k_fold:
            self.compute_train_valid(valid_proportion=valid_proportion, test_proportion=test_proportion)

        else:
            self.compute_k_fold(k=k, test_proportion=test_proportion)

    def train_model(self, model):
        """
        Train a model on the training set.

        Args:
            model: models.Model, model to train.

        Returns:
            train_scores: list, scores of the model during training.
            valid_scores: list, scores of the model during validation.
        """

        train_scores, valid_scores = [], []

        if not self.use_k_fold:
            train_scores.extend(model.train(data=self.train_set))
            valid_scores.extend(model.test(data=self.valid_set))

        else:
            for train_set, valid_set in self.k_fold:
                model.reset()
                train_scores.append(model.train(data=train_set))
                valid_scores.append(model.test(data=valid_set))

            model.reset()
            model.train(data=self.train_set)

            train_scores, valid_scores = mean(train_scores, axis=0), mean(valid_scores, axis=0)
            print("Mean score on the k-fold, train: {}; valid: {}".format(mean(train_scores), mean(valid_scores)))

        return train_scores, valid_scores

    def test_model(self, model):
        """
        Evaluate a model on the test set.

        Args:
            model: models.Model, model to test.

        Returns:
            list, scores of the model during testing.
        """

        test_scores = model.test(data=self.test_set)

        return test_scores

    # endregion

    # region Methods compute_

    def compute_raw_data(self):
        """ Compute the raw data using the methods from Database. """

        database = Database()
        self.raw_data = database.process_task(assignment_threshold=5)

        print("Raw data imported ({} samples).".format(self.raw_data.shape[0]))

    def compute_train_valid(self, valid_proportion, test_proportion):
        """
        Compute the data by splitting the raw data into train, valid and test sets.

        Args:
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in the valid set.
            test_proportion: float, fraction (between 0 and 1) of the data, to keep in the test set.
        """

        assert valid_proportion is not None and test_proportion is not None

        n = len(self.raw_data)
        n_valid, n_test = round(valid_proportion * n), round(test_proportion * n)
        n_train = n - n_test - n_valid

        assert n_train >= 0 and n_valid >= 0 and n_test >= 0

        train_set, valid_set, test_set = split(self.raw_data, [n_train, n_train + n_valid])

        train_set = train_set[:, 0], train_set[:, 1]
        valid_set = valid_set[:, 0], valid_set[:, 1]
        test_set = test_set[:, 0], test_set[:, 1]

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        print("Split into train ({}, {}%),".format(train_set[0].shape[0], round(100 * train_set[0].shape[0] / n)) +
              " valid ({}, {}%)".format(valid_set[0].shape[0], round(100 * valid_set[0].shape[0] / n)) +
              " and test ({}, {}%) sets.".format(test_set[0].shape[0], round(100 * test_set[0].shape[0] / n)))

    def compute_k_fold(self, k, test_proportion):
        """
        Compute the data by splitting the raw data into a k-fold cross validation and a test set.

        Args:
            k: int, number of folds to create.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in the test set.
        """

        assert k is not None and test_proportion is not None

        n = len(self.raw_data)
        n_test = round(test_proportion * n)
        n_test += (n - n_test) % k

        test_set, train_set_full = split(self.raw_data, [n_test])

        k_splits = split(train_set_full, k)
        k_fold = []

        for i in range(k):
            valid_set = k_splits[i]
            train_set = concatenate([k_splits[j] for j in range(k) if j != i])

            train_set = train_set[:, 0], train_set[:, 1]
            valid_set = valid_set[:, 0], valid_set[:, 1]

            k_fold.append((train_set, valid_set))

        s = list(set([(train_set[0].shape, valid_set[0].shape) for train_set, valid_set in k_fold]))
        assert len(s) == 1

        train_set_full = train_set_full[:, 0], train_set_full[:, 1]
        test_set = test_set[:, 0], test_set[:, 1]

        self.k_fold = k_fold
        self.train_set = train_set_full
        self.test_set = test_set

        print("Split into k-fold cross validation sets (train: {}, {}%,".format(s[0][0][0], round(100*s[0][0][0]/n)) +
              " valid: {}, {}%)".format(s[0][1][0], round(100*s[0][1][0]/n)) +
              " and a test set ({}, {}%).".format(test_set[0].shape[0], round(100*test_set[0].shape[0]/n)))

    # endregion
