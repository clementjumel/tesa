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
        self.test_set = None
        self.k_fold = None
        self.valid_set = None

    # endregion

    # region Main methods

    def process_data(self, valid_proportion, test_proportion=None, k=None):
        """
        Process the pipeline to generate learnable data.

        Args:
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in valid_set.
            test_proportion: float, fraction (between 0 and 1) of the data to keep in test_set.
            k: int, number of folds to create.
        """

        self.compute_raw_data()

        if not self.use_k_fold:
            self.compute_train_test(test_proportion=test_proportion, valid_proportion=valid_proportion)
        else:
            self.compute_k_fold(k=k, valid_proportion=valid_proportion)

    def train_model(self, model):
        """
        Train a model on the training set.

        Args:
            model: models.Model, model to train.
        """

        train_scores, test_scores = [], []

        if not self.use_k_fold:
            model.reset()
            train_scores.append(model.train(train_set=self.train_set))
            test_scores.append(model.test(test_set=self.test_set))

        else:
            for train_set, test_set in self.k_fold:
                model.reset()
                train_scores.append(model.train(train_set=train_set))
                test_scores.append(model.test(test_set=test_set))

            model.train(train_set=self.train_set)

            train_scores, test_scores = mean(train_scores, axis=0), mean(test_scores, axis=0)

        return train_scores, test_scores

    def validate_model(self, model):
        """
        Evaluate a model on the valid set.

        Args:
            model: models.Model, model to train.
        """

        validate_score = model.test(test_set=self.valid_set)

        return validate_score

    # endregion

    # region Methods compute_

    def compute_raw_data(self):
        """ Compute the raw data using the methods from Database. """

        database = Database()
        self.raw_data = database.process_task(assignment_threshold=5)

        print("Raw data imported ({} samples).".format(self.raw_data.shape[0]))

    def compute_train_test(self, test_proportion, valid_proportion):
        """
        Compute the data by splitting the raw data into train, test and valid sets.

        Args:
            test_proportion: float, fraction (between 0 and 1) of the data, to keep in test.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in valid.
        """

        assert test_proportion is not None and valid_proportion is not None

        n = len(self.raw_data)
        n_test, n_valid = round(test_proportion * n), round(valid_proportion * n)
        n_train = n - n_test - n_valid

        assert n_train >= 0 and n_test >= 0 and n_valid >= 0

        train_set, test_set, valid_set = split(self.raw_data, [n_train, n_train + n_test])

        train_set = train_set[:, 0], train_set[:, 1]
        test_set = test_set[:, 0], test_set[:, 1]
        valid_set = valid_set[:, 0], valid_set[:, 1]

        self.train_set = train_set
        self.test_set = test_set
        self.valid_set = valid_set

        print("Split into train ({}, {}%),".format(train_set[0].shape[0], round(100 * train_set[0].shape[0] / n)) +
              " test ({}, {}%)".format(test_set[0].shape[0], round(100 * test_set[0].shape[0] / n)) +
              " and valid ({}, {}%) sets.".format(valid_set[0].shape[0], round(100 * valid_set[0].shape[0] / n)))

    def compute_k_fold(self, k, valid_proportion):
        """
        Compute the data by splitting the raw data into train, test and valid sets.

        Args:
            k: int, number of folds to create.
            valid_proportion: float, fraction (between 0 and 1) of the data to keep in valid_set.
        """

        assert k is not None and valid_proportion is not None

        n = len(self.raw_data)
        n_valid = round(valid_proportion * n)
        n_valid += (n - n_valid) % k

        valid_set, train_set_full = split(self.raw_data, [n_valid])

        k_splits = split(train_set_full, k)
        k_fold = []

        for i in range(k):
            test_set = k_splits[i]
            train_set = concatenate([k_splits[j] for j in range(k) if j != i])

            train_set = train_set[:, 0], train_set[:, 1]
            test_set = test_set[:, 0], test_set[:, 1]

            k_fold.append((train_set, test_set))

        s = list(set([(train_set[0].shape, test_set[0].shape) for train_set, test_set in k_fold]))
        assert len(s) == 1

        train_set_full = train_set_full[:, 0], train_set_full[:, 1]
        valid_set = valid_set[:, 0], valid_set[:, 1]

        self.k_fold = k_fold
        self.train_set = train_set_full
        self.valid_set = valid_set

        print("Split into k-fold cross validation sets (train: {}, {}%,".format(s[0][0][0], round(100*s[0][0][0]/n)) +
              " test: {}, {}%)".format(s[0][1][0], round(100*s[0][1][0]/n)) +
              " and a valid set ({}, {}%).".format(valid_set[0].shape[0], round(100*valid_set[0].shape[0]/n)))

    # endregion


def main():
    pass


if __name__ == '__main__':
    main()
