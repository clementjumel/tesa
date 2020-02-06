from numpy.random import uniform
from numpy import asarray
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from modeling.utils import rank, ap, dcg, ndcg


class Model:
    # region Class Initialization

    def __init__(self, metric='ap', k=10):
        """
        Initializes an instance of Model.

        Args:
            metric: str, define the metric to use.
            k: int, number of ranks to take into account for the metrics, if necessary.
        """

        self.metric = metric
        self.k = k

    # endregion

    # region Other methods

    def score(self, y_pred, y_true):
        """
        Returns the performance evaluation.

        Args:
            y_pred: 1d np.array, labels predicted
            y_true: 1d np.array, true labels.

        Returns:
            float, metric of evaluation of the performance.
        """

        if self.metric == 'ap':
            return ap(y_pred, y_true)

        elif self.metric == 'dcg':
            return dcg(y_pred, y_true, self.k)

        elif self.metric == 'ndcg':
            return ndcg(y_pred, y_true, self.k)

        else:
            raise Exception

    def pred(self, x):
        """
        Performs the prediction for the input x.

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, predictions for x.
        """

        x_grades = self.grade(x)
        y_pred = rank(x_grades)

        return y_pred

    @staticmethod
    def grade(x):
        """
        Computes the grades for the input x (higher means more relevant).

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, grades for x.
        """

        return []

    # endregion


class Baseline(Model):
    # region Class initialization

    def __init__(self, metric='ap', k=10):
        """
        Initializes an instance of Baseline.

        Args:
            metric: str, define the metric to use.
            k: int, number of ranks to take into account for the metrics, if necessary.
        """

        super(Baseline, self).__init__(metric=metric, k=k)

        self.memory = None

        self.reset()

    # endregion

    # region Main methods

    def train(self, train_set):
        """
        Train the Baseline on the data from train_set.

        Args:
            train_set: pair of arrays, training inputs and outputs.

        Returns:
            1d-array, scores obtained during the training.
        """

        scores = []

        x_train, y_train = train_set
        assert len(x_train) == len(y_train)

        for i in range(len(x_train)):
            x, y_true = x_train[i], y_train[i]

            y_pred = self.pred(x)
            scores.append(self.score(y_pred, y_true))

            self.update(x, y_true)

        return asarray(scores)

    def test(self, test_set):
        """
        Test the Baseline to the data from test_set.

        Args:
            test_set: pair of arrays, testing inputs and outputs.

        Returns:
            1d-array, scores obtained during the testing.
        """

        scores = []

        x_test, y_test = test_set
        assert len(x_test) == len(y_test)

        for i in range(len(x_test)):
            x, y_true = x_test[i], y_test[i]

            y_pred = self.pred(x)
            scores.append(self.score(y_pred, y_true))

        return asarray(scores)

    # endregion

    # region Other methods

    def reset(self):
        """ Resets the Baseline. """

        self.memory = defaultdict(int)

    def update(self, x, y):
        """
        Updates the Baseline.

        Args:
            x: dict, x data.
            y: nd-array, y data.
        """

        pass

    # endregion


class RandomBaseline(Baseline):
    """ Baseline with random predictions. """

    # region Other methods

    @staticmethod
    def grade(x):
        """
        Computes the grades for the input x (higher means more relevant).

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, grades for x.
        """

        return uniform(0, 1, len(x['choices']))

    # endregion


class FrequencyBaseline(Baseline):
    """ Baseline based on answers' overall frequency. """

    # region Other methods

    def grade(self, x):
        """
        Computes the grades for the input x (higher means more relevant).

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, grades for x.
        """

        return asarray([self.memory[x['choices'][i]] for i in range(len(x['choices']))])

    def update(self, x, y):
        """
        Updates the Frequency Baseline.

        Args:
            x: dict, x data.
            y: nd-array, y data.
        """

        for i in range(len(x['choices'])):
            self.memory[x['choices'][i]] += y[i]

    # endregion


class NLPBaseline(Baseline):
    """ NLP Baselines. """

    nltk_stopwords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


class SummariesCountBaseline(NLPBaseline):
    """ NLP Baseline based on the count of words from choice that are in summaries. """

    # region Other methods

    def grade(self, x):
        """
        Computes the grades for the input x (higher means more relevant).

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, grades for x.
        """

        x_grades = []

        for i in range(len(x['choices'])):
            grade = 0

            choice_words = [self.lemmatizer.lemmatize(word)
                            for word in x['choices'][i].split() if word not in self.nltk_stopwords]

            for summary in x['summaries']:
                summary_words = [self.lemmatizer.lemmatize(word)
                                 for word in summary.split() if word not in self.nltk_stopwords]

                grade += len([word for word in choice_words if word in summary_words])

            x_grades.append(grade)

        return asarray(x_grades)

    # endregion


class SummariesOverlapBaseline(NLPBaseline):
    """ NLP Baseline based on the count of words from choice that are in the overlap of the summaries. """

    # region Other methods

    def grade(self, x):
        """
        Computes the grades for the input x (higher means more relevant).

        Args:
            x: dict, input for the prediction.

        Returns:
            np.array, grades for x.
        """

        x_grades = []

        for i in range(len(x['choices'])):
            choice_words = set([self.lemmatizer.lemmatize(word)
                                for word in x['choices'][i].split() if word not in self.nltk_stopwords])

            grade = len(choice_words.intersection([set([self.lemmatizer.lemmatize(word)
                                                        for word in summary.split() if word not in self.nltk_stopwords])
                                                   for summary in x['summaries']]))

            x_grades.append(grade)

        return asarray(x_grades)

    # endregion


def main():
    pass


if __name__ == '__main__':
    main()
