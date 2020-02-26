import modeling.utils as utils

from numpy import mean, arange
from numpy.random import shuffle
from collections import defaultdict
from string import punctuation as str_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm_notebook as tqdm
from gensim.models import KeyedVectors
import torch
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt


# region Base Model

class BaseModel:
    """ Base structure. """

    # region Class Initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of Base Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        self.scores_names = scores_names
        self.rank = utils.rank

        self.train_losses, self.valid_losses, self.test_losses = [], [], []
        self.train_scores, self.valid_scores, self.test_scores = defaultdict(list), defaultdict(list), defaultdict(list)

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.pretrained_embedding = None
        self.pretrained_embedding_dim = None

    # endregion

    # region Learning methods

    def train(self, train_loader, valid_loader, n_epochs, n_updates, is_regression):
        """
        Train the Model on train_loader and evaluate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
        """

        print("Training of the model...\n")

        reference = self.scores_names[0]
        train_losses, valid_losses, train_scores, valid_scores = [], [], defaultdict(list), defaultdict(list)

        for epoch in range(n_epochs):
            try:
                shuffle(train_loader), shuffle(valid_loader)

                train_epoch_losses, train_epoch_scores = self.train_epoch(data_loader=train_loader,
                                                                          n_updates=n_updates,
                                                                          is_regression=is_regression)

                valid_epoch_losses, valid_epoch_scores = self.test_epoch(data_loader=valid_loader,
                                                                         n_updates=n_updates,
                                                                         is_regression=is_regression)

                train_losses.append(train_epoch_losses), valid_losses.append(valid_epoch_losses)
                for name in self.scores_names:
                    train_scores[name].append(train_epoch_scores[name])
                    valid_scores[name].append(valid_epoch_scores[name])

                self.intermediate_plot(train_losses=train_losses, valid_losses=valid_losses, train_scores=train_scores,
                                       valid_scores=valid_scores)

                print('Epoch %d/%d: Validation Loss: %.5f Validation Score: %.5f'
                      % (epoch + 1, n_epochs, float(mean(valid_epoch_losses)),
                         float(mean(valid_epoch_scores[reference]))))

                self.update_lr_scheduler()
                print('--------------------------------------------------------------')

            except KeyboardInterrupt:
                print("Keyboard interruption, exiting and saving all results except current epoch...")
                break

        self.train_losses.append(train_losses), self.valid_losses.append(valid_losses)
        for name in self.scores_names:
            self.train_scores[name].append(train_scores[name]), self.valid_scores[name].append(valid_scores[name])

    def test(self, test_loader, n_updates, is_regression, is_test):
        """
        Evaluate the Model on test_loader.

        Args:
            test_loader: list of (inputs, targets) batches, testing inputs and outputs.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
            is_test: bool, whether to save the metrics as validation or testing.
        """

        print("Evaluation of the model...\n")

        reference = self.scores_names[0]

        shuffle(test_loader)

        losses, scores = self.test_epoch(data_loader=test_loader, n_updates=n_updates, is_regression=is_regression)

        print('Test Loss: %.5f Test Score: %.5f' % (float(mean(losses)), float(mean(scores[reference])))) \
            if losses is not None else print('Test Score: %.5f' % (float(mean(scores[reference]))))

        if is_test:
            self.test_losses.append(losses)
            for name in self.scores_names:
                self.test_scores[name].append(scores[name])

        else:
            self.valid_losses.append(losses)
            for name in self.scores_names:
                self.valid_scores[name].append(scores[name])

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        pass

    def test_epoch(self, data_loader, n_updates, is_regression):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        pass

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        pass

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        return torch.tensor(0)

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            dict, unchanged inputs of the batch.
        """

        pass

    def explanation(self, inputs):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            list, information retrieved.
        """

        return ['' for _ in range(len(inputs['choices']))]

    def update_lr_scheduler(self):
        """ Performs a step of the learning rate scheduler if there is one. """

        pass

    def get_scores(self, ranks, targets):
        """
        Returns the scores mentioned in scores_names for the ranks and targets.

        Args:
            ranks: torch.Tensor, ranks predicted for the batch.
            targets: torch.Tensor, true labels for the batch.

        Returns:
            dict, scores of the batch mapped with the scores' names.
        """

        return dict([(name, getattr(utils, name)(ranks, targets)) for name in self.scores_names])

    # endregion

    # region Words methods

    def get_words(self, s, remove_stopwords, remove_punctuation, lower, lemmatize):
        """
        Returns the words from the string s in a list, with various preprocessing.

        Args:
            s: str, words to deal with.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.

        Returns:
            list, preprocessed words of the string.
        """

        if remove_punctuation:
            s = s.translate(str.maketrans('', '', self.punctuation))

        words = s.split()

        if remove_stopwords:
            words = [word for word in words if word.lower() not in self.stopwords]

        if lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        if lower:
            words = [word.lower() for word in words]

        return words

    def get_choices_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=False):
        """
        Returns the words from the inputs' choices as a list of list.

        Args:
            inputs: dict, inputs of the prediction.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.

        Returns:
            list, words lists of the inputs' choices.
        """

        choices_words = []

        for choice in inputs['choices']:
            choices_words.append(self.get_words(s=choice,
                                                remove_stopwords=remove_stopwords,
                                                remove_punctuation=remove_punctuation,
                                                lower=lower,
                                                lemmatize=lemmatize))

        return choices_words

    def get_entities_words(self, inputs, remove_stopwords=False, remove_punctuation=True, lower=True, lemmatize=False):
        """
        Returns the words from the inputs' entities as a list of list.

        Args:
            inputs: dict, inputs of the prediction.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.

        Returns:
            list, words lists of the inputs' entities.
        """

        entities_words = []

        for entity in inputs['entities']:
            entities_words.append(self.get_words(s=entity,
                                                 remove_stopwords=remove_stopwords,
                                                 remove_punctuation=remove_punctuation,
                                                 lower=lower,
                                                 lemmatize=lemmatize))

        return entities_words

    def get_context_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=True):
        """
        Returns the words from the inputs' context as a list.

        Args:
            inputs: dict, inputs of the prediction.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.

        Returns:
            list, words of the inputs' context.
        """

        context_words = self.get_words(s=inputs['context'],
                                       remove_stopwords=remove_stopwords,
                                       remove_punctuation=remove_punctuation,
                                       lower=lower,
                                       lemmatize=lemmatize)

        return context_words

    def get_summaries_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=True):
        """
        Returns the words from the inputs' summaries as a list of list.

        Args:
            inputs: dict, inputs of the prediction.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.

        Returns:
            list, words lists of the inputs' summaries.
        """

        summaries_words = []

        for summary in inputs['summaries']:
            if summary != 'No information found.':
                summaries_words.append(self.get_words(s=summary,
                                                      remove_stopwords=remove_stopwords,
                                                      remove_punctuation=remove_punctuation,
                                                      lower=lower,
                                                      lemmatize=lemmatize))
            else:
                summaries_words.append([])

        return summaries_words

    def get_other_words(self, inputs):
        """
        Returns the "other" words from the inputs', that is the words of the fields that are not "choices" or
        "entities_type_".

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            list, other words of the inputs'.
        """

        entities_words = [word for entity_words in self.get_entities_words(inputs) for word in entity_words]
        context_words = self.get_context_words(inputs)
        summaries_words = [word for summary_words in self.get_summaries_words(inputs) for word in summary_words]

        return entities_words + context_words + summaries_words

    # endregion

    # region BOW & 1-hot encoding methods

    @staticmethod
    def save_vocab_idx(s, vocab):
        """
        Saves the string in the vocabulary dictionary.

        Args:
            s: str, element to retrieve.
            vocab: dict, corresponding vocabulary.
        """

        if s not in vocab:
            vocab[s] = len(vocab)

    @staticmethod
    def get_vocab_idx(s, vocab):
        """
        Returns the index of a string in the corresponding vocabulary dictionary.

        Args:
            s: str, element to retrieve.
            vocab: dict, corresponding vocabulary.

        Returns:
            int, index of the word in the dictionary.
        """

        return vocab[s] if s in vocab else len(vocab)

    def get_bow(self, strings, vocab):
        """
        Returns the bow of the items, given the vocabulary, as a torch.Tensor.

        Args:
            strings: iterable, strings to retrieve.
            vocab: dict, corresponding vocabulary.

        Returns:
            torch.tensor, bow of the items in a line tensor.Tensor.
        """

        n = len(vocab) + 1
        bow = torch.zeros(n, dtype=torch.float)

        for s in strings:
            idx = self.get_vocab_idx(s, vocab)
            bow[idx] += 1.

        return bow

    def get_one_hot(self, strings, vocab):
        """
        Returns the one hot encoding of the items, given the vocabulary, as a torch.Tensor.

        Args:
            strings: iterable, strings to retrieve.
            vocab: dict, corresponding vocabulary.

        Returns:
            torch.tensor, one hot encoding of the items in a line tensor.Tensor.
        """

        n1, n2 = len(strings), len(vocab)
        encoding = torch.zeros((n1, n2), dtype=torch.float)

        for i, s in enumerate(strings):
            j = self.get_vocab_idx(s, vocab)
            encoding[i, j] += 1.

        return encoding

    # endregion

    # region Embedding methods

    def initialize_word2vec_embedding(self):
        """ Initializes the Word2Vec pretrained embedding of dimension 300. """

        print("Initializing the Word2Vec pretrained embedding...")

        self.pretrained_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/' +
                                                                            'GoogleNews-vectors-negative300.bin',
                                                                      binary=True)

        key = list(self.pretrained_embedding.vocab.keys())[0]
        self.pretrained_embedding_dim = len(self.pretrained_embedding[key])

    def get_word_embedding(self, word):
        """
        Returns the pretrained embedding of a word in a line Tensor.

        Args:
            word: str, word to embed.

        Returns:
            torch.Tensor, embedding of word.
        """

        return torch.tensor(self.pretrained_embedding[word]) if word in self.pretrained_embedding.vocab else None

    def get_average_embedding(self, words):
        """
        Returns the average pretrained embedding of words in a line Tensor.

        Args:
            words: list, words to embed.

        Returns:
            torch.Tensor, average embedding of the words.
        """

        embeddings = [self.get_word_embedding(word=word) for word in words]
        embeddings = [embedding for embedding in embeddings if embedding is not None]

        return torch.stack(embeddings).mean(dim=0) if embeddings \
            else torch.zeros(self.pretrained_embedding_dim, dtype=torch.float)

    def get_embedding_similarity(self, word1, word2):
        """
        Returns the embedding similarity between two words. If the embeddings don't exist, return None.

        Args:
            word1: str, first word to compare.
            word2: str, second word to compare.

        Returns:
            float, similarity between word1 and word2.
        """

        embedding1, embedding2 = self.get_word_embedding(word1), self.get_word_embedding(word2)

        return cosine_similarity(embedding1, embedding2, dim=0).item() \
            if embedding1 is not None and embedding2 is not None else None

    def get_max_embedding_similarity(self, words1, words2):
        """
        Returns the maximal embedding similarity between the words of words1 and words2. If it is not defined,
        returns -1.

        Args:
            words1: list, first words to compare.
            words2: list, second words to compare.

        Returns:
            float, similarity between words1 and words2.
        """

        similarities = [self.get_embedding_similarity(word1, word2) for word1 in words1 for word2 in words2]
        similarities = [similarity for similarity in similarities if similarity is not None]

        return max(similarities) if similarities else -1.

    # endregion

    # region Display methods

    def display_metrics(self, scores_names=None):
        """
        Display the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
        """

        pass

    @staticmethod
    def plot(x1, x2, train_losses, valid_losses, train_scores, valid_scores, scores_names,
             display_training_scores):
        """
        Plot a single figure for the corresponding data.

        Args:
            x1: list, first x-axis of the plot, for the losses.
            x2: list, second x-axis of the plot, for the scores.
            train_losses: list, training losses to plot.
            valid_losses: list, validation losses to plot.
            train_scores: dict, training scores to plot.
            valid_scores: dict, validation scores to plot.
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
            display_training_scores: bool, whether or not to display the training scores.
        """

        colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:green',
                  'tab:olive', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:pink']
        color_idx = 0

        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('epochs')

        color, color_idx = colors[color_idx], color_idx + 1
        ax1.set_ylabel('loss', color=color)
        ax1.set_yscale('log')
        ax1.plot(x1, train_losses, color=color, label='training loss')
        ax1.scatter(x2, valid_losses, color=color, label='validation loss', s=50, marker='^')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        ax2.set_ylabel('scores')

        for name in scores_names:
            color, color_idx = colors[color_idx], color_idx + 1

            if display_training_scores:
                ax2.plot(x1, train_scores[name], color=color, label='training ' + name)
            ax2.scatter(x2, valid_scores[name], color=color, label='validation ' + name, s=50, marker='^')
            ax2.plot(x2, valid_scores[name], color=color, ls='--')

        fig.legend()

    def intermediate_plot(self, train_losses, valid_losses, train_scores, valid_scores):
        """
        Plot the metrics of the model with the data provided.

        Args:
            train_losses: list, training losses to plot.
            valid_losses: list, validation losses to plot.
            train_scores: dict, training scores to plot.
            valid_scores: dict, validation scores to plot.
        """

        scores_names = self.scores_names
        reference = scores_names[0]

        n_epochs = len(train_scores[reference])
        n_points = len(train_scores[reference][0])

        x1 = list(arange(0, n_epochs, 1. / n_points))
        x2 = list(arange(1, n_epochs + 1))

        train_losses = [x for epoch_losses in train_losses for x in epoch_losses]
        valid_losses = [mean(losses) for losses in valid_losses]
        train_scores = dict([(name, [x for epoch_scores in train_scores[name] for x in epoch_scores])
                             for name in scores_names])
        valid_scores = dict([(name, [mean(scores) for scores in valid_scores[name]])
                             for name in scores_names])

        self.plot(x1=x1, x2=x2, train_losses=train_losses, valid_losses=valid_losses, train_scores=train_scores,
                  valid_scores=valid_scores, scores_names=scores_names, display_training_scores=False)

    def final_plot(self, scores_names=None, align_experiments=False, display_training_scores=False):
        """
        Plot the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
            align_experiments: bool, whether or not to align the data from different experiments.
            display_training_scores: bool, whether or not to display the training scores.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names
        reference = scores_names[0]

        n_experiments = len(self.train_scores[reference])

        total_x1, total_x2, offset = [], [], 0
        total_train_losses, total_valid_losses = [], []
        total_train_scores, total_valid_scores = defaultdict(list), defaultdict(list)

        for i in range(n_experiments):
            n_epochs = len(self.train_scores[reference][i])
            n_points = len(self.train_scores[reference][i][0])

            x1 = list(arange(offset, offset + n_epochs, 1. / n_points))
            x2 = list(arange(offset + 1, offset + n_epochs + 1))
            offset += n_epochs

            train_losses = [x for epoch_losses in self.train_losses[i] for x in epoch_losses]
            valid_losses = [mean(losses) for losses in self.valid_losses[i]]
            train_scores = dict([(name, [x for epoch_scores in self.train_scores[name][i] for x in epoch_scores])
                                 for name in scores_names])
            valid_scores = dict([(name, [mean(scores) for scores in self.valid_scores[name][i]])
                                 for name in scores_names])

            if align_experiments:
                total_x1.extend(x1), total_x2.extend(x2)
                total_train_losses.extend(train_losses), total_valid_losses.extend(valid_losses)
                for name in scores_names:
                    total_train_scores[name].extend(train_scores[name])
                    total_valid_scores[name].extend(valid_scores[name])

            else:
                self.plot(x1=x1, x2=x2, train_losses=train_losses, valid_losses=valid_losses, train_scores=train_scores,
                          valid_scores=valid_scores, scores_names=scores_names,
                          display_training_scores=display_training_scores)

        if align_experiments:
            self.plot(x1=total_x1, x2=total_x2, train_losses=total_train_losses, valid_losses=total_valid_losses,
                      train_scores=total_train_scores, valid_scores=total_valid_scores, scores_names=scores_names,
                      display_training_scores=display_training_scores)

    def explain(self, data_loader, scores_names, display_explanations, n_samples, n_answers):
        """
        Explain the model by displaying the samples and the reason of the prediction.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
            display_explanations: bool, whether or not to display the explanations.
            n_samples: int, number of samples to explain.
            n_answers: int, number of best answers to look at.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names

        for batch_idx, (inputs, targets) in enumerate(data_loader[:n_samples]):
            outputs, explanations = self.pred(inputs), self.explanation(inputs)

            ranks = self.rank(outputs)
            scores = self.get_scores(ranks, targets)

            print('\nEntities (%s): %s' % (inputs['entities_type_'], ',  '.join(inputs['entities'])))
            print("Scores of the batch:", ', '.join(['%s: %.5f' % (name, scores[name].item())
                                                     for name in scores if name in scores_names]))

            best_answers = [(ranks[i],
                             inputs['choices'][i],
                             outputs[i].item(),
                             explanations[i])
                            for i in range(len(inputs['choices']))]

            best_answers = sorted(best_answers)[:n_answers]

            for rank, choice, output, explanation in best_answers:
                if isinstance(output, int):
                    print('%d: %s (%d)' % (rank, choice, output))
                else:
                    print('%d: %s (%.3f)' % (rank, choice, output))
                print('   ' + explanation) if display_explanations and explanation else None

    # endregion

# endregion


# region Baselines

class Baseline(BaseModel):
    """ Base structure for the Baselines. """

    # region Learning methods

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        raise Exception("A baseline cannot be trained.")

    def test_epoch(self, data_loader, n_updates, is_regression):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        epoch_scores = defaultdict(list)
        running_scores, n_running_scores = defaultdict(float), defaultdict(float)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = self.pred(inputs)

            ranks = self.rank(outputs)
            scores = self.get_scores(ranks, targets)

            for name, score in scores.items():
                if score is not None:
                    running_scores[name] += score.data.item()
                    n_running_scores[name] += 1.

            if (batch_idx + 1) % n_updates == 0:
                for name in self.scores_names:
                    s = running_scores[name] / n_running_scores[name] if n_running_scores[name] != 0. else 0.
                    epoch_scores[name].append(s)

                running_scores, n_running_scores = defaultdict(float), defaultdict(float)

        return None, epoch_scores

    # endregion

    # region Display methods

    def display_metrics(self, scores_names=None):
        """
        Display the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names

        train_scores = dict([(name, mean(self.valid_scores[name][0])) for name in scores_names])
        valid_scores = dict([(name, mean(self.valid_scores[name][1])) for name in scores_names])

        for name in scores_names:
            print('%s: training set %.5f validation set %.5f' % (name,
                                                                 float(train_scores[name]),
                                                                 float(valid_scores[name])))

    # endregion


class RandomBaseline(Baseline):
    """ Baseline with random predictions. """

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = torch.rand(len(inputs['choices'])).reshape((-1, 1))

        return grades

    # endregion


class CountsBaseline(Baseline):
    """ Baseline based on answers' overall frequency. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of CountsBaseline Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(CountsBaseline, self).__init__(scores_names=scores_names)

        self.counts = defaultdict(int)

    # endregion

    # region Learning methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        print("Learning answers counts...")

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            choices = inputs['choices']

            for i in range(len(choices)):
                self.counts[choices[i]] += targets[i].data.item()

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = [self.counts[choice] if choice in self.counts else 0 for choice in inputs['choices']]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion


class SummariesCountBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in one of the summaries. """

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words, summaries_words = self.get_choices_words(inputs), self.get_summaries_words(inputs)

        grades = [len([word for summary_words in summaries_words for word in summary_words if word in choices_words[i]])
                  for i in range(len(inputs['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, inputs):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words, summaries_words = self.get_choices_words(inputs), self.get_summaries_words(inputs)

        words = [', '.join([word for summary_words in summaries_words for word in summary_words
                            if word in choices_words[i]])
                 for i in range(len(inputs['choices']))]

        return words

    # endregion


class SummariesSoftOverlapBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the "soft" overlap of the summaries. """

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = set([word for summary_words in self.get_summaries_words(inputs) for word in summary_words])

        grades = [len(choices_words[i].intersection(summaries_words))
                  for i in range(len(inputs['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, inputs):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = set([word for summary_words in self.get_summaries_words(inputs) for word in summary_words])

        words = [', '.join(choices_words[i].intersection(summaries_words)) for i in range(len(inputs['choices']))]

        return words

    # endregion


class SummariesHardOverlapBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the "hard" overlap of the summaries. """

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(inputs)]
        summaries_words = set.intersection(*summaries_words)

        grades = [len(choices_words[i].intersection(summaries_words)) for i in range(len(inputs['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, inputs):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(inputs)]
        summaries_words = set.intersection(*summaries_words)

        words = [', '.join(choices_words[i].intersection(summaries_words)) for i in range(len(inputs['choices']))]

        return words

    # endregion


class ClosestAverageEmbedding(Baseline):
    """ Baseline with predictions based on the average embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of ClosestEmbedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(ClosestAverageEmbedding, self).__init__(scores_names=scores_names)

        self.initialize_word2vec_embedding()

    # endregion

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_embedding = torch.stack([self.get_average_embedding(words) for words in self.get_choices_words(inputs)])
        other_embedding = self.get_average_embedding(self.get_other_words(inputs)).reshape((1, -1))

        grades = cosine_similarity(choices_embedding, other_embedding, dim=1).reshape((-1, 1))

        return grades

    # endregion


class ClosestHardOverlapEmbedding(Baseline):
    """ Baseline with predictions based on the "hard" overlap embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of ClosestEmbedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(ClosestHardOverlapEmbedding, self).__init__(scores_names=scores_names)

        self.initialize_word2vec_embedding()

    # endregion

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(inputs)]
        summaries_words = set.intersection(*summaries_words)

        grades = [self.get_max_embedding_similarity(choices_words[i], summaries_words)
                  for i in range(len(inputs['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion


class ClosestSoftOverlapEmbedding(Baseline):
    """ Baseline with predictions based on the "soft" overlap embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of ClosestEmbedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(ClosestSoftOverlapEmbedding, self).__init__(scores_names=scores_names)

        self.initialize_word2vec_embedding()

    # endregion

    # region Learning methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(inputs)]
        summaries_words = set([word for summary_words in self.get_summaries_words(inputs) for word in summary_words])

        grades = [self.get_max_embedding_similarity(choices_words[i], summaries_words)
                  for i in range(len(inputs['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion

# endregion


# region ML Models

class MLModel(BaseModel):
    """ Base structure for the models. """

    # region Class initialization

    def __init__(self, net, optimizer, lr_scheduler, loss, scores_names):
        """
        Initializes an instance of the ML Model.

        Args:
            net: nn.Module, neural net to train.
            optimizer: torch.optim.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(MLModel, self).__init__(scores_names=scores_names)

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss

    # endregion

    # region Learning methods

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        epoch_losses, epoch_scores = [], defaultdict(list)
        running_loss, running_scores, n_running_scores = 0., defaultdict(float), defaultdict(float)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            features = self.features(inputs)

            self.optimizer.zero_grad()
            outputs = self.net(features)

            loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
            loss = self.loss(outputs, loss_targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()

            ranks = self.rank(outputs.detach())
            scores = self.get_scores(ranks, targets)

            for name, score in scores.items():
                if score is not None:
                    running_scores[name] += score.data.item()
                    n_running_scores[name] += 1.

            if (batch_idx + 1) % n_updates == 0:
                epoch_losses.append(running_loss / n_updates)

                for name in self.scores_names:
                    s = running_scores[name] / n_running_scores[name] if n_running_scores[name] != 0. else 0.
                    epoch_scores[name].append(s)

                running_loss, running_scores, n_running_scores = 0., defaultdict(float), defaultdict(float)

        return epoch_losses, epoch_scores

    def test_epoch(self, data_loader, n_updates, is_regression):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        self.net.eval()

        epoch_losses, epoch_scores = [], defaultdict(list)
        running_loss, running_scores, n_running_scores = 0., defaultdict(float), defaultdict(float)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            features = self.features(inputs)

            outputs = self.net(features)

            loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
            loss = self.loss(outputs.detach(), loss_targets)

            running_loss += loss.data.item()

            ranks = self.rank(outputs.detach())
            scores = self.get_scores(ranks, targets)

            for name, score in scores.items():
                if score is not None:
                    running_scores[name] += score.data.item()
                    n_running_scores[name] += 1.

            if (batch_idx + 1) % n_updates == 0:
                epoch_losses.append(running_loss / n_updates)

                for name in self.scores_names:
                    s = running_scores[name] / n_running_scores[name] if n_running_scores[name] != 0. else 0.
                    epoch_scores[name].append(s)

                running_loss, running_scores, n_running_scores = 0., defaultdict(float), defaultdict(float)

        self.net.train()

        return epoch_losses, epoch_scores

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        self.net.eval()

        features = self.features(inputs)
        grades = self.net(features)

        self.net.train()

        return grades

    def update_lr_scheduler(self):
        """ Performs a step of the learning rate scheduler if there is one. """

        self.lr_scheduler.step()

        print("Learning rate decreasing to %s" % (self.optimizer.param_groups[0]['lr']))

    # endregion

    # region Display methods

    def display_metrics(self, scores_names=None):
        """
        Display the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names

        train_scores = dict([(name, mean(self.train_scores[name][-1])) for name in scores_names])
        valid_scores = dict([(name, mean(self.valid_scores[name][-1])) for name in scores_names])

        for name in scores_names:
            print('%s: training set %.5f validation set %.5f' % (name,
                                                                 float(train_scores[name]),
                                                                 float(valid_scores[name])))

    # endregion


class HalfBOWModel(MLModel):
    """ Model that uses a 1-hot encoding for the choices and a BOW for the other words. """

    # region Class initialization

    def __init__(self, vocab_frequency_range, net, optimizer, lr_scheduler, loss, scores_names):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(HalfBOWModel, self).__init__(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                                           scores_names=scores_names)

        self.vocab_frequency_range = vocab_frequency_range

        self.choice_to_idx = dict()
        self.word_to_idx = dict()
        self.word_counts = defaultdict(int)

    # endregion

    # region Learning methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        print("Learning the vocabulary...")

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            for choice in inputs['choices']:
                self.save_vocab_idx(choice, self.choice_to_idx)

            for word in self.get_other_words(inputs):
                self.word_counts[word] += 1

        for word, count in self.word_counts.items():
            if self.vocab_frequency_range[0] <= count <= self.vocab_frequency_range[1]:
                self.save_vocab_idx(word, self.word_to_idx)

        print("Input size: %d" % (len(self.choice_to_idx) + len(self.word_to_idx) + 1))

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        one_hot = self.get_one_hot(strings=inputs['choices'], vocab=self.choice_to_idx)

        bow = self.get_bow(strings=self.get_other_words(inputs), vocab=self.word_to_idx)
        bow = bow.expand((n, -1))

        features = torch.cat((one_hot, bow), dim=1)

        return features

    # endregion


class FullBOWModel(MLModel):
    """ Model that uses a BOW for the choice and the other words. """

    # region Class initialization

    def __init__(self, vocab_frequency_range, net, optimizer, lr_scheduler, loss, scores_names):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(FullBOWModel, self).__init__(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                                           scores_names=scores_names)

        self.vocab_frequency_range = vocab_frequency_range

        self.word_to_idx = dict()
        self.word_counts = defaultdict(int)

    # endregion

    # region Learning methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        print("Learning the vocabulary...")

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            for words in self.get_choices_words(inputs):
                for word in words:
                    self.word_counts[word] += 1

            for word in self.get_other_words(inputs):
                self.word_counts[word] += 1

        for word, count in self.word_counts.items():
            if self.vocab_frequency_range[0] <= count <= self.vocab_frequency_range[1]:
                self.save_vocab_idx(word, self.word_to_idx)

        print("Input size: %d" % (len(self.word_to_idx) + 1))

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, features of the inputs.
        """

        other_words = self.get_other_words(inputs)

        features = [self.get_bow(words + other_words, self.word_to_idx) for words in self.get_choices_words(inputs)]
        features = torch.stack(features)

        return features

    # endregion


class EmbeddingModel(MLModel):
    """ Model that uses an average embedding both for the choice words and the context words. """

    # region Class initialization

    def __init__(self, net, optimizer, lr_scheduler, loss, scores_names):
        """
        Initializes an instance of the Embedding Model.

        Args:
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            scores_names: iterable, names of the scores to use, the first one being monitored.
        """

        super(EmbeddingModel, self).__init__(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                                             scores_names=scores_names)

        self.initialize_word2vec_embedding()

        print("Input dimension: %d" % (2 * self.pretrained_embedding_dim))

    # endregion

    # region Learning methods

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        choices_embedding = torch.stack([self.get_average_embedding(words=words)
                                         for words in self.get_choices_words(inputs)])

        other_embedding = self.get_average_embedding(words=self.get_other_words(inputs))
        other_embedding = other_embedding.expand((n, -1))

        features = torch.cat((choices_embedding, other_embedding), dim=1)

        return features

    # endregion

# endregion
