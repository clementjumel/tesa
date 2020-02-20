from modeling.utils import rank

from numpy import mean
from collections import defaultdict
from string import punctuation as str_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm_notebook as tqdm
from gensim.models import KeyedVectors
import torch


# region Base Model

class BaseModel:
    # region Class Initialization

    def __init__(self, score, k):
        """
        Initializes an instance of Base Model.

        Args:
            score: utils.score, score to use.
            k: int, number of ranks to take into account.
        """

        self.score = score
        self.k = k

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # endregion

    # region Main methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        pass

    def train(self, train_loader, valid_loader, n_epochs, n_updates, is_regression):
        """
        Train the Model on train_loader and evaluate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            train_losses: list, training losses, each row corresponding to an epoch.
            train_scores: list, training scores, each row corresponding to an epoch.
            valid_losses: list, validation losses, each row corresponding to an epoch.
            valid_scores: list, validation scores, each row corresponding to an epoch.
        """

        print("Training of the model...\n")

        train_losses, train_scores, valid_losses, valid_scores = [], [], [], []

        for epoch in range(n_epochs):

            train_epoch_losses, train_epoch_scores = self.train_epoch(data_loader=train_loader,
                                                                      n_updates=n_updates,
                                                                      is_regression=is_regression)
            valid_epoch_losses, valid_epoch_scores = self.test_epoch(data_loader=valid_loader,
                                                                     n_updates=n_updates,
                                                                     is_regression=is_regression)

            train_losses.append(train_epoch_losses), train_scores.append(train_epoch_scores)
            valid_losses.append(valid_epoch_losses), valid_scores.append(valid_epoch_scores)

            print('Epoch %d/%d: Validation Loss: %.5f Validation Score: %.5f' % (epoch + 1,
                                                                                 n_epochs,
                                                                                 float(mean(valid_epoch_losses)),
                                                                                 float(mean(valid_epoch_scores))))
            print('--------------------------------------------------------------')

        return train_losses, train_scores, valid_losses, valid_scores

    def test(self, test_loader, n_updates, is_regression):
        """
        Evaluate the Model on test_loader.

        Args:
            test_loader: list of (inputs, targets) batches, testing inputs and outputs.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            losses: list, testing losses.
            scores: list, testing scores.
        """

        print("Evaluation of the model...\n")

        losses, scores = self.test_epoch(data_loader=test_loader, n_updates=n_updates, is_regression=is_regression)

        print('Test Loss: %.5f Test Score: %.5f' % (float(mean(losses)), float(mean(scores)))) if losses is not None \
            else print('Test Score: %.5f' % (float(mean(scores))))

        return losses, scores

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            losses: list, training losses for the epoch.
            scores: list, training scores for the epoch.
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
            losses: list, testing losses for the epoch.
            scores: list, testing scores for the epoch.
        """

        pass

    # endregion

    # region Methods get_

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

    # region Other methods

    @staticmethod
    def features(inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            dict, unchanged inputs of the batch.
        """

        return inputs

    # endregion

# endregion


# region Baselines

class Baseline(BaseModel):
    """ Base Baseline. """

    # region Main methods

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            losses: list, training losses for the epoch.
            scores: list, training scores for the epoch.
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
            losses: list, testing losses for the epoch.
            scores: list, testing scores for the epoch.
        """

        scores = []
        running_score, n_running_score = 0., 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            features = self.features(inputs)
            outputs = self.pred(features)

            ranks = rank(outputs)
            score = self.score(ranks, targets, self.k)

            if score is not None:
                running_score += score.data.item()
                n_running_score += 1

            if (batch_idx + 1) % n_updates == 0:
                scores.append(running_score / n_running_score)
                running_score, n_running_score = 0., 0.

        return None, scores

    # endregion

    # region Other methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        return torch.tensor(0)

    # endregion


class RandomBaseline(Baseline):
    """ Baseline with random predictions. """

    # region Other methods

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

    def __init__(self, score, k):
        """
        Initializes an instance of CountsBaseline Model.

        Args:
            score: utils.score, score to use.
            k: int, number of results to take into account.
        """

        super(CountsBaseline, self).__init__(score=score, k=k)

        self.counts = defaultdict(int)

    # endregion

    # region Main methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        self.learn_counts(data_loader=data_loader)

    # endregion

    # region Other methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = [self.counts[choice] if choice in self.counts else 0 for choice in inputs['choices']]

        grades = torch.tensor(grades).type(dtype=torch.float).reshape((-1, 1))
        m = grades.max()
        grades = grades if m == 0 else torch.div(grades, m)

        return grades

    def learn_counts(self, data_loader):
        """
        Learn the answers counts on the data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        print("Learning answers counts...")

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            choices = inputs['choices']

            for i in range(len(choices)):
                self.counts[choices[i]] += targets[i].data.item()

    # endregion


class SummariesCountBaseline(Baseline):
    """ NLP Baseline based on the count of words from choice that are in summaries. """

    # region Other methods

    def pred(self, inputs):
        """
        Predicts the outputs from the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words, summaries_words = self.get_choices_words(inputs), self.get_summaries_words(inputs)

        grades = [sum([len([word for summary_words in summaries_words
                            for word in summary_words if word in choices_words[i]])])
                  for i in range(len(inputs['choices']))]

        grades = torch.tensor(grades).type(dtype=torch.float).reshape((-1, 1))
        m = grades.max()
        grades = grades if m == 0 else torch.div(grades, m)

        return grades

    # endregion


class SummariesOverlapBaseline(Baseline):
    """ NLP Baseline based on the count of words from choice that are in the overlap of the summaries. """

    # region Other methods

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

        grades = torch.tensor(grades).type(dtype=torch.float).reshape((-1, 1))
        m = grades.max()
        grades = grades if m == 0 else torch.div(grades, m)

        return grades

    # endregion

# endregion


# region ML Models

class MLModel(BaseModel):
    # region Class initialization

    def __init__(self, net, optimizer, loss, score, k):
        """
        Initializes an instance of the ML Model.

        Args:
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            loss: torch.nn.Loss, loss to use.
            score: utils.score, score to use.
            k: int, number of results to take into account.
        """

        super(MLModel, self).__init__(score=score, k=k)

        self.net = net
        self.optimizer = optimizer
        self.loss = loss

    # endregion

    # region Main methods

    def train_epoch(self, data_loader, n_updates, is_regression):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            losses: list, training losses for the epoch.
            scores: list, training scores for the epoch.
        """

        losses, scores = [], []
        running_loss, running_score, n_running_score = 0., 0., 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            features = self.features(inputs)

            self.optimizer.zero_grad()
            outputs = self.net(features)

            loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
            loss = self.loss(outputs, loss_targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()

            ranks = rank(outputs.detach())
            score = self.score(ranks, targets, self.k)

            if score is not None:
                running_score += score.data.item()
                n_running_score += 1

            if (batch_idx + 1) % n_updates == 0:
                losses.append(running_loss / n_updates), scores.append(running_score / n_running_score)
                running_loss, running_score, n_running_score = 0., 0., 0.

        return losses, scores

    def test_epoch(self, data_loader, n_updates, is_regression):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            losses: list, testing losses for the epoch.
            scores: list, testing scores for the epoch.
        """

        self.net.eval()

        losses, scores = [], []
        running_loss, running_score, n_running_score = 0., 0., 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            features = self.features(inputs)

            outputs = self.net(features)

            loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
            loss = self.loss(outputs.detach(), loss_targets)

            running_loss += loss.data.item()

            ranks = rank(outputs.detach())
            score = self.score(ranks, targets, self.k)

            if score is not None:
                running_score += score.data.item()
                n_running_score += 1

            if (batch_idx + 1) % n_updates == 0:
                losses.append(running_loss / n_updates), scores.append(running_score / n_running_score)
                running_loss, running_score, n_running_score = 0., 0., 0.

        self.net.train()

        return losses, scores

    # endregion


class BOWModel(MLModel):
    # region Class initialization

    def __init__(self, min_vocab_frequency, net, optimizer, loss, score, k):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            min_vocab_frequency: int, minimum frequency for a word to be taken into account in the BOW.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            loss: torch.nn.Loss, loss to use.
            score: utils.score, score to use.
            k: int, number of results to take into account.
        """

        super(BOWModel, self).__init__(net=net, optimizer=optimizer, loss=loss, score=score, k=k)

        self.min_vocab_frequency = min_vocab_frequency

        self.choice_to_idx = dict()
        self.context_to_idx = dict()

    # endregion

    # region Main methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        self.learn_vocabulary(data_loader)

    # endregion

    # region Other methods

    def learn_vocabulary(self, data_loader):
        """
        Learns the vocabulary from data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        print("Learning the vocabulary...")

        context_counts = defaultdict(int)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            for choice in inputs['choices']:
                if choice not in self.choice_to_idx:
                    self.choice_to_idx[choice] = len(self.choice_to_idx)

            other_words = self.get_other_words(inputs)

            for word in other_words:
                context_counts[word] += 1

                if context_counts[word] >= self.min_vocab_frequency and word not in self.context_to_idx:
                    self.context_to_idx[word] = len(self.context_to_idx)

        print("Input size: %d" % (len(self.choice_to_idx) + len(self.context_to_idx) + 2))

    @staticmethod
    def to_idx(word, vocabulary_dict):
        """
        Returns the index of a choice or a context word in the corresponding vocabulary dictionary.

        Args:
            word: str, choice or context word to save.
            vocabulary_dict: dict, corresponding vocabulary.

        Returns:
            int, index of the word in the dictionary.
        """

        if word in vocabulary_dict:
            return vocabulary_dict[word]

        else:
            return len(vocabulary_dict)

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, features of the inputs.
        """

        choices = inputs['choices']

        n_lines = len(choices)
        n_choices, n_context = len(self.choice_to_idx) + 1, len(self.context_to_idx) + 1

        choices_one_hot = torch.zeros(size=(n_lines, n_choices), dtype=torch.float)
        context_bow = torch.zeros(n_context, dtype=torch.float)

        for i in range(n_lines):
            j = self.to_idx(choices[i], self.choice_to_idx)
            choices_one_hot[i, j] += 1.

        context_idxs = [self.to_idx(word, self.context_to_idx) for word in self.get_context_words(inputs)]
        for j in context_idxs:
            context_bow[j] += 1.

        context_bow = torch.stack([context_bow for _ in range(n_lines)])

        features = torch.cat((choices_one_hot, context_bow), dim=1)

        return features

    # endregion


class EmbeddingModel(MLModel):
    # region Class initialization

    def __init__(self, net, optimizer, loss, score, k):
        """
        Initializes an instance of the Embedding Model.

        Args:
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            loss: torch.nn.Loss, loss to use.
            score: utils.score, score to use.
            k: int, number of results to take into account.
        """

        super(EmbeddingModel, self).__init__(net=net, optimizer=optimizer, loss=loss, score=score, k=k)

        self.general_embedding = None
        self.entity_embedding = None
        self.general_embedding_dim = None
        self.entity_embedding_dim = None

        self.initialize_word2vec_embedding()

        print("Input dimension: %d" % (self.general_embedding_dim + self.general_embedding_dim))

    # endregion

    # region Other methods

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        choices_embedding = torch.stack([self.get_average_embedding(words=words, is_entity=False)
                                         for words in self.get_choices_words(inputs)])

        other_embedding = self.get_average_embedding(words=self.get_other_words(inputs), is_entity=False)
        other_embedding = other_embedding.expand((n, -1))

        features = torch.cat((choices_embedding, other_embedding), dim=1)

        return features

    def initialize_word2vec_embedding(self):
        """ Initializes the Word2Vec embedding of dimension 300. """

        print("Initializing the Word2Vec embedding...")

        self.general_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/' +
                                                                         'GoogleNews-vectors-negative300.bin',
                                                                   binary=True)
        key = list(self.general_embedding.vocab.keys())[0]
        self.general_embedding_dim = len(self.general_embedding[key])

    def initialize_freebase_embedding(self):
        """ Initializes the freebase embedding of dimension 1000. """

        print("Initializing the FreeBase embedding...")

        self.entity_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/' +
                                                                        'freebase-vectors-skipgram1000-en.bin',
                                                                  binary=True)
        key = list(self.entity_embedding.vocab.keys())[0]
        self.entity_embedding_dim = len(self.entity_embedding[key])

    def get_word_embedding(self, word, is_entity):
        """
        Returns the general or entity embedding of a word in a line Tensor.

        Args:
            word: str, word to embed.
            is_entity: bool, whether to use the entity embedding or the general one.

        Returns:
            torch.Tensor, embedding of word.
        """

        embedding = self.general_embedding if not is_entity else self.entity_embedding
        word = word if not is_entity else '/en/' + word

        return torch.tensor(embedding[word]) if word in embedding.vocab else None

    def get_average_embedding(self, words, is_entity):
        """
        Returns the average general or entity embedding of words in a line Tensor.

        Args:
            words: list, words to embed.
            is_entity: bool, whether to use the entity embedding or the general one.

        Returns:
            torch.Tensor, average embedding of the words.
        """

        embeddings = [self.get_word_embedding(word=word, is_entity=is_entity) for word in words]
        embeddings = [embedding for embedding in embeddings if embedding is not None]

        if embeddings:
            return torch.stack(embeddings).mean(dim=0)
        else:
            return torch.zeros(self.general_embedding_dim) if not is_entity else torch.zeros(self.entity_embedding_dim)

    def get_entity_embedding(self, words):
        """
        Returns the embedding for several entity words as a line Tensor.

        Args:
            words: list, words to embed.

        Returns:
            torch.Tensor, embedding of the words
        """

        s = '/en/' + '_'.join(words)

        if s in self.entity_embedding.vocab:
            return torch.tensor(self.entity_embedding[s])

        else:
            return self.get_average_embedding(words=words, is_entity=True)

    # endregion

# endregion
