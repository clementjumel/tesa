from modeling.nn import MLP
from modeling.utils import rank, ap_at_k

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

    def __init__(self):
        """ Initializes an instance of Base Model. """

        self.loss = torch.nn.MSELoss()
        self.score = ap_at_k
        self.k = 10

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # endregion

    # region Main methods

    def train(self, train_loader, valid_loader, n_epochs=1, n_updates=50):
        """
        Train the Model on train_loader and evaluate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.

        Returns:
            train_losses: np.array, training losses, averaged between the epochs.
            valid_losses: np.array, validation losses, averaged between the epochs.
            valid_scores: np.array, validation scores, averaged between the epochs.
        """

        print("Training of the model...\n")

        train_losses, valid_losses, valid_scores = [], [], []

        for epoch in range(n_epochs):

            train_epoch_losses = self.train_epoch(data_loader=train_loader, n_updates=n_updates)
            valid_epoch_losses, valid_epoch_scores = self.test_epoch(data_loader=valid_loader, n_updates=n_updates)

            train_losses.append(train_epoch_losses)
            valid_losses.append(valid_epoch_losses), valid_scores.append(valid_epoch_scores)

            print('Epoch %d/%d: Validation Loss: %.3f Validation Score: %.3f' % (epoch + 1,
                                                                                 n_epochs,
                                                                                 float(mean(valid_epoch_losses)),
                                                                                 float(mean(valid_epoch_scores))))
            print('--------------------------------------------------------------')

        train_losses = mean(train_losses, axis=0)
        valid_losses, valid_scores = mean(valid_losses, axis=0), mean(valid_scores, axis=0)

        return train_losses, valid_losses, valid_scores

    def test(self, test_loader, n_updates=50):
        """
        Evaluate the Model on test_loader.

        Args:
            test_loader: list of (inputs, targets) batches, testing inputs and outputs.
            n_updates: int, number of batches between the updates.

        Returns:
            losses: np.array, testing losses averaged between the epochs.
            scores: np.array, testing scores averaged between the epochs.
        """

        print("Evaluation of the model...\n")

        losses, scores = self.test_epoch(data_loader=test_loader, n_updates=n_updates)

        print('Test Loss: %.3f Test Score: %.3f' % (float(mean(losses)), float(mean(scores))))

        return losses, scores

    def train_epoch(self, data_loader, n_updates):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            np.array, training losses for the epoch.
        """

        pass

    def test_epoch(self, data_loader, n_updates):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            losses: np.array, testing losses for the epoch.
            scores: np.array, testing scores for the epoch.
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
            summaries_words.append(self.get_words(s=summary,
                                                  remove_stopwords=remove_stopwords,
                                                  remove_punctuation=remove_punctuation,
                                                  lower=lower,
                                                  lemmatize=lemmatize))

        return summaries_words

    def get_full_context_words(self, inputs):
        """
        Returns the words from the inputs' entities, contexts and summaries as a list.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            list, words of the inputs' full context.
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

    def train_epoch(self, data_loader, n_updates):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            np.array, training losses for the epoch.
        """

        losses = []
        running_loss = 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            features = self.features(inputs)
            outputs = self.pred(features)

            loss = self.loss(outputs, targets)
            running_loss += loss.data.item()

            if batch_idx % n_updates == 0 and batch_idx != 0:
                losses.append(running_loss / n_updates)
                running_loss = 0.

        return losses

    def test_epoch(self, data_loader, n_updates):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            losses: np.array, testing losses for the epoch.
            scores: np.array, testing scores for the epoch.
        """

        losses, scores = [], []
        running_loss, running_score = 0., 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            features = self.features(inputs)
            outputs = self.pred(features)

            loss = self.loss(outputs, targets)
            running_loss += loss.data.item()

            ranks = rank(outputs)
            score = self.score(ranks, targets, self.k)
            running_score += score.data.item()

            if batch_idx % n_updates == 0 and batch_idx != 0:
                losses.append(running_loss / n_updates), scores.append(running_score / n_updates)
                running_loss, running_score = 0., 0.

        return losses, scores

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

        return torch.rand(len(inputs['choices']))

    # endregion


class CountsBaseline(Baseline):
    """ Baseline based on answers' overall frequency. """

    # region Class initialization

    def __init__(self):
        """ Initializes an instance of Model. """

        super(CountsBaseline, self).__init__()

        self.memory = defaultdict(int)

    # endregion

    # region Main methods

    def train(self, train_loader, valid_loader, n_epochs=1, n_updates=50):
        """
        Train the Model on train_loader and evaluate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.

        Returns:
            train_losses: np.array, training losses, averaged between the epochs.
            valid_losses: np.array, validation losses, averaged between the epochs.
            valid_scores: np.array, validation scores, averaged between the epochs.
        """

        self.learn_counts(data_loader=train_loader)

        return super(CountsBaseline, self).train(train_loader=train_loader,
                                                 valid_loader=valid_loader,
                                                 n_epochs=n_epochs,
                                                 n_updates=n_updates)

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

        pred = torch.tensor([self.memory[choice] if choice in self.memory else 0 for choice in inputs['choices']])
        pred = pred.type(dtype=torch.float)

        return pred

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
                self.memory[choices[i]] += targets[i].data.item()

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

        grades = torch.tensor(grades).type(dtype=torch.float)

        return torch.div(grades, grades.max())

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

        grades = torch.tensor(grades).type(dtype=torch.float)

        return torch.div(grades, grades.max())

    # endregion

# endregion


# region ML Models

class MLModel(BaseModel):
    # region Class initialization

    def __init__(self):
        """ Initializes an instance of the ML Model. """

        super(MLModel, self).__init__()

        self.input_dim = None
        self.hidden_dim = 100
        self.output_dim = 1
        self.dropout = 0.2
        self.lr = 1e-4

        self.net = None
        self.optimizer = None

    # endregion

    # region Main methods

    def train_epoch(self, data_loader, n_updates):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            np.array, training losses for the epoch.
        """

        losses = []
        running_loss = 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            features = self.features(inputs)

            self.optimizer.zero_grad()
            outputs = self.net(features).squeeze()

            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()

            if batch_idx % n_updates == 0 and batch_idx != 0:
                losses.append(running_loss / n_updates)
                running_loss = 0.

        return losses

    def test_epoch(self, data_loader, n_updates):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.

        Returns:
            losses: np.array, testing losses for the epoch.
            scores: np.array, testing scores for the epoch.
        """

        self.net.eval()

        losses, scores = [], []
        running_loss, running_score = 0., 0.

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

            features = self.features(inputs)
            outputs = self.net(features).squeeze()

            loss = self.loss(outputs.detach(), targets)
            running_loss += loss.data.item()

            ranks = rank(outputs.detach())
            score = self.score(ranks, targets, self.k)
            running_score += score.data.item()

            if batch_idx % n_updates == 0 and batch_idx != 0:
                losses.append(running_loss / n_updates), scores.append(running_score / n_updates)
                running_loss, running_score = 0., 0.

        self.net.train()

        return losses, scores

    # endregion

    # region Other methods

    def initialize_network(self):
        """ Initializes the mlp network. """

        self.net = MLP(input_dim=self.input_dim,
                       hidden_dim=self.hidden_dim,
                       output_dim=self.output_dim,
                       dropout=self.dropout)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    # endregion


class BOWModel(MLModel):
    # region Class initialization

    def __init__(self):
        """ Initializes an instance of the Bag of words Model. """

        super(BOWModel, self).__init__()

        self.choice_to_idx = dict()
        self.context_to_idx = dict()

        self.min_vocab_frequency = 5

    # endregion

    # region Main methods

    def train(self, train_loader, valid_loader, n_epochs=1, n_updates=50):
        """
        Train the Model on train_loader and evaluate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.

        Returns:
            train_losses: np.array, training losses, averaged between the epochs.
            valid_losses: np.array, validation losses, averaged between the epochs.
            valid_scores: np.array, validation scores, averaged between the epochs.
        """

        self.learn_vocabulary(train_loader)
        self.initialize_network()

        return super(BOWModel, self).train(train_loader, valid_loader, n_epochs, n_updates)

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

            full_context_words = self.get_full_context_words(inputs)

            for word in full_context_words:
                context_counts[word] += 1

                if context_counts[word] >= self.min_vocab_frequency and word not in self.context_to_idx:
                    self.context_to_idx[word] = len(self.context_to_idx)

        self.input_dim = len(self.choice_to_idx) + len(self.context_to_idx) + 2

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

    def __init__(self):
        """ Initializes an instance of the Embedding Model. """

        super(EmbeddingModel, self).__init__()

        self.general_embedding = None
        self.entity_embedding = None
        self.general_embedding_dim = None
        self.entity_embedding_dim = None

        self.initialize_word2vec_embedding()
        self.initialize_freebase_embedding()

        self.input_dim = self.general_embedding_dim + self.entity_embedding_dim + self.general_embedding_dim
        self.initialize_network()

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

        choices_words = self.get_choices_words(inputs)
        choices_embedding = torch.stack([self.get_average_embedding(words=words, is_entity=False)
                                         for words in choices_words])

        entities_words = [word for entity_words in self.get_entities_words(inputs) for word in entity_words]
        entities_embedding = self.get_average_embedding(words=entities_words, is_entity=True)
        entities_embedding = entities_embedding.unsqueeze(dim=0).expand((n, self.entity_embedding_dim))

        context_words = self.get_context_words(inputs)
        summaries_words = [word for summary_words in self.get_summaries_words(inputs) for word in summary_words]
        other_words = context_words + summaries_words
        other_embedding = self.get_average_embedding(words=other_words, is_entity=False)
        other_embedding = other_embedding.unsqueeze(dim=0).expand((n, self.general_embedding_dim))

        features = torch.cat([choices_embedding, entities_embedding, other_embedding], dim=1)

        return features

    def initialize_word2vec_embedding(self):
        """ Initializes the Word2Vec embedding of dimension 300. """

        self.general_embedding_dim = 300
        self.general_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/' +
                                                                         'GoogleNews-vectors-negative300.bin',
                                                                   binary=True)

    # TODO
    def initialize_glove_embedding(self):
        pass

    def initialize_freebase_embedding(self):
        """ Initializes the freebase embedding of dimension 1000. """

        self.entity_embedding_dim = 1000
        self.entity_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/' +
                                                                        'freebase-vectors-skipgram1000-en.bin',
                                                                  binary=True)

    # TODO: change for entities
    def get_word_embedding(self, word, is_entity=False):
        """
        Returns the embedding of a general word or an entity's word in a line Tensor.

        Args:
            word: str, word to embed.
            is_entity: bool, whether to use the entity_embedding or the general one.

        Returns:
            torch.Tensor, embedding of word.
        """

        embedding = self.general_embedding if not is_entity else self.entity_embedding

        return torch.tensor(embedding[word]) if word in embedding.vocab else None

    def get_average_embedding(self, words, is_entity=False):
        """
        Returns the average embedding of general words or entities' words in a line Tensor.

        Args:
            words: list, words to embed.
            is_entity: bool, whether to use the entity_embedding or the general one.

        Returns:
            torch.Tensor, average embedding of the words.
        """

        embeddings = [self.get_word_embedding(word=word, is_entity=is_entity) for word in words]
        average_embedding = torch.stack([embedding for embedding in embeddings if embedding is not None]).mean(dim=0)

        return average_embedding

    # endregion

# endregion
