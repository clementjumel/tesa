import modeling.utils as utils
from modeling.utils import ranking, dict_mean, dict_append, dict_remove_none

from numpy import mean, arange
from numpy.random import shuffle
from collections import defaultdict
from string import punctuation as str_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm_notebook as tqdm
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# region Base Model

class BaseModel:
    """ Base structure. """

    # region Class Initialization

    punctuation = str_punctuation
    stopwords = set(nltk_stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def __init__(self, scores_names, pretrained_model=None, pretrained_model_dim=None, tokenizer=None):
        """
        Initializes an instance of Base Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        self.scores_names, self.reference = scores_names, scores_names[0]

        self.pretrained_model = pretrained_model
        self.pretrained_model_dim = pretrained_model_dim
        self.tokenizer = tokenizer

        self.train_losses, self.train_scores = [], defaultdict(list)
        self.valid_losses, self.valid_scores = [], defaultdict(list)
        self.test_losses, self.test_scores = [], defaultdict(list)

    # endregion

    # region Training methods

    def train(self, train_loader, valid_loader, n_epochs, n_updates, is_regression):
        """
        Train the Model on train_loader and validate on valid_loader at each epoch.

        Args:
            train_loader: list of (inputs, targets) batches, training inputs and outputs.
            valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
            n_epochs: int, number of epochs to perform.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
        """

        print("Training of the model...\n")

        train_losses, valid_losses, train_scores, valid_scores = [], [], defaultdict(list), defaultdict(list)

        for epoch in range(n_epochs):

            try:
                shuffle(train_loader), shuffle(valid_loader)

                train_epoch_losses, train_epoch_scores = self.train_epoch(data_loader=train_loader,
                                                                          n_updates=n_updates,
                                                                          is_regression=is_regression,
                                                                          epoch=epoch)

                valid_epoch_loss, valid_epoch_score = self.test_epoch(data_loader=valid_loader,
                                                                      is_regression=is_regression)

                self.write_tensorboard(mean(train_epoch_losses), dict_mean(train_epoch_scores), 'epoch train', epoch)
                self.write_tensorboard(valid_epoch_loss, valid_epoch_score, 'epoch valid', epoch)

                train_losses.append(train_epoch_losses), valid_losses.append(valid_epoch_loss)
                dict_append(train_scores, train_epoch_scores), dict_append(valid_scores, valid_epoch_score)

                print('Epoch %d/%d: Validation Loss: %.5f Validation Score: %.5f'
                      % (epoch+1, n_epochs, valid_epoch_loss, valid_epoch_score[self.reference]))

                self.update_lr_scheduler()

                print('--------------------------------------------------------------')

            except KeyboardInterrupt:
                print("Keyboard interruption, exiting and saving all results except current epoch...")
                break

        if train_losses and train_scores and valid_losses and valid_scores:
            self.train_losses.append(train_losses), self.valid_losses.append(valid_losses)
            dict_append(self.train_scores, train_scores), dict_append(self.valid_scores, valid_scores)

    def valid(self, data_loader, is_regression):
        """
        Validate the Model on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            is_regression: bool, whether to use the regression set up for the task.
        """

        print("Validation of the model...\n")

        shuffle(data_loader)

        loss, score = self.test_epoch(data_loader=data_loader, is_regression=is_regression)

        print('Validation Loss: %.5f Validation Score: %.5f' % (loss, score[self.reference])) if loss is not None \
            else print('Validation Score: %.5f' % (score[self.reference]))

        self.valid_losses.append(loss), dict_append(self.valid_scores, score)

    def test(self, data_loader, is_regression):
        """
        Test the Model on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            is_regression: bool, whether to use the regression set up for the task.
        """

        print("Testing of the model...\n")

        shuffle(data_loader)

        loss, score = self.test_epoch(data_loader=data_loader, is_regression=is_regression)

        print('Test Loss: %.5f Test Score: %.5f' % (loss, score[self.reference])) if loss is not None \
            else print('Test Score: %.5f' % (score[self.reference]))

        self.test_losses.append(loss), dict_append(self.test_scores, score)

    def train_epoch(self, data_loader, n_updates, is_regression, epoch):
        """
        Trains the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_updates: int, number of batches between the updates.
            is_regression: bool, whether to use the regression set up for the task.
            epoch: int, epoch number of the training.

        Returns:
            epoch_losses: list, losses for the epoch.
            epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
        """

        n_batches = len(data_loader)

        epoch_losses, epoch_scores = [], defaultdict(list)
        running_loss, running_score = [], defaultdict(list)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=n_batches):
            features = self.features(inputs)

            batch_loss, batch_score = self.train_batch(features, targets, is_regression)

            running_loss.append(batch_loss), dict_append(running_score, batch_score)

            if (batch_idx + 1) % n_updates == 0:
                dict_remove_none(running_score)

                running_loss, running_score = mean(running_loss), dict_mean(running_score)
                epoch_losses.append(running_loss), dict_append(epoch_scores, running_score)

                self.write_tensorboard(running_loss, running_score, 'train', epoch*n_batches+batch_idx)

                running_loss, running_score = [], defaultdict(list)

        return epoch_losses, epoch_scores

    def test_epoch(self, data_loader, is_regression):
        """
        Tests the model for one epoch on data_loader.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            epoch_loss: float, mean loss of the epoch.
            epoch_score: dict, mean score of the epoch mapped with the score's names.
        """

        self.eval_mode()

        n_batches = len(data_loader)

        running_loss, running_score = [], defaultdict(list)

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=n_batches):
            features = self.features(inputs)

            batch_loss, batch_score = self.test_batch(features, targets, is_regression)

            running_loss.append(batch_loss), dict_append(running_score, batch_score)

        running_loss = [loss for loss in running_loss if loss is not None]
        epoch_loss = mean(running_loss) if running_loss else None
        epoch_score = dict_mean(running_score)

        self.train_mode()

        return epoch_loss, epoch_score

    def train_batch(self, features, targets, is_regression):
        """
        Perform the training on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores (float) of the batch of data.
        """

        pass

    def test_batch(self, features, targets, is_regression):
        """
        Perform the test or validation on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores (float) of the batch of data.
        """

        pass

    # endregion

    # region Learning methods

    def preview_data(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
        """

        pass

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the batch.

        Returns:
            dict or torch.Tensor, features of the batch.
        """

        return inputs

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        pass

    def explanation(self, features):
        """
        Explain the model by returning some relevant information from the features as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        return ['' for _ in range(len(features['choices']))]

    def get_score(self, ranks, targets):
        """
        Returns the scores mentioned in self.scores_names for the ranks and targets.

        Args:
            ranks: torch.Tensor, ranks predicted for the batch.
            targets: torch.Tensor, true labels for the batch.

        Returns:
            dict, scores (float) of the batch mapped with the scores' names.
        """

        score = {name: getattr(utils, name)(ranks, targets) for name in self.scores_names}

        for name in self.scores_names:
            if score[name] is not None:
                score[name] = score[name].data.item()

        return score

    # endregion

    # region ML models

    def update_lr_scheduler(self):
        """ Performs a step of the learning rate scheduler if there is one. """

        pass

    def eval_mode(self):
        """ Sets the model on evaluation mode if there is one. """

        pass

    def train_mode(self):
        """ Sets the model on training mode if there is one. """

        pass

    def write_tensorboard(self, loss, score, tag, step):
        """
        Write the data using Tensorboard.

        Args:
            loss: float, loss to write.
            score: dict, score (float) to write, mapped with the name of the scores.
            tag: str, tag to write.
            step: int, index of the step to write.
        """

        pass

    # endregion

    # region Words methods

    @classmethod
    def get_words(cls, s, remove_stopwords, remove_punctuation, lower, lemmatize):
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
            s = s.translate(str.maketrans('', '', cls.punctuation))

        words = s.split()

        if remove_stopwords:
            words = [word for word in words if word.lower() not in cls.stopwords]

        if lemmatize:
            words = [cls.lemmatizer.lemmatize(word) for word in words]

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

    # region Encoding methods

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

    # region Word2Vec embedding methods

    def get_word_embedding(self, word):
        """
        Returns the pretrained embedding of a word in a line Tensor.

        Args:
            word: str, word to embed.

        Returns:
            torch.Tensor, embedding of word.
        """

        return torch.tensor(self.pretrained_model[word]) if word in self.pretrained_model.vocab else None

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
            else torch.zeros(self.pretrained_model_dim, dtype=torch.float)

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

    # TODO
    # region Bert embedding methods

    def get_bert_tokenize(self, text, segment):
        """
        Returns the tokens and segments lists (to tensorize) of the string text.

        Args:
            text: str, part of BERT input.
            segment: int, index (0 or 1) for first or second part of the BERT input.

        Returns:
            tokens_tensor: torch.Tensor, indexes of the tokens.
            segments_tensor: torch.Tensor, indexes of the segments.
        """

        text = '[CLS] ' + text + ' [SEP]' if segment == 0 else text + ' [SEP]'

        tokenized_text = self.tokenizer.tokenize(text)
        length = len(tokenized_text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [segment for _ in range(length)]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensor

    def get_bert_embedding(self, tokens_tensor, segments_tensor):
        """
        Returns the BERT embedding of the text.

        Args:
            tokens_tensor: torch.Tensor, representation of the tokens.
            segments_tensor: torch.Tensor, representation of the segments.

        Returns:
            torch.Tensor, embedding of the tokens.
        """

        with torch.no_grad():
            encoded_layer, _ = self.pretrained_model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

        return encoded_layer[:, 0, :]

    def get_bert_nsp_logits(self, tokens_tensor, segments_tensor):
        """
        Returns the BERT embedding of the text.

        Args:
            tokens_tensor: torch.Tensor, representation of the tokens.
            segments_tensor: torch.Tensor, representation of the segments.

        Returns:
            torch.Tensor, embedding of the tokens.
        """

        with torch.no_grad():
            logits = self.pretrained_model(tokens_tensor, segments_tensor)

        return logits

    # endregion

    # region Display methods

    def display_metrics(self, scores_names=None):
        """
        Display the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names
        score = {name: self.valid_scores[name][-1] for name in scores_names}

        print("Scores evaluated on the validation set:")

        for name, s in score.items():
            print('%s: %.5f' % (name, s))

    def final_plot(self, scores_names=None):
        """
        Plot the metrics of the model registered during the experiments.

        Args:
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
        """

        def plot(x1, x2, train_losses, valid_losses, valid_scores, scores_names):
            """
            Plot a single figure for the corresponding data.

            Args:
                x1: list, first x-axis of the plot, for the losses.
                x2: list, second x-axis of the plot, for the scores.
                train_losses: list, training losses to plot.
                valid_losses: list, validation losses to plot.
                valid_scores: dict, validation scores to plot.
                scores_names: iterable, names of the scores to plot, if not, plot all of them.
            """

            color_idx = 0
            colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:green',
                      'tab:olive', 'tab:gray', 'tab:brown', 'tab:purple', 'tab:pink']

            fig, ax1 = plt.subplots(figsize=(14, 8))
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

                ax2.scatter(x2, valid_scores[name], color=color, label='validation ' + name, s=50, marker='^')
                ax2.plot(x2, valid_scores[name], color=color, ls='--')

            fig.legend(loc='upper center')
            plt.show()

        scores_names = scores_names if scores_names is not None else self.scores_names

        n_experiments = len(self.train_scores[self.reference])

        total_x1, total_x2, offset = [], [], 0
        total_train_losses, total_valid_losses = [], []
        total_valid_scores = defaultdict(list)

        for i in range(n_experiments):
            n_epochs = len(self.train_scores[self.reference][i])
            n_points = len(self.train_scores[self.reference][i][0])

            x1 = list(arange(offset, offset + n_epochs, 1. / n_points))
            x2 = list(arange(offset + 1, offset + n_epochs + 1))
            offset += n_epochs

            train_losses = [x for epoch_losses in self.train_losses[i] for x in epoch_losses]
            valid_losses = self.valid_losses[i]
            valid_scores = {name: self.valid_scores[name][i] for name in scores_names}

            total_x1.extend(x1), total_x2.extend(x2)
            total_train_losses.extend(train_losses), total_valid_losses.extend(valid_losses)
            for name in scores_names:
                total_valid_scores[name].extend(valid_scores[name])

        plot(x1=total_x1, x2=total_x2, train_losses=total_train_losses, valid_losses=total_valid_losses,
             valid_scores=total_valid_scores, scores_names=scores_names)

    def explain(self, data_loader, scores_names, n_samples, n_answers):
        """
        Explain the model by displaying the samples and the reason of the prediction.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            scores_names: iterable, names of the scores to plot, if not, plot all of them.
            n_samples: int, number of samples to explain.
            n_answers: int, number of best answers to look at.
        """

        scores_names = scores_names if scores_names is not None else self.scores_names

        for batch_idx, (inputs, targets) in enumerate(data_loader[:n_samples]):
            features = self.features(inputs)

            outputs = self.pred(features)
            outputs = outputs[:, -1].reshape((-1, 1)) if len(outputs.shape) == 2 else outputs
            explanations = self.explanation(features)

            ranks = ranking(outputs)
            score = self.get_score(ranks, targets)

            print('\nEntities (%s): %s' % (inputs['entities_type_'], ',  '.join(inputs['entities'])))
            print("Scores:", ', '.join(['%s: %.5f' % (name, score[name]) for name in score if name in scores_names]))

            best_answers = [(ranks[i], inputs['choices'][i], outputs[i].item(), explanations[i])
                            for i in range(len(inputs['choices']))]
            best_answers = sorted(best_answers)[:n_answers]

            for rank, choice, output, explanation in best_answers:
                print('%d: %s (%d)' % (rank, choice, output)) if isinstance(output, int) \
                    else print('%d: %s (%.3f)' % (rank, choice, output))
                print('   ' + explanation) if explanation else None

    # endregion

# endregion


# region Baselines

class Baseline(BaseModel):
    """ Base structure for the Baselines. """

    # region Train methods

    def train_batch(self, features, targets, is_regression):
        """
        Perform the training on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores (float) of the batch of data.
        """

        raise Exception("A baseline cannot be trained.")

    def test_batch(self, features, targets, is_regression):
        """
        Perform the test or validation on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores (float) of the batch of data.
        """

        outputs = self.pred(features)
        ranks = ranking(outputs)
        batch_score = self.get_score(ranks, targets)

        return None, batch_score

    # endregion


# region Simple Baselines

class RandomBaseline(Baseline):
    """ Baseline with random predictions. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of the Random Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
        """

        super().__init__(scores_names=scores_names)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = torch.rand(len(features['choices'])).reshape((-1, 1))

        return grades

    # endregion


class CountsBaseline(Baseline):
    """ Baseline based on answers' overall frequency. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of the Counts Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
        """

        super().__init__(scores_names=scores_names)

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

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = [self.counts[choice] if choice in self.counts else 0 for choice in features['choices']]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion


class SummariesCountBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in one of the summaries. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of the Summaries Counts Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
        """

        super().__init__(scores_names=scores_names)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words, summaries_words = self.get_choices_words(features), self.get_summaries_words(features)
        context_words = self.get_context_words(features)

        grades = [len([word for summary_words in summaries_words for word in summary_words+context_words if word in
                       choices_words[i]])
                  for i in range(len(features['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, features):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words, summaries_words = self.get_choices_words(features), self.get_summaries_words(features)

        words = [', '.join([word for summary_words in summaries_words for word in summary_words
                            if word in choices_words[i]])
                 for i in range(len(features['choices']))]

        return words

    # endregion


class SummariesSoftOverlapBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the "soft" overlap of the summaries. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of the Summaries Soft Overlap Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
        """

        super().__init__(scores_names=scores_names)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = set([word for summary_words in self.get_summaries_words(features) for word in summary_words])

        grades = [len(choices_words[i].intersection(summaries_words)) for i in range(len(features['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, features):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = set([word for summary_words in self.get_summaries_words(features) for word in summary_words])

        words = [', '.join(choices_words[i].intersection(summaries_words)) for i in range(len(features['choices']))]

        return words

    # endregion


class SummariesHardOverlapBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the "hard" overlap of the summaries. """

    # region Class initialization

    def __init__(self, scores_names):
        """
        Initializes an instance of the Summaries Hard Overlap Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
        """

        super().__init__(scores_names=scores_names)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        grades = [len(choices_words[i].intersection(summaries_words)) for i in range(len(features['choices']))]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    def explanation(self, features):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        words = [', '.join(choices_words[i].intersection(summaries_words)) for i in range(len(features['choices']))]

        return words

    # endregion

# endregion


# region Word2Vec embedding Baselines

class ClosestAverageEmbedding(Baseline):
    """ Baseline with predictions based on the average embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names, pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the Closest Average Embedding Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_embedding = torch.stack([self.get_average_embedding(words)
                                         for words in self.get_choices_words(features)])
        other_embedding = self.get_average_embedding(self.get_other_words(features)).reshape((1, -1))

        grades = cosine_similarity(choices_embedding, other_embedding, dim=1).reshape((-1, 1))

        return grades

    # endregion


class ClosestHardOverlapEmbedding(Baseline):
    """ Baseline with predictions based on the "hard" overlap embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names, pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the Closest Hard Overlap Embedding Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_embedding = torch.stack([self.get_average_embedding(words)
                                         for words in self.get_choices_words(features)])

        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()
        summaries_embedding = self.get_average_embedding(summaries_words).reshape((1, -1))

        grades = cosine_similarity(choices_embedding, summaries_embedding, dim=1).reshape((-1, 1))

        return grades

    def explanation(self, features):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = self.get_choices_words(features)

        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        words = [', '.join(choices_words[i]) + '/' + ', '.join(summaries_words)
                 for i in range(len(features['choices']))]

        return words

    # endregion


class ClosestSoftOverlapEmbedding(Baseline):
    """ Baseline with predictions based on the "soft" overlap embedding proximity. """

    # region Class initialization

    def __init__(self, scores_names, pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the Closest Soft Overlap Embedding Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        choices_embedding = torch.stack([self.get_average_embedding(words)
                                         for words in self.get_choices_words(features)])

        summaries_words = set([word for summary_words in self.get_summaries_words(features) for word in summary_words])
        summaries_embedding = self.get_average_embedding(summaries_words).reshape((1, -1))

        grades = cosine_similarity(choices_embedding, summaries_embedding, dim=1).reshape((-1, 1))

        return grades

    def explanation(self, features):
        """
        Explain the model by returning some relevant information as a list of strings.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            list, information retrieved.
        """

        choices_words = self.get_choices_words(features)

        summaries_words = set([word for summary_words in self.get_summaries_words(features) for word in summary_words])

        words = [', '.join(choices_words[i]) + '/' + ', '.join(summaries_words)
                 for i in range(len(features['choices']))]

        return words

    # endregion

# endregion


# TODO
# region BERT embedding Baselines

class BertEmbedding(Baseline):
    """ Baseline with predictions based on the dot product of BERT embeddings. """

    # region Class initialization

    def __init__(self, scores_names, pretrained_model, tokenizer):
        """
        Initializes an instance of the BERT Embedding Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model, tokenizer=tokenizer)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = []

        text1 = features['context'] + ', '.join(features['entities']) + ', '.join(features['summaries'])
        tokens_tensor1, segments_tensor1 = self.get_bert_tokenize(text1, 0)
        embedding1 = self.get_bert_embedding(tokens_tensor1, segments_tensor1)

        for i in range(len(features['choices'])):
            text2 = features['choices'][i]
            tokens_tensor2, segments_tensor2 = self.get_bert_tokenize(text2, 0)
            embedding2 = self.get_bert_embedding(tokens_tensor2, segments_tensor2)

            grades.append(torch.dot(embedding1, embedding2).data.item())

        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion


class NSPBertEmbedding(Baseline):
    """ Baseline with predictions based on Next Sentence Prediction BERT. """

    # region Class initialization

    def __init__(self, scores_names, pretrained_model, tokenizer):
        """
        Initializes an instance of the Next Sentence Prediction BERT Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            pretrained_model: unknown, pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model, tokenizer=tokenizer)

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        grades = []

        text1 = features['context'] + ', '.join(features['entities']) + ', '.join(features['summaries'])
        tokens_tensor1, segments_tensor1 = self.get_bert_tokenize(text1, 0)

        for i in range(len(features['choices'])):
            text2 = features['choices'][i]
            tokens_tensor2, segments_tensor2 = self.get_bert_tokenize(text2, 1)

            tokens_tensor = torch.cat((tokens_tensor1, tokens_tensor2), dim=1)
            segments_tensor = torch.cat((segments_tensor1, segments_tensor2), dim=1)

            logits = self.get_bert_nsp_logits(tokens_tensor, segments_tensor)

            grades.append(logits[0, 0].data.item())

        grades = torch.tensor(grades).reshape((-1, 1))

        return grades

    # endregion

# endregion

# endregion


# region ML Models

class MLModel(BaseModel):
    """ Base structure for the ML models. """

    # region Class initialization

    model_name = None

    def __init__(self, scores_names, net, optimizer, lr_scheduler, loss, experiment_name, pretrained_model=None,
                 pretrained_model_dim=None, tokenizer=None):
        """
        Initializes an instance of the ML Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            net: nn.Module, neural net to train.
            optimizer: torch.optim.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        super().__init__(scores_names=scores_names, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim, tokenizer=tokenizer)

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        if experiment_name is not None:
            self.writer = SummaryWriter('logs/' + experiment_name + '/' + self.model_name)

    # endregion

    # region Train methods

    def train_batch(self, features, targets, is_regression):
        """
        Perform the training on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores of the batch of data.
        """

        self.optimizer.zero_grad()

        outputs = self.net(features)

        loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
        loss = self.loss(outputs, loss_targets)

        loss.backward()
        self.optimizer.step()
        batch_loss = loss.data.item()

        ranks = ranking(outputs.detach())
        batch_score = self.get_score(ranks, targets)

        return batch_loss, batch_score

    def test_batch(self, features, targets, is_regression):
        """
        Perform the test or validation on a batch of features.

        Args:
            features: dict or torch.Tensor, features of the data.
            targets: torch.Tensor, targets of the data.
            is_regression: bool, whether to use the regression set up for the task.

        Returns:
            batch_loss: float, loss on the batch of data.
            batch_score: dict, various scores (float) of the batch of data.
        """

        outputs = self.net(features)

        loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
        loss = self.loss(outputs.detach(), loss_targets)

        batch_loss = loss.data.item()

        ranks = ranking(outputs.detach())
        batch_score = self.get_score(ranks, targets)

        return batch_loss, batch_score

    # endregion

    # region Learning methods

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
        """

        self.eval_mode()
        grades = self.net(features)
        self.train_mode()

        return grades

    # endregion

    # region ML models

    def update_lr_scheduler(self):
        """ Performs a step of the learning rate scheduler if there is one. """

        old_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']

        print("Learning rate decreasing from %s to %s" % (old_lr, new_lr)) if old_lr != new_lr else None

    def eval_mode(self):
        """ Sets the model on evaluation mode if there is one. """

        self.net.eval()

    def train_mode(self):
        """ Sets the model on training mode if there is one. """

        self.net.train()

    def write_tensorboard(self, loss, score, tag, step):
        """
        Write the data using Tensorboard.

        Args:
            loss: float, loss to write.
            score: dict, score (float) to write, mapped with the name of the scores.
            tag: str, tag to write.
            step: int, index of the step to write.
        """

        if self.writer is not None:
            self.writer.add_scalar(tag=tag+'/loss', scalar_value=loss, global_step=step)
            self.writer.add_scalars(main_tag=tag+'/score', tag_scalar_dict=score, global_step=step)

    # endregion


class HalfBOWModel(MLModel):
    """ Model that uses a 1-hot encoding for the choices and a BOW for the other words. """

    # region Class initialization

    model_name = 'half_bow'

    def __init__(self, scores_names, net, optimizer, lr_scheduler, loss, experiment_name, vocab_frequency_range):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
        """

        super().__init__(scores_names=scores_names, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                         experiment_name=experiment_name)

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
            dict or torch.Tensor, features of the inputs.
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

    model_name = 'full_bow'

    def __init__(self, scores_names, net, optimizer, lr_scheduler, loss, experiment_name, vocab_frequency_range):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
        """

        super().__init__(scores_names=scores_names, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                         experiment_name=experiment_name)

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
            dict or torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        choices_embedding = [self.get_bow(words, self.word_to_idx) for words in self.get_choices_words(inputs)]
        choices_embedding = torch.stack(choices_embedding)

        other_words = self.get_other_words(inputs)
        other_embedding = self.get_bow(other_words, self.word_to_idx)
        other_embedding = other_embedding.expand((n, -1))

        features = torch.add(input=choices_embedding, other=other_embedding)

        return features

    # endregion


class EmbeddingModel(MLModel):
    """ Model that uses an average embedding both for the choice words and the context words. """

    # region Class initialization

    model_name = 'embedding_linear'

    def __init__(self, scores_names, net, optimizer, lr_scheduler, loss, experiment_name, pretrained_model,
                 pretrained_model_dim):
        """
        Initializes an instance of the linear Embedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                         experiment_name=experiment_name, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim)

    # endregion

    # region Learning methods

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            dict or torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        choices_embedding = torch.stack([self.get_average_embedding(words=words)
                                         for words in self.get_choices_words(inputs)])

        other_embedding = self.get_average_embedding(words=self.get_other_words(inputs))
        other_embedding = other_embedding.expand((n, -1))

        features = torch.cat((choices_embedding, other_embedding), dim=1)

        return features

    # endregion


class EmbeddingBilinearModel(MLModel):
    """ Model that uses an average embedding both for the choice words and the context words. """

    # region Class initialization

    model_name = 'embedding_bilinear'

    def __init__(self, scores_names, net, optimizer, lr_scheduler, loss, experiment_name, pretrained_model,
                 pretrained_model_dim):
        """
        Initializes an instance of the bilinear Embedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss,
                         experiment_name=experiment_name, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim)

    # endregion

    # region Learning methods

    def features(self, inputs):
        """
        Computes the features of the inputs.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            dict or torch.Tensor, features of the inputs.
        """

        n = len(inputs['choices'])

        choices_embedding = torch.stack([self.get_average_embedding(words=words)
                                         for words in self.get_choices_words(inputs)])

        other_embedding = self.get_average_embedding(words=self.get_other_words(inputs))
        other_embedding = other_embedding.expand((n, -1))

        features1, features2 = choices_embedding, other_embedding

        return features1, features2

    # endregion

# endregion
