from modeling import metrics
from modeling.utils import ranking, dict_mean, dict_append, dict_remove_none

from numpy import mean, arange
from numpy.random import seed, shuffle
from collections import defaultdict
from string import punctuation as str_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm_notebook as tqdm
from pyperclip import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class BaseModel:
    """ Base structure. """

    def __init__(self, scores_names, relevance_level, pretrained_model=None, pretrained_model_dim=None,
                 tokenizer=None, random_seed=2):
        """
        Initializes an instance of Base Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
            random_seed: int, the seed to use for the random processes.
        """

        self.scores_names = scores_names
        self.reference_score = scores_names[0]
        self.relevance_level = relevance_level

        self.pretrained_model = pretrained_model
        self.pretrained_model_dim = pretrained_model_dim
        self.tokenizer = tokenizer

        self.train_losses, self.train_scores = [], defaultdict(list)
        self.valid_losses, self.valid_scores = [], defaultdict(list)
        self.test_losses, self.test_scores = [], defaultdict(list)

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        seed(random_seed), torch.manual_seed(random_seed)

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
                      % (epoch+1, n_epochs, valid_epoch_loss, valid_epoch_score[self.reference_score]))

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

        print('Validation Loss: %.5f Validation Score: %.5f' % (loss, score[self.reference_score])) \
            if loss is not None else print('Validation Score: %.5f' % (score[self.reference_score]))

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

        print('Test Loss: %.5f Test Score: %.5f' % (loss, score[self.reference_score])) \
            if loss is not None else print('Test Score: %.5f' % (score[self.reference_score]))

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
            explanations: str, explanations of the prediction, optional.
        """

        return None, None

    def get_score(self, ranks, targets):
        """
        Returns the scores mentioned in self.scores_names for the ranks and targets.

        Args:
            ranks: torch.Tensor, ranks predicted for the batch.
            targets: torch.Tensor, true labels for the batch.

        Returns:
            dict, scores (float) of the batch mapped with the scores' names.
        """

        score = {name: getattr(metrics, name)(ranks=ranks.clone(),
                                              targets=targets.clone(),
                                              relevance_level=self.relevance_level)
                 for name in self.scores_names}

        for name in self.scores_names:
            if score[name] is not None:
                score[name] = score[name].data.item()

        return score

    def sort_batch(self, inputs, targets):
        """
        Returns lists of sub-batches of the inputs and targets where all the choices have the same number of words.

        Args:
            inputs: dict, inputs of the batch.
            targets: targets: torch.Tensor, targets of the data.

        Returns:
            sub_inputs: list, inputs sub-batches as dictionaries.
            sub_targets: list, targets sub-batches as torch.Tensors.
        """

        sorted_inputs, sorted_targets = dict(), dict()

        choices = inputs['choices']
        choices_words = self.get_choices_words(inputs)
        other = {key: value for key, value in inputs.items() if key != 'choices'}

        for i, choice in enumerate(choices):
            n_words = len(choices_words[i])

            if n_words not in sorted_inputs:
                sorted_inputs[n_words] = {'choices': []}
                sorted_inputs[n_words].update(other)
                sorted_targets[n_words] = []

            sorted_inputs[n_words]['choices'].append(choice)
            sorted_targets[n_words].append(targets[i].data.item())

        sub_inputs = [item for _, item in sorted_inputs.items()]
        sub_targets = [torch.tensor(item) for _, item in sorted_targets.items()]

        return sub_inputs, sub_targets

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

    @staticmethod
    def get_summaries_string(inputs):
        """
        Returns a string with the summaries that are not empty.

        Args:
            inputs: dict, inputs of the prediction.

        Returns:
            str, summaries without the empty ones.
        """

        summaries = [summary for summary in inputs['summaries'] if summary != 'No information found.']

        return ' '.join(summaries)

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

        context_words = self.get_context_words(inputs)
        summaries_words = [word for summary_words in self.get_summaries_words(inputs) for word in summary_words]

        return context_words + summaries_words

    @staticmethod
    def get_lists_counts(words_lists, words):
        """
        Returns the counts of words appearing in words that appear also in each list of words from words_lists.

        Args:
            words_lists: list, first words to compare, as a list of list of words.
            words: list, second words to compare, as a list of words.

        Returns:
            counts: torch.Tensor, counts of words as a column tensor.
            explanations: str, explanation of the counts.
        """

        counted_words = [[word for word in words if word in words_list] for words_list in words_lists]

        counts = torch.tensor([len(w) for w in counted_words]).reshape((-1, 1))
        explanations = [', '.join(w) for w in counted_words]

        return counts, explanations

    @staticmethod
    def get_sets_counts(words_sets, words):
        """
        Returns the counts of words appearing in words that appear also in each set of words from words_sets.

        Args:
            words_sets: set, first words to compare, as a list of sets of words.
            words: set, second words to compare, as a set of words.

        Returns:
            counts: torch.Tensor, counts of words as a column tensor.
            explanations: str, explanation of the counts.
        """

        counted_words = [words_set.intersection(words) for words_set in words_sets]

        counts = torch.tensor([len(w) for w in counted_words]).reshape((-1, 1))
        explanations = [', '.join(w) for w in counted_words]

        return counts, explanations

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

    def get_average_embedding_similarity(self, words_lists, words):
        """
        Returns the similarities between the average embeddings of the lists of words of words_lists and the average
        embedding of words.

        Args:
            words_lists: list, first words to compare, as a list of list of words.
            words: list, second words to compare, as a list of words.

        Returns:
            torch.tensor, similarities in a column tensor.
        """

        stacked_embeddings = torch.stack([self.get_average_embedding(words_list) for words_list in words_lists])
        embedding = self.get_average_embedding(words).reshape((1, -1))

        return torch.nn.functional.cosine_similarity(stacked_embeddings, embedding, dim=1).reshape((-1, 1))

    # endregion

    # region Bert embedding methods

    # TODO
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

    # TODO
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

    # TODO
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

    def get_bert_similarity(self, words_lists, words):
        """
        Returns the similarities between the BERT embeddings of the lists of words of words_lists and the average
        embedding of words.

        Args:
            words_lists: list, first words to compare, as a list of list of words.
            words: list, second words to compare, as a list of words.

        Returns:
            torch.tensor, similarities in a column tensor.
        """

        # TODO
        pass

# TODO
# class BertEmbedding(Baseline):
#     """ Baseline with predictions based on the dot product of BERT embeddings. """
#
#     def pred(self, features):
#         """
#         Predicts the outputs from the features.
#
#         Args:
#             features: dict or torch.Tensor, features of the batch.
#
#         Returns:
#             torch.Tensor, outputs of the prediction.
#         """
#
#         grades = []
#
#         text1 = features['context'] + ', '.join(features['entities']) + self.get_summaries_string(features)
#         tokens_tensor1, segments_tensor1 = self.get_bert_tokenize(text1, 0)
#         embedding1 = self.get_bert_embedding(tokens_tensor1, segments_tensor1)
#
#         for i in range(len(features['choices'])):
#             text2 = features['choices'][i]
#             tokens_tensor2, segments_tensor2 = self.get_bert_tokenize(text2, 0)
#             embedding2 = self.get_bert_embedding(tokens_tensor2, segments_tensor2)
#
#             grades.append(torch.dot(embedding1, embedding2).data.item())
#
#         grades = torch.tensor(grades).reshape((-1, 1))
#
#         return grades



# class NextSentencePredictionBert(Baseline):
#     """ Baseline with predictions based on Next Sentence Prediction BERT. """
#
#     def pred(self, features):
#         """
#         Predicts the outputs from the features.
#
#         Args:
#             features: dict or torch.Tensor, features of the batch.
#
#         Returns:
#             torch.Tensor, outputs of the prediction.
#         """
#
#         grades = []
#
#         text1 = features['context'] + ', '.join(features['entities']) + self.get_summaries_string(features)
#         tokens_tensor1, segments_tensor1 = self.get_bert_tokenize(text1, 0)
#
#         for i in range(len(features['choices'])):
#             text2 = features['choices'][i]
#             tokens_tensor2, segments_tensor2 = self.get_bert_tokenize(text2, 1)
#
#             tokens_tensor = torch.cat((tokens_tensor1, tokens_tensor2), dim=1)
#             segments_tensor = torch.cat((segments_tensor1, segments_tensor2), dim=1)
#
#             p = self.get_bert_next_sentence_probability(tokens_tensor, segments_tensor)
#
#             grades.append(p)
#
#         grades = torch.tensor(grades).reshape((-1, 1))
#
#         return grades

    # endregion

    # region Display methods

    def display_metrics(self, valid=True, test=False):
        """
        Display the validation or test metrics of the model registered during the last experiment.

        Args:
            valid: bool, whether or not to display the validation scores.
            test: bool, whether or not to display the test scores.
        """

        to_copy = ''

        if valid:
            print("\nScores evaluated on the validation set:")
            score = {name: self.valid_scores[name][-1] for name in self.scores_names}

            for name, s in score.items():
                print('%s: %.5f' % (name, s))

            to_copy += ' & ' + ' & '.join([str(round(score[name], 5)) for name in self.scores_names]) + ' &  '

        if test:
            print("\nScores evaluated on the test set:")
            score = {name: self.test_scores[name][-1] for name in self.scores_names}

            for name, s in score.items():
                print('%s: %.5f' % (name, s))

            to_copy += '\n' if to_copy else ''
            to_copy += ' & ' + ' & '.join([str(round(score[name], 5)) for name in self.scores_names]) + ' &  '

        print("\nLatex code for scores:\n" + to_copy)
        copy(to_copy)

    def plot(self, x1, x2, train_losses, valid_losses, valid_scores):
        """
        Plot a single figure for the corresponding data.

        Args:
            x1: list, first x-axis of the plot, for the losses.
            x2: list, second x-axis of the plot, for the scores.
            train_losses: list, training losses to plot.
            valid_losses: list, validation losses to plot.
            valid_scores: dict, validation scores to plot.
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

        for name in self.scores_names:
            color, color_idx = colors[color_idx], color_idx + 1

            ax2.scatter(x2, valid_scores[name], color=color, label='validation ' + name, s=50, marker='^')
            ax2.plot(x2, valid_scores[name], color=color, ls='--')

        fig.legend(loc='upper center')
        plt.show()

    def final_plot(self):
        """ Plot the metrics of the model registered during the experiments. """

        n_experiments = len(self.train_scores[self.reference_score])

        total_x1, total_x2, offset = [], [], 0
        total_train_losses, total_valid_losses = [], []
        total_valid_scores = defaultdict(list)

        for i in range(n_experiments):
            n_epochs = len(self.train_scores[self.reference_score][i])
            n_points = len(self.train_scores[self.reference_score][i][0])

            x1 = list(arange(offset, offset + n_epochs, 1. / n_points))
            x2 = list(arange(offset + 1, offset + n_epochs + 1))
            offset += n_epochs

            train_losses = [x for epoch_losses in self.train_losses[i] for x in epoch_losses]
            valid_losses = self.valid_losses[i]
            valid_scores = {name: self.valid_scores[name][i] for name in self.scores_names}

            total_x1.extend(x1), total_x2.extend(x2)
            total_train_losses.extend(train_losses), total_valid_losses.extend(valid_losses)
            for name in self.scores_names:
                total_valid_scores[name].extend(valid_scores[name])

        self.plot(x1=total_x1, x2=total_x2, train_losses=total_train_losses, valid_losses=total_valid_losses,
                  valid_scores=total_valid_scores)

    def explain(self, data_loader, n_samples, n_answers):
        """
        Explain the model by displaying the samples and the reason of the prediction.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            n_samples: int, number of samples to explain.
            n_answers: int, number of best answers to look at.
        """

        for batch_idx, (inputs, targets) in enumerate(data_loader[:n_samples]):
            features = self.features(inputs)

            outputs, explanations = self.pred(features)
            outputs = outputs[:, -1].reshape((-1, 1)) if len(outputs.shape) == 2 else outputs
            explanations = ['' for _ in range(len(inputs['choices']))] if explanations is None else explanations

            ranks = ranking(outputs)
            score = self.get_score(ranks, targets)

            print('\nEntities (%s): %s' % (inputs['entities_type_'], ',  '.join(inputs['entities'])))
            print("Scores:", ', '.join(['%s: %.5f' % (name, score[name])
                                        for name in score if name in self.scores_names]))

            best_answers = [(ranks[i], inputs['choices'][i], outputs[i].item(), explanations[i], targets[i].item())
                            for i in range(len(inputs['choices']))]

            first_answers = sorted(best_answers)[:n_answers]
            true_answers = [answer for answer in sorted(best_answers) if answer[4]][:n_answers]

            print("\nTop ranked answers:")
            for rank, choice, output, explanation, target in first_answers:
                print('%d (%d): %s (%d)' % (rank, target, choice, output)) if isinstance(output, int) \
                    else print('%d (%d): %s (%.3f)' % (rank, target, choice, output))
                print('   ' + explanation) if explanation else None

            print("\nGold/silver standard answers:")
            for rank, choice, output, explanation, target in true_answers:
                print('%d (%d): %s (%d)' % (rank, target, choice, output)) if isinstance(output, int) \
                    else print('%d (%d): %s (%.3f)' % (rank, target, choice, output))
                print('   ' + explanation) if explanation else None

    # endregion


# region Baselines

# region Base structures

class Baseline(BaseModel):
    """ Base structure for the Baselines. """

    def __init__(self, scores_names, relevance_level):
        """
        Initializes an instance of the Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level)

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

        outputs, _ = self.pred(features)
        ranks = ranking(outputs)
        batch_score = self.get_score(ranks, targets)

        return None, batch_score


class EmbeddingBaseline(Baseline):
    """ Base structure for the Baselines using embeddings. """

    def __init__(self, scores_names, relevance_level, pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the Embedding Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level)

        self.pretrained_model = pretrained_model
        self.pretrained_model_dim = pretrained_model_dim


class BertBaseline(Baseline):
    """ Base structure for the Baselines using BERT. """

    def __init__(self, scores_names, relevance_level, pretrained_model, tokenizer):
        """
        Initializes an instance of the BERT Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            pretrained_model: unknown, pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level)

        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer

# endregion


# region Simple Baselines

class RandomBaseline(Baseline):
    """ Baseline with random predictions. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        grades = torch.rand(len(features['choices'])).reshape((-1, 1))

        return grades, None


class FrequencyBaseline(Baseline):
    """ Baseline based on answers' overall frequency. """

    def __init__(self, scores_names, relevance_level):
        """
        Initializes an instance of the Frequency Baseline.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level)

        self.counts = defaultdict(int)

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
            explanations: str, explanations of the prediction, optional.
        """

        grades = [self.counts[choice] if choice in self.counts else 0 for choice in features['choices']]
        grades = torch.tensor(grades).reshape((-1, 1))

        return grades, None

    # endregion

# endregion


# region Summaries-dependent baselines

class SummariesCountBaseline(Baseline):
    """ Baseline based on the count of words of the summaries that are in the choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = [word for words in self.get_summaries_words(features) for word in words]

        counts, explanations = self.get_lists_counts(words_lists=choices_words, words=summaries_words)

        return counts, explanations


class SummariesUniqueCountBaseline(Baseline):
    """ Baseline based on the count of unique words of all the summaries that are in choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = set([word for summary_words in self.get_summaries_words(features) for word in summary_words])

        counts, explanation = self.get_sets_counts(words_sets=choices_words, words=summaries_words)

        return counts, explanation


class SummariesOverlapBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the overlap of all the summaries. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        counts, explanations = self.get_sets_counts(words_sets=choices_words, words=summaries_words)

        return counts, explanations


class SummariesAverageEmbeddingBaseline(EmbeddingBaseline):
    """ Baseline with predictions based on the average embedding proximity between the choice and all the summaries. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = [word for summary_words in self.get_summaries_words(features) for word in summary_words]

        similarity = self.get_average_embedding_similarity(words_lists=choices_words, words=summaries_words)

        return similarity, None


class SummariesOverlapAverageEmbeddingBaseline(EmbeddingBaseline):
    """ Baseline with predictions based on the average embedding proximity between the choice and the overlap of the
    summaries. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        similarity = self.get_average_embedding_similarity(words_lists=choices_words, words=summaries_words)

        return similarity, None


class ActivatedSummariesBaseline(Baseline):
    """ Baseline based on the number of summaries that have words matching the answer. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = self.get_summaries_words(features)

        activated_summaries = [[set(choice_words).intersection(set(summary_words)) for summary_words in summaries_words]
                               for choice_words in choices_words]
        activated_summaries = [[summary for summary in summaries if summary] for summaries in activated_summaries]

        counts = torch.tensor([len(summaries) for summaries in activated_summaries]).reshape((-1, 1))
        explanations = ['/'.join([', '.join(summary) for summary in summaries]) for summaries in activated_summaries]

        return counts, explanations


class SummariesBertBaseline(BertBaseline):
    """ Baseline with prediction based on the BERT's embeddings similarity between the choice and all the summaries. """

    # TODO
    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = [word for summary_words in self.get_summaries_words(features) for word in summary_words]

        similarity = self.get_bert_similarity(words_lists=choices_words, words=summaries_words)

        return similarity, None


class SummariesOverlapBertBaseline(BertBaseline):
    """ Baseline with prediction based on the BERT's embeddings similarity between the choice and all summaries'
    overlap. """

    # TODO
    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()

        similarity = self.get_bert_similarity(words_lists=choices_words, words=summaries_words)

        return similarity, None

# endregion


# region Context-only baselines

class ContextCountBaseline(Baseline):
    """ Baseline based on the count of words of the context that are in the choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        context_words = self.get_context_words(features)

        counts, explanations = self.get_lists_counts(words_lists=choices_words, words=context_words)

        return counts, explanations


class ContextUniqueCountBaseline(Baseline):
    """ Baseline based on the count of unique words of the context that are in choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        context_words = set(self.get_context_words(features))

        counts, explanations = self.get_sets_counts(words_sets=choices_words, words=context_words)

        return counts, explanations


class ContextAverageEmbeddingBaseline(EmbeddingBaseline):
    """ Baseline with predictions based on the average embedding proximity between the choice and the context. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        context_words = self.get_context_words(features)

        similarity = self.get_average_embedding_similarity(words_lists=choices_words, words=context_words)

        return similarity, None


class ContextBertBaseline(BertBaseline):
    """ Baseline with predictions based on the Bert embedding similarity between the choice and the context. """

    # TODO
    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        context_words = self.get_context_words(features)

        similarity = self.get_bert_similarity(words_lists=choices_words, words=context_words)

        return similarity, None

# endregion


# region Summaries & context dependent baselines

class SummariesContextCountBaseline(Baseline):
    """ Baseline based on the count of words of the summaries and the context that are in the choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        other_words = self.get_other_words(features)

        counts, explanations = self.get_lists_counts(words_lists=choices_words, words=other_words)

        return counts, explanations


class SummariesContextUniqueCountBaseline(Baseline):
    """ Baseline based on the count of unique words of all the summaries and the context that are in choice. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]
        other_words = set(self.get_other_words(features))

        counts, explanations = self.get_sets_counts(words_sets=choices_words, words=other_words)

        return counts, explanations


class SummariesOverlapContextBaseline(Baseline):
    """ Baseline based on the count of words from choice that are in the overlap of all the summaries or in the
    context. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = [set(choice_words) for choice_words in self.get_choices_words(features)]

        summaries_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        summaries_words = set.intersection(*summaries_words) if summaries_words else set()
        summaries_words.update(set(self.get_context_words(features)))

        counts, explanations = self.get_sets_counts(words_sets=choices_words, words=summaries_words)

        return counts, explanations


class SummariesContextAverageEmbeddingBaseline(EmbeddingBaseline):
    """ Baseline with predictions based on the average embedding proximity between the choice and all the summaries and
    the context. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        other_words = self.get_other_words(features)

        similarity = self.get_average_embedding_similarity(words_lists=choices_words, words=other_words)

        return similarity, None


class SummariesOverlapContextAverageEmbeddingBaseline(EmbeddingBaseline):
    """ Baseline with predictions based on the average embedding proximity between the choice and the overlap of the
    summaries and the context. """

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)

        other_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        other_words = set.intersection(*other_words) if other_words else set()
        other_words.update(set(self.get_context_words(features)))

        similarity = self.get_average_embedding_similarity(words_lists=choices_words, words=other_words)

        return similarity, None


class SummariesContextBertBaseline(BertBaseline):
    """ Baseline with prediction based on the BERT's similarity between the choice and all summaries and the
    context. """

    # TODO
    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)
        other_words = self.get_other_words(features)

        similarity = self.get_bert_similarity(words_lists=choices_words, words=other_words)

        return similarity, None


class SummariesOverlapContextBertBaseline(BertBaseline):
    """ Baseline with prediction based on the BERT's embeddings similarity between the choice and all summaries'
    overlap and the context. """

    # TODO
    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        choices_words = self.get_choices_words(features)

        other_words = [set(summary_words) for summary_words in self.get_summaries_words(features) if summary_words]
        other_words = set.intersection(*other_words) if other_words else set()
        other_words.update(set(self.get_context_words(features)))

        similarity = self.get_bert_similarity(words_lists=choices_words, words=other_words)

        return similarity, None

# endregion

# endregion


# region ML Models

class MLModel(BaseModel):
    """ Base structure for the ML models. """
    model_name = None

    def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
                 pretrained_model=None, pretrained_model_dim=None, tokenizer=None):
        """
        Initializes an instance of the ML Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            net: nn.Module, neural net to train.
            optimizer: torch.optim.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
            tokenizer: transformers.tokenizer, tokenizer.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, pretrained_model=pretrained_model,
                         pretrained_model_dim=pretrained_model_dim, tokenizer=tokenizer)

        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        if experiment_name is not None:
            self.writer = SummaryWriter('logs/' + experiment_name + '/' + self.model_name)

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

    def pred(self, features):
        """
        Predicts the outputs from the features.

        Args:
            features: dict or torch.Tensor, features of the batch.

        Returns:
            torch.Tensor, outputs of the prediction.
            explanations: str, explanations of the prediction, optional.
        """

        self.eval_mode()
        pred = self.net(features)
        self.train_mode()

        return pred, None

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
    model_name = 'half_bow'

    def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
                 vocab_frequency_range):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name)

        self.vocab_frequency_range = vocab_frequency_range

        self.choice_to_idx = dict()
        self.word_to_idx = dict()
        self.word_counts = defaultdict(int)

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
    model_name = 'full_bow'

    def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
                 vocab_frequency_range):
        """
        Initializes an instance of the Bag of Word Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name)

        self.vocab_frequency_range = vocab_frequency_range

        self.word_to_idx = dict()
        self.word_counts = defaultdict(int)

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
    model_name = 'embedding_linear'

    def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
                 pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the linear Embedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name,
                         pretrained_model=pretrained_model, pretrained_model_dim=pretrained_model_dim)

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


class EmbeddingBilinearModel(MLModel):
    """ Model that uses an average embedding both for the choice words and the context words. """
    model_name = 'embedding_bilinear'

    def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
                 pretrained_model, pretrained_model_dim):
        """
        Initializes an instance of the bilinear Embedding Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            net: nn.Module, neural net to train.
            optimizer: torch.optimizer, optimizer for the neural net.
            lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
            loss: torch.nn.Loss, loss to use.
            experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
            pretrained_model: unknown, pretrained embedding or model.
            pretrained_model_dim: int, size of the pretrained embedding or model.
        """

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name,
                         pretrained_model=pretrained_model, pretrained_model_dim=pretrained_model_dim)

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


class BertModel(MLModel):
    # TODO
    pass

# endregion
