from modeling import metrics
from modeling.utils import get_ranks, list_remove_none, dict_append, dict_mean, dict_std
from toolbox.utils import inputs_to_context

from numpy import arange, mean, std
from numpy.random import seed, shuffle
from collections import defaultdict
from copy import deepcopy
from re import findall
from string import punctuation as str_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter
from fairseq.data.data_utils import collate_tokens
import matplotlib.pyplot as plt
import torch


class BaseModel:
    """ Model structure. """

    def __init__(self, scores_names, relevance_level, trained_model, tensorboard_logs_path, experiment_name,
                 random_seed, root=""):
        """
        Initializes an instance of Model.

        Args:
            scores_names: iterable, names of the scores to use, the first one being monitored during training.
            relevance_level: int, minimum label to consider a choice as relevant.
            trained_model: unknown, pretrained embedding or model.
            tensorboard_logs_path: str, path to the tensorboard logs folder.
            experiment_name: str, if not None, name of the folder to save the tensorboard in.
            random_seed: int, the seed to use for the random processes.
            root: str, path to the root of the project.
        """

        self.scores_names = scores_names
        self.relevance_level = relevance_level
        self.trained_model = trained_model

        self.train_losses, self.train_scores = [], defaultdict(list)
        self.valid_losses, self.valid_scores = [], defaultdict(list)
        self.test_losses, self.test_scores = [], defaultdict(list)

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.writer = None

        if experiment_name is not None:
            model_name = self.__class__.__name__
            model_name = "_".join([word.lower() for word in findall(r'[A-Z][^A-Z]*', model_name)])

            self.writer = SummaryWriter(root + tensorboard_logs_path + experiment_name + "/" + model_name)

        seed(random_seed), torch.manual_seed(random_seed)

    # region Training pipeline methods

    def preview(self, data_loader):
        """
        Preview the data for the model.

        Args:
            data_loader: list, list of ranking tasks, which are lists of (inputs, targets) batches.
        """

        pass

    def valid(self, data_loader):
        """
        Validate the Model on data_loader.

        Args:
            data_loader: list, list of ranking tasks, which are lists of (inputs, targets) batches.
        """

        epoch_losses, epoch_scores = self.evaluate_epoch(data_loader=data_loader)

        self.valid_losses.append(epoch_losses)
        dict_append(self.valid_scores, epoch_scores)

        self.print_metrics(epoch_losses=epoch_losses, epoch_scores=epoch_scores)

    def test(self, data_loader):
        """
        Test the Model on data_loader.

        Args:
            data_loader: list, list of ranking tasks, which are lists of (inputs, targets) batches.
        """

        epoch_losses, epoch_scores = self.evaluate_epoch(data_loader=data_loader)

        self.test_losses.append(epoch_losses)
        dict_append(self.test_scores, epoch_scores)

        self.print_metrics(epoch_losses=epoch_losses, epoch_scores=epoch_scores)

    def evaluate_epoch(self, data_loader):
        """
        Evaluate the model for one epoch on data_loader.

        Args:
            data_loader: list, list of ranking tasks, which are lists of (inputs, targets) batches.

        Returns:
            epoch_losses: list, losses (float) of the ranking tasks of an epoch.
            epoch_scores: dict, list of scores (float) of the ranking tasks of an epoch, mapped with the score's names.
        """

        epoch_losses, epoch_scores = [], defaultdict(list)
        n_rankings = len(data_loader)

        shuffle(data_loader)

        for ranking_idx, ranking in tqdm(enumerate(data_loader), total=n_rankings):
            ranking_loss, ranking_score = self.evaluate_ranking(ranking=ranking)

            self.write_tensorboard(loss=ranking_loss, score=ranking_score, tag='test', step=ranking_idx)
            epoch_losses.append(ranking_loss), dict_append(epoch_scores, ranking_score)

        return epoch_losses, epoch_scores

    def evaluate_ranking(self, ranking):
        """
        Evaluate the model for one ranking task.

        Args:
            ranking: list, batches (inputs, targets) of the ranking task.

        Returns:
            ranking_loss: float, loss of the ranking task.
            ranking_score: dict, list of scores (float) of the ranking task, mapped with the score's names.
        """

        ranking_outputs, ranking_targets = [], []

        for inputs, targets in ranking:
            outputs = self.pred(inputs)
            ranking_outputs.append(outputs), ranking_targets.append(targets)

        ranking_outputs, ranking_targets = torch.cat(ranking_outputs), torch.cat(ranking_targets)

        ranks = get_ranks(ranking_outputs)
        batch_score = self.get_score(ranks, ranking_targets)

        return None, batch_score

    def pred(self, inputs):
        """
        Predicts the batch outputs from its inputs.

        Args:
            inputs: dict, inputs of a batch.

        Returns:
            torch.Tensor, outputs of the prediction in a column Tensor.
        """

        return torch.tensor([0])

    # endregion

    # region Scores methods

    def get_score(self, ranks, targets):
        """
        Returns the scores of the ranks, given the targets.

        Args:
            ranks: torch.Tensor, ranks predicted for a ranking task.
            targets: torch.Tensor, true labels for a ranking task.

        Returns:
            dict, scores (float) of the ranking task mapped with the scores' names.
        """

        score_dict = dict()

        for name in self.scores_names:
            score = getattr(metrics, name)(ranks=ranks.clone(),
                                           targets=targets.clone(),
                                           relevance_level=self.relevance_level)

            score_dict[name] = score.data.item()

        return score_dict

    @staticmethod
    def get_mean_std(epoch_losses, epoch_scores):
        """
        Returns the mean and standard deviation of the list of losses and the dict of list of scores.

        Args:
            epoch_losses: list, losses (float) of an epoch.
            epoch_scores: dict, list of scores (float) of an epoch, mapped with the name of the score.

        Returns:
            loss_mean: float, mean of the losses.
            loss_std: float, standard deviation of the losses.
            score_mean: float, mean of the scores.
            score_std: float, standard deviation of the scores.
        """

        epoch_scores, epoch_losses = deepcopy(epoch_scores), deepcopy(epoch_losses)

        epoch_losses = list_remove_none(epoch_losses)
        loss_mean = float(mean(epoch_losses)) if epoch_losses else None
        loss_std = float(std(epoch_losses)) if epoch_losses else None

        score_mean = dict_mean(epoch_scores)
        score_std = dict_std(epoch_scores)

        return loss_mean, loss_std, score_mean, score_std

    def print_metrics(self, epoch_losses, epoch_scores):
        """
        Prints the scores of the model registered during the given epoch.

        Args:
            epoch_losses: list, losses (float) of an epoch.
            epoch_scores: dict, list of scores (float) of an epoch, mapped with the name of the score.
        """

        loss_mean, loss_std, score_mean, score_std = self.get_mean_std(epoch_losses=epoch_losses,
                                                                       epoch_scores=epoch_scores)

        if loss_mean is not None and loss_std is not None:
            print("Loss: %.5f (+/-%.5f)" % (loss_mean, loss_std))

        for name in self.scores_names:
            print("%s: %.5f (+/-%.5f)" % (name, score_mean[name], score_std[name]))

        print()

    def write_tensorboard(self, loss, score, tag, step):
        """
        Write the metrics using Tensorboard.

        Args:
            loss: float, loss to write.
            score: dict, score (float) to write, mapped with the name of the scores.
            tag: str, tag to write.
            step: int, index of the step to write.
        """

        if self.writer is not None:
            if loss is not None:
                self.writer.add_scalar(tag=tag + '/loss', scalar_value=loss, global_step=step)

            for name in self.scores_names:
                if score[name] is not None:
                    self.writer.add_scalar(tag=tag + '/' + name, scalar_value=score[name],
                                           global_step=step)

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

    def get_choices_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=False,
                          setten=False):
        """
        Returns the words from the inputs' choices as a list of list.

        Args:
            inputs: dict, inputs of a batch.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.
            setten: bool, whether or not to return the results as a list of set.

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

        if setten:
            choices_words = [set(words) for words in choices_words]

        return choices_words

    def get_entities_words(self, inputs, remove_stopwords=False, remove_punctuation=True, lower=True, lemmatize=False,
                           flatten=False):
        """
        Returns the words from the inputs' entities as a list of list.

        Args:
            inputs: dict, inputs of a batch.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.
            flatten: bool, whether or not to flatten the output list.

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

        if flatten:
            entities_words = [word for words in entities_words for word in words]

        return entities_words

    def get_context_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=True,
                          setten=False):
        """
        Returns the words from the inputs' context as a list.

        Args:
            inputs: dict, inputs of a batch.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.
            setten: bool, whether or not to return the results as a list of set.

        Returns:
            list, words of the inputs' context.
        """

        context_words = self.get_words(s=inputs['context'],
                                       remove_stopwords=remove_stopwords,
                                       remove_punctuation=remove_punctuation,
                                       lower=lower,
                                       lemmatize=lemmatize)

        if setten:
            context_words = set(context_words)

        return context_words

    @staticmethod
    def get_wikipedia_string(inputs):
        """
        Returns a string with the wikipedia summaries that are not empty.

        Args:
            inputs: dict, inputs of a batch.
        """

        summaries = [wikipedia for wikipedia in inputs['wikipedia'] if wikipedia != 'No information found.']

        return ' '.join(summaries)

    def get_wikipedia_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=True,
                            flatten=False, setten=False):
        """
        Returns the words from the inputs' wikipedia summaries as a list of list.

        Args:
            inputs: dict, inputs of a batch.
            remove_stopwords: bool, whether to remove the stopwords or not.
            remove_punctuation: bool, whether to remove the punctuation or not.
            lower: bool, whether to remove the capitals or not.
            lemmatize: bool, whether or not to lemmatize the words or not.
            flatten: bool, whether or not to flatten the result.
            setten: bool, whether or not to return the results as a set or a list of set.

        Returns:
            list, words lists of the inputs' wikipedia summaries.
        """

        wikipedia_words = []

        for wikipedia in inputs['wikipedia']:
            if wikipedia != "No information found.":
                wikipedia_words.append(self.get_words(s=wikipedia,
                                                      remove_stopwords=remove_stopwords,
                                                      remove_punctuation=remove_punctuation,
                                                      lower=lower,
                                                      lemmatize=lemmatize))
            else:
                wikipedia_words.append([])

        if flatten:
            wikipedia_words = [word for words in wikipedia_words for word in words]
            if setten:
                wikipedia_words = set(wikipedia_words)

        else:
            if setten:
                wikipedia_words = [set(words) for words in wikipedia_words]

        return wikipedia_words

    def get_other_words(self, inputs, setten=False):
        """
        Returns the words from the inputs' entities, context and wikipedia summaries, in a list.

        Args:
            inputs: dict, inputs of a batch.
            setten: bool, whether or not to return the results as a set.
        """

        entities_words = self.get_entities_words(inputs, flatten=True)
        context_words = self.get_context_words(inputs)
        wikipedia_words = self.get_wikipedia_words(inputs, flatten=True)

        other_words = entities_words + context_words + wikipedia_words

        if setten:
            other_words = set(other_words)

        return other_words

    @staticmethod
    def get_lists_counts(words_lists, words):
        """
        Returns the counts of words appearing in words that appear also in each list of words from words_lists.

        Args:
            words_lists: list, first words to compare, as a list of list of words.
            words: list, second words to compare, as a list of words.

        Returns:
            torch.Tensor, counts of words as a column tensor.
        """

        counted_words = [[word for word in words if word in words_list] for words_list in words_lists]

        return torch.tensor([len(w) for w in counted_words]).reshape((-1, 1))

    @staticmethod
    def get_sets_counts(words_sets, words):
        """
        Returns the counts of words appearing in words that appear also in each set of words from words_sets.

        Args:
            words_sets: set, first words to compare, as a list of sets of words.
            words: set, second words to compare, as a set of words.

        Returns:
            torch.Tensor, counts of words as a column tensor.
        """

        counted_words = [words_set.intersection(words) for words_set in words_sets]

        return torch.tensor([len(w) for w in counted_words]).reshape((-1, 1))

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

        return torch.tensor(self.trained_model[word]) if word in self.trained_model.vocab else None

    def get_average_embedding(self, words):
        """
        Returns the average pretrained embedding of words in a line Tensor.

        Args:
            words: list, words to embed.

        Returns:
            torch.Tensor, average embedding of the words.
        """

        embeddings = [self.get_word_embedding(word) for word in words]
        embeddings = [embedding for embedding in embeddings if embedding is not None]

        if embeddings:
            return torch.stack(embeddings).mean(dim=0)

        else:
            word = list(self.trained_model.vocab)[0]
            embedding = self.get_word_embedding(word)

            return torch.zeros_like(embedding, dtype=torch.float)

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

    # region Display methods

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

    # TODO
    def final_plot(self):
        """ Plot the metrics of the model. """

        reference = self.scores_names[0]
        n_experiments = len(self.train_scores[reference])

        total_x1, total_x2, offset = [], [], 0
        total_train_losses, total_valid_losses = [], []
        total_valid_scores = defaultdict(list)

        for i in range(n_experiments):
            n_epochs = len(self.train_scores[reference][i])
            n_points = len(self.train_scores[reference][i][0])

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

    # TODO
    def show(self, data_loader, show_rankings, show_choices):
        """
        Show the model results on different rankings.

        Args:
            data_loader: list, pairs of (inputs, targets) batches.
            show_rankings: int, number of rankings to show.
            show_choices: int, number of best choices to show.
        """

        for ranking_idx, ranking_task in enumerate(data_loader[:show_rankings]):
            pass

            # outputs, explanations = self.pred(inputs)
            # outputs = outputs[:, -1].reshape((-1, 1)) if len(outputs.shape) == 2 else outputs
            # explanations = ['' for _ in range(len(inputs['choices']))] if explanations is None else explanations
            #
            # ranks = get_ranks(outputs)
            # score = self.get_score(ranks, targets)
            #
            # print('\nEntities (%s): %s' % (inputs['entities_type_'], ',  '.join(inputs['entities'])))
            # print("Scores:", ', '.join(['%s: %.5f' % (name, score[name])
            #                             for name in score if name in self.scores_names]))
            #
            # best_answers = [(ranks[i], inputs['choices'][i], outputs[i].item(), explanations[i], targets[i].item())
            #                 for i in range(len(inputs['choices']))]
            #
            # first_answers = sorted(best_answers)[:n_answers]
            # true_answers = [answer for answer in sorted(best_answers) if answer[4]][:n_answers]
            #
            # print("\nTop ranked answers:")
            # for rank, choice, output, explanation, target in first_answers:
            #     print('%d (%d): %s (%d)' % (rank, target, choice, output)) if isinstance(output, int) \
            #         else print('%d (%d): %s (%.3f)' % (rank, target, choice, output))
            #     print('   ' + explanation) if explanation else None
            #
            # print("\nGold/silver standard answers:")
            # for rank, choice, output, explanation, target in true_answers:
            #     print('%d (%d): %s (%d)' % (rank, target, choice, output)) if isinstance(output, int) \
            #         else print('%d (%d): %s (%.3f)' % (rank, target, choice, output))
            #     print('   ' + explanation) if explanation else None

    # endregion


# region Baselines

# region Simple Baselines

class Random(BaseModel):
    """ Baseline with random predictions. """

    def pred(self, inputs):
        return torch.rand(len(inputs['choices'])).reshape((-1, 1))


class Frequency(BaseModel):
    """ Baseline based on answers' overall frequency. """

    def __init__(self, scores_names, relevance_level, trained_model, tensorboard_logs_path, experiment_name,
                 random_seed, root=""):

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, trained_model=trained_model,
                         tensorboard_logs_path=tensorboard_logs_path, experiment_name=experiment_name,
                         random_seed=random_seed, root=root)

        self.counts = defaultdict(int)

    def preview(self, data_loader):
        print("Learning answers counts...")

        for ranking_task in tqdm(data_loader, total=len(data_loader)):
            for inputs, targets in ranking_task:
                choices = inputs['choices']
                for i in range(len(choices)):
                    self.counts[choices[i]] += targets[i].data.item()

    def pred(self, inputs):
        grades = [self.counts[choice] if choice in self.counts else 0 for choice in inputs['choices']]

        return torch.tensor(grades).reshape((-1, 1))

# endregion


# region Summaries-dependent baselines

class SummariesCount(BaseModel):
    """ Baseline based on the count of words of the summaries that are in the choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikipedia_words = self.get_wikipedia_words(inputs, flatten=True)

        return self.get_lists_counts(words_lists=choices_words, words=wikipedia_words)


class SummariesUniqueCount(BaseModel):
    """ Baseline based on the count of unique words of all the summaries that are in choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)
        wikipedia_words = self.get_wikipedia_words(inputs, flatten=True, setten=True)

        return self.get_sets_counts(words_sets=choices_words, words=wikipedia_words)


class SummariesOverlap(BaseModel):
    """ Baseline based on the count of words from choice that are in the overlap of all the summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)

        wikipedia_words = [words for words in self.get_wikipedia_words(inputs, setten=True) if words]
        wikipedia_words = set.intersection(*wikipedia_words) if wikipedia_words else set()

        return self.get_sets_counts(words_sets=choices_words, words=wikipedia_words)


class ActivatedSummaries(BaseModel):
    """ Baseline based on the number of summaries that have words matching the answer. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikipedia_words = self.get_wikipedia_words(inputs)

        activated_summaries = [[set(choice_words).intersection(set(summary_words)) for summary_words in wikipedia_words]
                               for choice_words in choices_words]
        activated_summaries = [[summary for summary in summaries if summary] for summaries in activated_summaries]

        return torch.tensor([len(summaries) for summaries in activated_summaries]).reshape((-1, 1))


class SummariesAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and all the summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikipedia_words = self.get_wikipedia_words(inputs, flatten=True)

        return self.get_average_embedding_similarity(words_lists=choices_words, words=wikipedia_words)


class SummariesOverlapAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and the overlap of the
    summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)

        wikipedia_words = [words for words in self.get_wikipedia_words(inputs, setten=True) if words]
        wikipedia_words = set.intersection(*wikipedia_words) if wikipedia_words else set()

        return self.get_average_embedding_similarity(words_lists=choices_words, words=wikipedia_words)

# endregion


# region Context-only baselines

class ContextCount(BaseModel):
    """ Baseline based on the count of words of the context that are in the choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        context_words = self.get_context_words(inputs)

        return self.get_lists_counts(words_lists=choices_words, words=context_words)


class ContextUniqueCount(BaseModel):
    """ Baseline based on the count of unique words of the context that are in choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)
        context_words = self.get_context_words(inputs, setten=True)

        return self.get_sets_counts(words_sets=choices_words, words=context_words)


class ContextAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and the context. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        context_words = self.get_context_words(inputs)

        return self.get_average_embedding_similarity(words_lists=choices_words, words=context_words)

# endregion


# region Summaries & context dependent baselines

class SummariesContextCount(BaseModel):
    """ Baseline based on the count of words of the summaries and the context that are in the choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        other_words = self.get_other_words(inputs)

        return self.get_lists_counts(words_lists=choices_words, words=other_words)


class SummariesContextUniqueCount(BaseModel):
    """ Baseline based on the count of unique words of all the summaries and the context that are in choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)
        other_words = self.get_other_words(inputs, setten=True)

        return self.get_sets_counts(words_sets=choices_words, words=other_words)


class SummariesContextOverlap(BaseModel):
    """ Baseline based on the count of words from choice that are in the overlap of all the summaries or in the
    context. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)

        wikipedia_words = [words for words in self.get_wikipedia_words(inputs, setten=True) if words]
        wikipedia_words = set.intersection(*wikipedia_words) if wikipedia_words else set()
        wikipedia_words.update(self.get_context_words(inputs, setten=True))

        return self.get_sets_counts(words_sets=choices_words, words=wikipedia_words)


class SummariesContextAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and all the summaries and
    the context. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        other_words = self.get_other_words(inputs)

        return self.get_average_embedding_similarity(words_lists=choices_words, words=other_words)


class SummariesContextOverlapAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and the overlap of the
    summaries and the context. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)

        other_words = [words for words in self.get_wikipedia_words(inputs, setten=True) if words]
        other_words = set.intersection(*other_words) if other_words else set()
        other_words.update(self.get_context_words(inputs, setten=True))

        return self.get_average_embedding_similarity(words_lists=choices_words, words=other_words)

# endregion

# endregion


# region BART

class ClassificationBart(BaseModel):
    """ BART trained as a classifier. """

    def __init__(self, scores_names, relevance_level, trained_model, tensorboard_logs_path, experiment_name,
                 random_seed, root=""):

        super().__init__(scores_names=scores_names, relevance_level=relevance_level, trained_model=trained_model,
                         tensorboard_logs_path=tensorboard_logs_path, experiment_name=experiment_name,
                         random_seed=random_seed, root=root)

        self.label_fn = lambda label: self.trained_model.task.label_dictionary.string(
            [label + self.trained_model.task.label_dictionary.nspecial])
        ###
        print(self.label_fn)
        raise Exception
        ###

    # TODO
    def pred(self, inputs):
        sentence1 = inputs_to_context(inputs)
        batch_encoding = [self.trained_model.encode(sentence1, sentence2) for sentence2 in inputs['choices']]

        batch_tokens = collate_tokens(values=batch_encoding, pad_idx=1)

        if torch.cuda.is_available():
            batch_tokens.cuda()

        logprobs = self.trained_model.predict('sentence_classification_head', batch_tokens)
        prediction = logprobs.argmax().item()
        prediction_label = self.label_fn(prediction)

        # return logprobs[:, 2].reshape((-1, 1))

# endregion


# class MLModel(BaseModel):
#     """ Base structure for the ML models. """
#     model_name = None
#
#     def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
#                  pretrained_model=None, pretrained_model_dim=None, tokenizer=None):
#         """
#         Initializes an instance of the ML Model.
#
#         Args:
#             scores_names: iterable, names of the scores to use, the first one being monitored during training.
#             relevance_level: int, minimum label to consider a choice as relevant.
#             net: nn.Module, neural net to train.
#             optimizer: torch.optim.optimizer, optimizer for the neural net.
#             lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
#             loss: torch.nn.Loss, loss to use.
#             experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
#             pretrained_model: unknown, pretrained embedding or model.
#             pretrained_model_dim: int, size of the pretrained embedding or model.
#             tokenizer: transformers.tokenizer, tokenizer.
#         """
#
#         super().__init__(scores_names=scores_names, relevance_level=relevance_level,
#                          pretrained_model=pretrained_model, pretrained_model_dim=pretrained_model_dim,
#                          tokenizer=tokenizer)
#
#         self.net = net
#         self.optimizer = optimizer
#         self.lr_scheduler = lr_scheduler
#         self.loss = loss
#
#     # region Train methods
#
#     def train(self, train_loader, valid_loader, n_epochs, n_updates, is_regression):
#         """
#         Train the Model on train_loader and validate on valid_loader at each epoch.
#
#         Args:
#             train_loader: list of (inputs, targets) batches, training inputs and outputs.
#             valid_loader: list of (inputs, targets) batches, valid inputs and outputs.
#             n_epochs: int, number of epochs to perform.
#             n_updates: int, number of batches between the updates.
#             is_regression: bool, whether to use the regression set up for the task.
#         """
#
#         print("Training of the model...\n")
#
#         train_losses, valid_losses, train_scores, valid_scores = [], [], defaultdict(list), defaultdict(list)
#
#         for epoch in range(n_epochs):
#
#             try:
#                 shuffle(train_loader), shuffle(valid_loader)
#
#                 train_epoch_losses, train_epoch_scores = self.train_epoch(data_loader=train_loader,
#                                                                           n_updates=n_updates,
#                                                                           is_regression=is_regression,
#                                                                           epoch=epoch)
#
#                 valid_epoch_loss, valid_epoch_score = self.test_epoch(data_loader=valid_loader,
#                                                                       is_regression=is_regression)
#
#                 self.write_tensorboard(mean(train_epoch_losses), dict_mean(train_epoch_scores), 'epoch train', epoch)
#                 self.write_tensorboard(valid_epoch_loss, valid_epoch_score, 'epoch valid', epoch)
#
#                 train_losses.append(train_epoch_losses), valid_losses.append(valid_epoch_loss)
#                 dict_append(train_scores, train_epoch_scores), dict_append(valid_scores, valid_epoch_score)
#
#                 print('Epoch %d/%d: Validation Loss: %.5f Validation Score: %.5f'
#                       % (epoch + 1, n_epochs, valid_epoch_loss, valid_epoch_score[self.reference_score]))
#
#                 self.update_lr_scheduler()
#
#                 print('--------------------------------------------------------------')
#
#             except KeyboardInterrupt:
#                 print("Keyboard interruption, exiting and saving all results except current epoch...")
#                 break
#
#         if train_losses and train_scores and valid_losses and valid_scores:
#             self.train_losses.append(train_losses), self.valid_losses.append(valid_losses)
#             dict_append(self.train_scores, train_scores), dict_append(self.valid_scores, valid_scores)
#
#     def train_epoch(self, data_loader, n_updates, is_regression, epoch):
#         """
#         Trains the model for one epoch on data_loader.
#
#         Args:
#             data_loader: list, pairs of (inputs, targets) batches.
#             n_updates: int, number of batches between the updates.
#             is_regression: bool, whether to use the regression set up for the task.
#             epoch: int, epoch number of the training.
#
#         Returns:
#             epoch_losses: list, losses for the epoch.
#             epoch_scores: dict, scores for the epoch as lists, mapped with the scores' names.
#         """
#
#         n_batches = len(data_loader)
#
#         epoch_losses, epoch_scores = [], defaultdict(list)
#         running_loss, running_score = [], defaultdict(list)
#
#         for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=n_batches):
#
#             batch_loss, batch_score = self.train_batch(inputs, targets, is_regression)
#
#             running_loss.append(batch_loss), dict_append(running_score, batch_score)
#
#             if (batch_idx + 1) % n_updates == 0:
#                 dict_remove_none(running_score)
#
#                 running_loss, running_score = mean(running_loss), dict_mean(running_score)
#                 epoch_losses.append(running_loss), dict_append(epoch_scores, running_score)
#
#                 self.write_tensorboard(running_loss, running_score, 'train', epoch * n_batches + batch_idx)
#
#                 running_loss, running_score = [], defaultdict(list)
#
#         return epoch_losses, epoch_scores
#
#     def train_batch(self, inputs, targets, is_regression):
#         """
#         Perform the training on a batch of features.
#
#         Args:
#             features: dict or torch.Tensor, features of the data.
#             targets: torch.Tensor, targets of the data.
#             is_regression: bool, whether to use the regression set up for the task.
#
#         Returns:
#             batch_loss: float, loss on the batch of data.
#             batch_score: dict, various scores of the batch of data.
#         """
#
#         self.optimizer.zero_grad()
#
#         outputs = self.net(features)
#
#         loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
#         loss = self.loss(outputs, loss_targets)
#
#         loss.backward()
#         self.optimizer.step()
#         batch_loss = loss.data.item()
#
#         ranks = ranking(outputs.detach())
#         batch_score = self.get_score(ranks, targets)
#
#         return batch_loss, batch_score
#
#     def test_batch(self, features, targets, is_regression):
#         """
#         Perform the test or validation on a batch of features.
#
#         Args:
#             features: dict or torch.Tensor, features of the data.
#             targets: torch.Tensor, targets of the data.
#             is_regression: bool, whether to use the regression set up for the task.
#
#         Returns:
#             batch_loss: float, loss on the batch of data.
#             batch_score: dict, various scores (float) of the batch of data.
#         """
#
#         outputs = self.net(features)
#
#         loss_targets = targets if not is_regression else targets.type(dtype=torch.float).reshape((-1, 1))
#         loss = self.loss(outputs.detach(), loss_targets)
#
#         batch_loss = loss.data.item()
#
#         ranks = ranking(outputs.detach())
#         batch_score = self.get_score(ranks, targets)
#
#         return batch_loss, batch_score
#
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
#             explanations: str, explanations of the prediction, optional.
#         """
#
#         self.eval_mode()
#         pred = self.net(features)
#         self.train_mode()
#
#         return pred, None
#
#     # region ML models
#
#     def update_lr_scheduler(self):
#         """ Performs a step of the learning rate scheduler if there is one. """
#
#         old_lr = self.optimizer.param_groups[0]['lr']
#         self.lr_scheduler.step()
#         new_lr = self.optimizer.param_groups[0]['lr']
#
#         print("Learning rate decreasing from %s to %s" % (old_lr, new_lr)) if old_lr != new_lr else None
#
#     def eval_mode(self):
#         """ Sets the model on evaluation mode if there is one. """
#
#         self.net.eval()
#
#     def train_mode(self):
#         """ Sets the model on training mode if there is one. """
#
#         self.net.train()
#
#         # endregion
#
#         # region Encoding methods
#
#         @staticmethod
#         def save_vocab_idx(s, vocab):
#             """
#             Saves the string in the vocabulary dictionary.
#
#             Args:
#                 s: str, element to retrieve.
#                 vocab: dict, corresponding vocabulary.
#             """
#
#             if s not in vocab:
#                 vocab[s] = len(vocab)
#
#         @staticmethod
#         def get_vocab_idx(s, vocab):
#             """
#             Returns the index of a string in the corresponding vocabulary dictionary.
#
#             Args:
#                 s: str, element to retrieve.
#                 vocab: dict, corresponding vocabulary.
#
#             Returns:
#                 int, index of the word in the dictionary.
#             """
#
#             return vocab[s] if s in vocab else len(vocab)
#
#         def get_bow(self, strings, vocab):
#             """
#             Returns the bow of the items, given the vocabulary, as a torch.Tensor.
#
#             Args:
#                 strings: iterable, strings to retrieve.
#                 vocab: dict, corresponding vocabulary.
#
#             Returns:
#                 torch.tensor, bow of the items in a line tensor.Tensor.
#             """
#
#             n = len(vocab) + 1
#             bow = torch.zeros(n, dtype=torch.float)
#
#             for s in strings:
#                 idx = self.get_vocab_idx(s, vocab)
#                 bow[idx] += 1.
#
#             return bow
#
#         def get_one_hot(self, strings, vocab):
#             """
#             Returns the one hot encoding of the items, given the vocabulary, as a torch.Tensor.
#
#             Args:
#                 strings: iterable, strings to retrieve.
#                 vocab: dict, corresponding vocabulary.
#
#             Returns:
#                 torch.tensor, one hot encoding of the items in a line tensor.Tensor.
#             """
#
#             n1, n2 = len(strings), len(vocab)
#             encoding = torch.zeros((n1, n2), dtype=torch.float)
#
#             for i, s in enumerate(strings):
#                 j = self.get_vocab_idx(s, vocab)
#                 encoding[i, j] += 1.
#
#             return encoding
#
#         # endregion
#
# class HalfBOWModel(MLModel):
#     """ Model that uses a 1-hot encoding for the choices and a BOW for the other words. """
#     model_name = 'half_bow'
#
#     def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
#                  vocab_frequency_range):
#         """
#         Initializes an instance of the Bag of Word Model.
#
#         Args:
#             scores_names: iterable, names of the scores to use, the first one being monitored during training.
#             relevance_level: int, minimum label to consider a choice as relevant.
#             net: nn.Module, neural net to train.
#             optimizer: torch.optimizer, optimizer for the neural net.
#             lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
#             loss: torch.nn.Loss, loss to use.
#             experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
#             vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
#         """
#
#         super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
#                          lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name)
#
#         self.vocab_frequency_range = vocab_frequency_range
#
#         self.choice_to_idx = dict()
#         self.word_to_idx = dict()
#         self.word_counts = defaultdict(int)
#
#     # region Learning methods
#
#     def preview(self, data_loader):
#         """
#         Preview the data for the model.
#
#         Args:
#             data_loader: list, pairs of (inputs, targets) batches.
#         """
#
#         print("Learning the vocabulary...")
#
#         for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
#
#             for choice in inputs['choices']:
#                 self.save_vocab_idx(choice, self.choice_to_idx)
#
#             for word in self.get_other_words(inputs):
#                 self.word_counts[word] += 1
#
#         for word, count in self.word_counts.items():
#             if self.vocab_frequency_range[0] <= count <= self.vocab_frequency_range[1]:
#                 self.save_vocab_idx(word, self.word_to_idx)
#
#         print("Input size: %d" % (len(self.choice_to_idx) + len(self.word_to_idx) + 1))
#
#     def features(self, inputs):
#         """
#         Computes the features of the inputs.
#
#         Args:
#             inputs: dict, inputs of the prediction.
#
#         Returns:
#             dict or torch.Tensor, features of the inputs.
#         """
#
#         n = len(inputs['choices'])
#
#         one_hot = self.get_one_hot(strings=inputs['choices'], vocab=self.choice_to_idx)
#
#         bow = self.get_bow(strings=self.get_other_words(inputs), vocab=self.word_to_idx)
#         bow = bow.expand((n, -1))
#
#         features = torch.cat((one_hot, bow), dim=1)
#
#         return features
#
#     # endregion
#
#
# class FullBOWModel(MLModel):
#     """ Model that uses a BOW for the choice and the other words. """
#     model_name = 'full_bow'
#
#     def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
#                  vocab_frequency_range):
#         """
#         Initializes an instance of the Bag of Word Model.
#
#         Args:
#             scores_names: iterable, names of the scores to use, the first one being monitored during training.
#             relevance_level: int, minimum label to consider a choice as relevant.
#             net: nn.Module, neural net to train.
#             optimizer: torch.optimizer, optimizer for the neural net.
#             lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
#             loss: torch.nn.Loss, loss to use.
#             experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
#             vocab_frequency_range: tuple, pair (min, max) for the frequency for a word to be taken into account.
#         """
#
#         super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
#                          lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name)
#
#         self.vocab_frequency_range = vocab_frequency_range
#
#         self.word_to_idx = dict()
#         self.word_counts = defaultdict(int)
#
#     # region Learning methods
#
#     def preview(self, data_loader):
#         """
#         Preview the data for the model.
#
#         Args:
#             data_loader: list, pairs of (inputs, targets) batches.
#         """
#
#         print("Learning the vocabulary...")
#
#         for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
#
#             for words in self.get_choices_words(inputs):
#                 for word in words:
#                     self.word_counts[word] += 1
#
#             for word in self.get_other_words(inputs):
#                 self.word_counts[word] += 1
#
#         for word, count in self.word_counts.items():
#             if self.vocab_frequency_range[0] <= count <= self.vocab_frequency_range[1]:
#                 self.save_vocab_idx(word, self.word_to_idx)
#
#         print("Input size: %d" % (len(self.word_to_idx) + 1))
#
#     def features(self, inputs):
#         """
#         Computes the features of the inputs.
#
#         Args:
#             inputs: dict, inputs of the prediction.
#
#         Returns:
#             dict or torch.Tensor, features of the inputs.
#         """
#
#         n = len(inputs['choices'])
#
#         choices_embedding = [self.get_bow(words, self.word_to_idx) for words in self.get_choices_words(inputs)]
#         choices_embedding = torch.stack(choices_embedding)
#
#         other_words = self.get_other_words(inputs)
#         other_embedding = self.get_bow(other_words, self.word_to_idx)
#         other_embedding = other_embedding.expand((n, -1))
#
#         features = torch.add(input=choices_embedding, other=other_embedding)
#
#         return features
#
#     # endregion
#
#
# class EmbeddingModel(MLModel):
#     """ Model that uses an average embedding both for the choice words and the context words. """
#     model_name = 'embedding_linear'
#
#     def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
#                  pretrained_model, pretrained_model_dim):
#         """
#         Initializes an instance of the linear Embedding Model.
#
#         Args:
#             scores_names: iterable, names of the scores to use, the first one being monitored during training.
#             relevance_level: int, minimum label to consider a choice as relevant.
#             net: nn.Module, neural net to train.
#             optimizer: torch.optimizer, optimizer for the neural net.
#             lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
#             loss: torch.nn.Loss, loss to use.
#             experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
#             pretrained_model: unknown, pretrained embedding or model.
#             pretrained_model_dim: int, size of the pretrained embedding or model.
#         """
#
#         super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
#                          lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name,
#                          pretrained_model=pretrained_model, pretrained_model_dim=pretrained_model_dim)
#
#     def features(self, inputs):
#         """
#         Computes the features of the inputs.
#
#         Args:
#             inputs: dict, inputs of the prediction.
#
#         Returns:
#             dict or torch.Tensor, features of the inputs.
#         """
#
#         n = len(inputs['choices'])
#
#         choices_embedding = torch.stack([self.get_average_embedding(words=words)
#                                          for words in self.get_choices_words(inputs)])
#
#         other_embedding = self.get_average_embedding(words=self.get_other_words(inputs))
#         other_embedding = other_embedding.expand((n, -1))
#
#         features = torch.cat((choices_embedding, other_embedding), dim=1)
#
#         return features
#
#
# class EmbeddingBilinearModel(MLModel):
#     """ Model that uses an average embedding both for the choice words and the context words. """
#     model_name = 'embedding_bilinear'
#
#     def __init__(self, scores_names, relevance_level, net, optimizer, lr_scheduler, loss, experiment_name,
#                  pretrained_model, pretrained_model_dim):
#         """
#         Initializes an instance of the bilinear Embedding Model.
#
#         Args:
#             scores_names: iterable, names of the scores to use, the first one being monitored during training.
#             relevance_level: int, minimum label to consider a choice as relevant.
#             net: nn.Module, neural net to train.
#             optimizer: torch.optimizer, optimizer for the neural net.
#             lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler for the neural net.
#             loss: torch.nn.Loss, loss to use.
#             experiment_name: str, name of the experiment to save (if None, doesn't save the results in Tensorboard).
#             pretrained_model: unknown, pretrained embedding or model.
#             pretrained_model_dim: int, size of the pretrained embedding or model.
#         """
#
#         super().__init__(scores_names=scores_names, relevance_level=relevance_level, net=net, optimizer=optimizer,
#                          lr_scheduler=lr_scheduler, loss=loss, experiment_name=experiment_name,
#                          pretrained_model=pretrained_model, pretrained_model_dim=pretrained_model_dim)
#
#     def features(self, inputs):
#         """
#         Computes the features of the inputs.
#
#         Args:
#             inputs: dict, inputs of the prediction.
#
#         Returns:
#             dict or torch.Tensor, features of the inputs.
#         """
#
#         n = len(inputs['choices'])
#
#         choices_embedding = torch.stack([self.get_average_embedding(words=words)
#                                          for words in self.get_choices_words(inputs)])
#
#         other_embedding = self.get_average_embedding(words=self.get_other_words(inputs))
#         other_embedding = other_embedding.expand((n, -1))
#
#         features1, features2 = choices_embedding, other_embedding
#
#         return features1, features2
#
# LATEX_CODE = ''
#
# SCORE_TO_LATEX_NAME = {
#     'average_precision': r'\shortstack{av.\\prec.}',
#     'precision_at_10': r'\shortstack{prec.\\@10}',
#     'precision_at_100': r'\shortstack{prec.\\@100}',
#     'recall_at_10': r'\shortstack{recall\\@10}',
#     'recall_at_100': r'\shortstack{recall\\@100}',
#     'reciprocal_best_rank': r'\shortstack{recip.\\best\\rank}',
#     'reciprocal_average_rank': r'\shortstack{recip.\\av.\\rank}',
#     'ndcg_at_10': r'\shortstack{ndcg\\@10}',
#     'ndcg_at_100': r'\shortstack{ndcg\\@100}'
# }
#
# MODEL_TO_LATEX_NAME = {
#     'RandomBaseline': 'random',
#     'FrequencyBaseline':  'frequency',
#     'SummariesCountBaseline':  r'\shortstack{summ.\\count}',
#     'SummariesUniqueCountBaseline':  r'\shortstack{summ.\\un. count}',
#     'SummariesOverlapBaseline':  r'\shortstack{summ.\\overlap}',
#     'SummariesAverageEmbeddingBaseline':  r'\shortstack{summ. av.\\embed.}',
#     'SummariesOverlapAverageEmbeddingBaseline':  r'\shortstack{summ. overlap\\av. embed.}',
#     'ActivatedSummariesBaseline':  r'\shortstack{activated\\summ.}',
#     'ContextCountBaseline':  r'\shortstack{cont.\\count}',
#     'ContextUniqueCountBaseline':  r'\shortstack{cont. un.\\ count}',
#     'ContextAverageEmbeddingBaseline':  r'\shortstack{cont. av.\\embed.}',
#     'SummariesContextCountBaseline':  r'\shortstack{summ. cont.\\count}',
#     'SummariesContextUniqueCountBaseline':  r'\shortstack{summ. cont.\\un. count}',
#     'SummariesOverlapContextBaseline':  r'\shortstack{summ.\\ overlap cont.}',
#     'SummariesContextAverageEmbeddingBaseline':  r'\shortstack{summ. cont.\\av. embed.}',
#     'SummariesOverlapContextAverageEmbeddingBaseline':  r'\shortstack{summ.\\overlap cont.\\av. embed.}'
# }
#
#
# def init_latex_code(scores_names):
#     """
#     Initializes the global variable LATEX_CODE.
#
#     Args:
#         scores_names: iterable, names of the scores to use, the first one being monitored during training.
#     """
#
#     global LATEX_CODE
#
#     LATEX_CODE = ''
#
#     line = r'\begin{tabular}{r|'
#     line += '|'.join(['c' for _ in range(len(scores_names))])
#     line += '}'
#     LATEX_CODE += line + '\n'
#
#     names = [SCORE_TO_LATEX_NAME[name] if name in SCORE_TO_LATEX_NAME else name for name in scores_names]
#     line = 'Method & '
#     line += ' & '.join(names)
#     line += r' \\ \hline'
#     LATEX_CODE += line + '\n'
#
#
# def update_latex_code(model, valid=True, test=False):
#     """
#     Updates the global variable LATEX_CODE with the scores of display_metrics of the model.
#
#     Args:
#         model: models.Model, model to evaluate.
#         valid: bool, whether or not to display the validation scores.
#         test: bool, whether or not to display the test scores.
#     """
#
#     global LATEX_CODE
#
#     assert not (valid and test) and (valid or test)
#
#     name = type(model).__name__
#     name = MODEL_TO_LATEX_NAME[name] if name in MODEL_TO_LATEX_NAME else name
#
#     latex_code_valid, latex_code_test = model.display_metrics(valid=valid, test=test, silent=True)
#     latex_code = latex_code_valid or latex_code_test
#
#     line = name + latex_code + r' \\'
#     LATEX_CODE += line + '\n'
#
#
# def end_latex_code():
#     """ Ends the global variable LATEX_CODE. """
#
#     global LATEX_CODE
#
#     LATEX_CODE += r'\end{tabular}'
#
#
# def display_latex_code():
#     """ Display the global variable LATEX_CODE. """
#
#     print(LATEX_CODE)
#
#
# def display_metrics_valid(self, valid, test):
#     """
#     Display the validation or test metrics of the model registered during the last experiment.
#
#     Args:
#         valid: bool, whether or not to display the validation scores.
#         test: bool, whether or not to display the test scores.
#         silent: bool, whether to print or not the results.
#
#     Returns:
#         latex_code_valid: str, code latex for the validation scores.
#         latex_code_test: str, code latex for the testing scores.
#     """
#
#     latex_code_valid, latex_code_test = '', ''
#
#     if valid:
#         print("\nScores evaluated on the validation set:") if not silent else None
#         score = {name: self.valid_scores[name][-1] for name in self.scores_names}
#
#         if not silent:
#             for name, s in score.items():
#                 print('%s: %.5f' % (name, s))
#
#         latex_code_valid = ' & ' + ' & '.join([str(round(score[name], 5))[1:] for name in self.scores_names])
#
#     if test:
#         print("\nScores evaluated on the test set:") if not silent else None
#         score = {name: self.test_scores[name][-1] for name in self.scores_names}
#
#         if not silent:
#             for name, s in score.items():
#                 print('%s: %.5f' % (name, s))
#
#         latex_code_test = ' & ' + ' & '.join([str(round(score[name], 5))[1:] for name in self.scores_names])
#
#     return latex_code_valid, latex_code_test
#
# def sort_batch(self, inputs, targets):
#     """
#     Returns lists of sub-batches of the inputs and targets where all the choices have the same number of words.
#
#     Args:
#         inputs: dict, inputs of the batch.
#         targets: targets: torch.Tensor, targets of the data.
#
#     Returns:
#         sub_inputs: list, inputs sub-batches as dictionaries.
#         sub_targets: list, targets sub-batches as torch.Tensors.
#     """
#
#     sorted_inputs, sorted_targets = dict(), dict()
#
#     choices = inputs['choices']
#     choices_words = self.get_choices_words(inputs)
#     other = {key: value for key, value in inputs.items() if key != 'choices'}
#
#     for i, choice in enumerate(choices):
#         n_words = len(choices_words[i])
#
#         if n_words not in sorted_inputs:
#             sorted_inputs[n_words] = {'choices': []}
#             sorted_inputs[n_words].update(other)
#             sorted_targets[n_words] = []
#
#         sorted_inputs[n_words]['choices'].append(choice)
#         sorted_targets[n_words].append(targets[i].data.item())
#
#     sub_inputs = [item for _, item in sorted_inputs.items()]
#     sub_targets = [torch.tensor(item) for _, item in sorted_targets.items()]
#
#     return sub_inputs, sub_targets
#
#     def get_bert_tokenize(self, text, segment):
#         """
#         Returns the tokens and segments lists (to tensorize) of the string text.
#
#         Args:
#             text: str, part of BERT input.
#             segment: int, index (0 or 1) for first or second part of the BERT input.
#
#         Returns:
#             tokens_tensor: torch.Tensor, indexes of the tokens.
#             segments_tensor: torch.Tensor, indexes of the segments.
#         """
#
#         text = '[CLS] ' + text + ' [SEP]' if segment == 0 else text + ' [SEP]'
#
#         tokenized_text = self.tokenizer.tokenize(text)
#         length = len(tokenized_text)
#
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
#         segments_ids = [segment for _ in range(length)]
#
#         tokens_tensor = torch.tensor([indexed_tokens])
#         segments_tensor = torch.tensor([segments_ids])
#
#         return tokens_tensor, segments_tensor
#
#     def get_bert_embedding(self, tokens_tensor, segments_tensor):
#         """
#         Returns the BERT embedding of the text.
#
#         Args:
#             tokens_tensor: torch.Tensor, representation of the tokens.
#             segments_tensor: torch.Tensor, representation of the segments.
#
#         Returns:
#             torch.Tensor, embedding of the tokens.
#         """
#
#         with torch.no_grad():
#             encoded_layer, _ = self.pretrained_model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)
#
#         return encoded_layer[:, 0, :]
#
#     def get_bert_nsp_logits(self, tokens_tensor, segments_tensor):
#         """
#         Returns the BERT embedding of the text.
#
#         Args:
#             tokens_tensor: torch.Tensor, representation of the tokens.
#             segments_tensor: torch.Tensor, representation of the segments.
#
#         Returns:
#             torch.Tensor, embedding of the tokens.
#         """
#
#         with torch.no_grad():
#             logits = self.pretrained_model(tokens_tensor, segments_tensor)
#
#         return logits
#
#     def get_bert_similarity(self, words_lists, words):
#         """
#         Returns the similarities between the BERT embeddings of the lists of words of words_lists and the average
#         embedding of words.
#
#         Args:
#             words_lists: list, first words to compare, as a list of list of words.
#             words: list, second words to compare, as a list of words.
#
#         Returns:
#             torch.tensor, similarities in a column tensor.
#         """
#
#         pass
#
#
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
#
#
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
