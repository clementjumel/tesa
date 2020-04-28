from modeling.utils import *
from modeling import metrics

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
from os.path import join as path_join
import matplotlib.pyplot as plt
import torch


class BaseModel:
    """ Model structure. """

    def __init__(self, args, pretrained_model):
        """
        Initializes an instance of Model.

        Args:
            args: argparse.ArgumentParser, arguments passed to the script.
            pretrained_model: unknown, pretrained word embedding or BART.
        """

        self.scores_names = args.scores_names
        self.max_context_size = args.max_context_size
        self.context_format = args.context_format
        self.targets_format = args.targets_format
        self.pretrained_model = pretrained_model

        self.train_losses, self.train_scores = [], defaultdict(list)
        self.valid_losses, self.valid_scores = [], defaultdict(list)
        self.test_losses, self.test_scores = [], defaultdict(list)

        self.punctuation = str_punctuation
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.writer = None

        if args.tensorboard:
            root = args.root
            tensorboard_logs_path = args.tensorboard_logs_path
            model_name = self.__class__.__name__
            model_name = "_".join([word.lower() for word in findall(r'[A-Z][^A-Z]*', model_name)])

            self.writer = SummaryWriter(path_join(root, tensorboard_logs_path, model_name))

        seed(args.random_seed), torch.manual_seed(args.random_seed)

    # region Training pipeline methods

    def play(self, task, args):
        """
        Performs the preview and the evaluation of a model on the task.

        Args:
            task: modeling_task.ModelingTask, task to evaluate the model on.
            args: argparse.ArgumentParser, arguments passed to the script.
        """

        show = args.show
        show_rankings = args.show_rankings
        show_choices = args.show_choices

        self.preview(task.train_loader)

        print("Evaluation on the train_loader...")
        self.valid(task.train_loader)

        print("Evaluation on the valid_loader...")
        self.valid(task.valid_loader)

        if show:
            self.show(task, show_rankings=show_rankings, show_choices=show_choices)

    def show(self, task, show_rankings, show_choices):
        """
        Show the model results on different rankings.

        Args:
            task: modeling_task.ModelingTask, task to evaluate the model on.
            show_rankings: int, number of rankings to show.
            show_choices: int, number of best choices to show.
        """

        data_loader = task.valid_loader
        shuffle(data_loader)

        for ranking_idx, ranking in enumerate(data_loader[:show_rankings]):
            ranking_choices, ranking_outputs, ranking_targets = [], [], []

            for inputs, targets in ranking:
                outputs = self.pred(inputs)

                ranking_choices.extend(inputs['choices'])
                ranking_outputs.append(outputs)
                ranking_targets.append(targets)

            ranking_outputs, ranking_targets = torch.cat(ranking_outputs), torch.cat(ranking_targets)

            ranking_ranks = get_ranks(ranking_outputs)
            batch_score = self.get_score(ranks=ranking_ranks, targets=ranking_targets)

            print("Ranking %i/%i: " % (ranking_idx + 1, show_rankings))

            inputs, _ = ranking[0]
            print("Entities (%s): %s" % (inputs['entities_type'], ', '.join(inputs['entities'])))

            if self.context_format is not None:
                print("Context:\n%s" % format_context(inputs, context_format=self.context_format,
                                                      max_context_size=self.max_context_size))

            print("\nScores:")
            for score_name in self.scores_names:
                print("%s: %.3f" % (score_name, batch_score[score_name]))

            results = zip(ranking_ranks.squeeze().tolist(),
                          ranking_choices,
                          ranking_outputs.squeeze().tolist(),
                          ranking_targets.squeeze().tolist())
            results = sorted(results, key=lambda x: x[0])

            print("\nGold standards (rank: choice (output/target):")
            for result in results:
                if result[3]:
                    print("%i: %s (%.2f/%i)" % result)

            print("\nTop %i results (rank: choice (output/target):" % show_choices)
            for result in results[:show_choices]:
                print("%i: %s (%.2f/%i)" % result)

            print()

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
                                           targets=targets.clone())

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
        """ Returns the words from the inputs' choices as a list of list (or sets, if setten). """

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
        """ Returns the words from the inputs' entities as a list of list (or a list, if flatten). """

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
        """ Returns the words from the inputs' NYT titles and contexts as a list (or set, if setten). """

        context_words = []

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_words.extend(self.get_words(s=nyt_title,
                                                remove_stopwords=remove_stopwords,
                                                remove_punctuation=remove_punctuation,
                                                lower=lower,
                                                lemmatize=lemmatize))

            context_words.extend(self.get_words(s=nyt_context,
                                                remove_stopwords=remove_stopwords,
                                                remove_punctuation=remove_punctuation,
                                                lower=lower,
                                                lemmatize=lemmatize))

        if setten:
            context_words = set(context_words)

        return context_words

    @staticmethod
    def get_wikis_str(inputs):
        """ Returns a string with the wikipedia articles. """

        return ' '.join([wiki_article for wiki_article in inputs['wiki_articles'] if wiki_article])

    def get_wikis_words(self, inputs, remove_stopwords=True, remove_punctuation=True, lower=True, lemmatize=True,
                        flatten=False, setten=False):
        """ Return the words from the inputs' wikipedia articles as a list of list (or a list if flatten, or a list of
        set if setten, or a set if both). """

        wikis_words = []

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                wikis_words.append(self.get_words(s=wiki_article,
                                                  remove_stopwords=remove_stopwords,
                                                  remove_punctuation=remove_punctuation,
                                                  lower=lower,
                                                  lemmatize=lemmatize))
            else:
                wikis_words.append([])

        if flatten:
            wikis_words = [word for words in wikis_words for word in words]
            if setten:
                wikis_words = set(wikis_words)

        else:
            if setten:
                wikis_words = [set(words) for words in wikis_words]

        return wikis_words

    def get_other_words(self, inputs, setten=False):
        """ Return the words from the inputs' entities, NYT articles and wikipedia articles, in a list (or a set, if
        setten). """

        entities_words = self.get_entities_words(inputs, flatten=True)
        context_words = self.get_context_words(inputs)
        wikis_words = self.get_wikis_words(inputs, flatten=True)

        other_words = entities_words + context_words + wikis_words

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

        return torch.tensor(self.pretrained_model[word]) if word in self.pretrained_model.vocab else None

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
            word = list(self.pretrained_model.vocab)[0]
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

    # region Plot methods

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

    # endregion


# region Baselines

# region Simple Baselines

class Random(BaseModel):
    """ Baseline with random predictions. """

    def pred(self, inputs):
        return torch.rand(len(inputs['choices'])).reshape((-1, 1))


class Frequency(BaseModel):
    """ Baseline based on answers' overall frequency. """

    def __init__(self, args, pretrained_model):
        super().__init__(args=args, pretrained_model=pretrained_model)

        self.counts = defaultdict(int)

    def preview(self, data_loader):
        print("Learning answers counts...")

        for ranking_task in tqdm(data_loader, total=len(data_loader)):
            for inputs, targets in ranking_task:
                for choice, target in zip(inputs['choices'], targets):
                    self.counts[choice] += target.data.item()

    def pred(self, inputs):
        grades = [self.counts[choice] if choice in self.counts else 0 for choice in inputs['choices']]
        return torch.tensor(grades).reshape((-1, 1))

# endregion


# region Summaries-dependent baselines

class SummariesCount(BaseModel):
    """ Baseline based on the count of words of the summaries that are in the choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikis_words = self.get_wikis_words(inputs, flatten=True)

        return self.get_lists_counts(words_lists=choices_words, words=wikis_words)


class SummariesUniqueCount(BaseModel):
    """ Baseline based on the count of unique words of all the summaries that are in choice. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)
        wikis_words = self.get_wikis_words(inputs, flatten=True, setten=True)

        return self.get_sets_counts(words_sets=choices_words, words=wikis_words)


class SummariesOverlap(BaseModel):
    """ Baseline based on the count of words from choice that are in the overlap of all the summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs, setten=True)

        wikis_words = [wiki_words for wiki_words in self.get_wikis_words(inputs, setten=True) if wiki_words]
        wikis_words = set.intersection(*wikis_words) if wikis_words else set()

        return self.get_sets_counts(words_sets=choices_words, words=wikis_words)


class ActivatedSummaries(BaseModel):
    """ Baseline based on the number of summaries that have words matching the answer. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikis_words = self.get_wikis_words(inputs)

        activated_wikis = [[set(choice_words).intersection(set(wiki_words)) for wiki_words in wikis_words]
                           for choice_words in choices_words]
        activated_wikis = [[wiki for wiki in activated_wiki if wiki] for activated_wiki in activated_wikis]

        return torch.tensor([len(wikis) for wikis in activated_wikis]).reshape((-1, 1))


class SummariesAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and all the summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)
        wikis_words = self.get_wikis_words(inputs, flatten=True)

        return self.get_average_embedding_similarity(words_lists=choices_words, words=wikis_words)


class SummariesOverlapAverageEmbedding(BaseModel):
    """ Baseline with predictions based on the average embedding proximity between the choice and the overlap of the
    summaries. """

    def pred(self, inputs):
        choices_words = self.get_choices_words(inputs)

        wikis_words = [wiki_words for wiki_words in self.get_wikis_words(inputs, setten=True) if wiki_words]
        wikis_words = set.intersection(*wikis_words) if wikis_words else set()

        return self.get_average_embedding_similarity(words_lists=choices_words, words=wikis_words)

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

        wikis_words = [wiki_words for wiki_words in self.get_wikis_words(inputs, setten=True) if wiki_words]
        wikis_words = set.intersection(*wikis_words) if wikis_words else set()
        wikis_words.update(self.get_context_words(inputs, setten=True))

        return self.get_sets_counts(words_sets=choices_words, words=wikis_words)


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

        other_words = [wiki_words for wiki_words in self.get_wikis_words(inputs, setten=True) if wiki_words]
        other_words = set.intersection(*other_words) if other_words else set()
        other_words.update(self.get_context_words(inputs, setten=True))

        return self.get_average_embedding_similarity(words_lists=choices_words, words=other_words)

# endregion

# endregion


# region BART

class ClassifierBart(BaseModel):
    """ BART finetuned as a classifier between aggregation and not_aggregation. """

    def __init__(self, args, pretrained_model):
        super().__init__(args=args, pretrained_model=pretrained_model)

        bart = self.pretrained_model
        labels = [bart.task.label_dictionary.string([torch.tensor([0]) + bart.task.label_dictionary.nspecial]),
                  bart.task.label_dictionary.string([torch.tensor([1]) + bart.task.label_dictionary.nspecial])]

        self.idx = labels.index("aggregation")

    def pred(self, inputs):
        sentence1 = format_context(inputs, context_format=self.context_format, max_context_size=self.max_context_size)

        batch_encoding = [self.pretrained_model.encode(sentence1, sentence2) for sentence2 in inputs['choices']]
        batch_tokens = collate_tokens(batch_encoding, pad_idx=1)

        with torch.no_grad():
            logprobs = self.pretrained_model.predict('sentence_classification_head', batch_tokens)

        return logprobs[:, self.idx].reshape((-1, 1))


class GeneratorBart(BaseModel):
    """ BART finetuned as a generator of aggregation. """

    def __init__(self, args, pretrained_model):
        super().__init__(args=args, pretrained_model=pretrained_model)

        self.pretrained_model.half()

        self.beam = args.bart_beam
        self.lenpen = args.bart_lenpen
        self.max_len_b = args.bart_len_b
        self.min_len = args.bart_min_len
        self.no_repeat_ngram_size = args.bart_no_repeat_ngram_size
        self.results_path = args.results_path

    def play(self, task, args):
        print("Evaluation on the train_loader...")
        self.bart_valid(task.train_loader, fname="train")

        print("Evaluation on the valid_loader...")
        self.bart_valid(task.valid_loader, fname="valid")

    def bart_valid(self, data_loader, fname):
        """
        Generate the hypothesis of BART on the data_loader and write them in some files.

        Args:
            data_loader: list, list of ranking tasks, which are lists of (inputs, targets) batches.
            fname: str, name of the file to write in.
        """

        n_rankings = len(data_loader)
        shuffle(data_loader)

        for idx, ranking in tqdm(enumerate(data_loader), total=n_rankings):
            source = format_context(ranking, context_format=self.context_format, max_context_size=self.max_context_size)
            targets = format_targets(ranking, targets_format=self.targets_format)
            entities = ranking[0][0]['entities']

            with torch.no_grad():
                hypotheses = self.pretrained_model.sample([source],
                                                          beam=self.beam,
                                                          lenpen=self.lenpen,
                                                          max_len_b=self.max_len_b,
                                                          min_len=self.min_len,
                                                          no_repeat_ngram_size=self.no_repeat_ngram_size)[0]

            with open(path_join(self.results_path, fname + ".source", 'a')) as source_file:
                source_file.write(str(idx) + ' - ' + source + '\n')
            with open(path_join(self.results_path, fname + ".targets", 'a')) as targets_file:
                targets_file.write(str(idx) + ' - ' + ', '.join(targets) + '\n')
            with open(path_join(self.results_path, fname + ".entities", 'a')) as entities_file:
                entities_file.write(str(idx) + ' - ' + ', '.join(entities) + '\n')
            with open(path_join(self.results_path, fname + ".hypotheses", 'a')) as hypotheses_file:
                hypotheses_file.write(str(idx) + ' - ' + ', '.join(["%s [%.3f]" % (hypo[0], 2 ** hypo[1])
                                                                    for hypo in hypotheses]) + '\n')

# endregion
