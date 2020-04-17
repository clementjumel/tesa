from numpy.random import shuffle, choice
from numpy import asarray, arange, split
import torch


class RankingTask:
    """ Class for a single Ranking Task. """

    def __init__(self, queries, labelled_answers, ranking_size, batch_size):
        """
        Initializes the RankingTask instance.

        Args:
            queries: list, initial Queries of the annotation (corresponding to different NYT articles).
            labelled_answers: dict, answers and their labels (0 for negative answers).
            ranking_size: int, number of choices to compute for each ranking task.
            batch_size: int, number of samples in each batch.
        """

        self.ranking_size = ranking_size
        self.batch_size = batch_size

        self.entities = self.get_entities(queries)
        self.entities_type = self.get_entities_type(queries)
        self.wiki_articles = self.get_wiki_articles(queries)
        self.nyt_titles = self.get_nyt_titles(queries)
        self.nyt_contexts = self.get_nyt_contexts(queries)

        labelled_answers = self.filter_answers(labelled_answers)
        self.choices, self.labels = self.get_choices_labels(labelled_answers)

    # region Methods get_

    @staticmethod
    def get_entities(queries):
        """ Return the entities of the RankingTask as a list of str. """

        assert len(set([', '.join(query.entities) for query in queries])) == 1
        return queries[0].entities

    @staticmethod
    def get_entities_type(queries):
        """ Return the entities_type of the RankingTask as a str. """

        assert len(set([query.entities_type_ for query in queries])) == 1
        return queries[0].entities_type_

    def get_wiki_articles(self, queries):
        """ Return the wikipedia articles of the RankingTask as a list of str ('' if no information is found). """

        assert len(set([', '.join(query.summaries) for query in queries])) == 1

        wiki_dict = queries[0].summaries
        for entity in self.entities:
            if wiki_dict[entity] == "No information found.":
                wiki_dict[entity] = ''

        return [wiki_dict[entity] for entity in self.entities]

    @staticmethod
    def get_nyt_titles(queries):
        """ Return the NYT titles of the RankingTask as a list of str. """

        return [query.title for query in queries]

    @staticmethod
    def get_nyt_contexts(queries):
        """ Return the NYT contexts of the RankingTask as a list of str. """

        return [query.get_context_readable() for query in queries]

    @staticmethod
    def get_choices_labels(labelled_answers):
        """ Return the choices of the RankingTask as a (shuffled) list of str and their associated labels as a list of
        int. """

        labelled_answers = [(answer, label) for answer, label in labelled_answers.items()]
        labelled_answers = sorted(labelled_answers)
        shuffle(labelled_answers)

        choices = [labelled_answer[0] for labelled_answer in labelled_answers]
        labels = [labelled_answer[1] for labelled_answer in labelled_answers]

        return choices, labels

    # endregion

    # region Other methods

    def filter_answers(self, labelled_answers):
        """ Return labelled_answers limited to self.ranking_size number of possible answers. If self.ranking_size is
        None, don't do anything; is it is 0, compute only the positive answers (for generation task). """

        if self.ranking_size is not None:
            negative_answers = sorted([key for key, value in labelled_answers.items() if value == 0])
            labelled_answers = {key: value for key, value in labelled_answers.items() if value > 0}

            if self.ranking_size:
                n = len(labelled_answers)
                if n > self.ranking_size:
                    raise Exception("Too small ranking size, some answers will be lost (should be at least %i)." % n)

                negative_answers = {answer: 0 for answer in choice(a=negative_answers,
                                                                   size=self.ranking_size - n,
                                                                   replace=False)}

                labelled_answers.update(negative_answers)

        return labelled_answers

    def input_batches(self):
        """ Returns the input batches of the RakingTask as a list of dict (one dict of each batch). """

        generic_inputs = {'entities': self.entities,
                          'entities_type': self.entities_type,
                          'wiki_articles': self.wiki_articles,
                          'nyt_titles': self.nyt_titles,
                          'nyt_contexts': self.nyt_contexts}

        idxs = arange(self.batch_size, len(self.choices), self.batch_size)
        input_batches = [{'choices': list(choices)} for choices in split(asarray(self.choices), idxs)]

        for input_batch in input_batches:
            input_batch.update(generic_inputs)

        shuffle(input_batches)
        return input_batches

    def target_batches(self):
        """ Returns the target batches of the RankingTask in a list of line torch.Tensors of type torch.long. """

        idxs = arange(self.batch_size, len(self.labels), self.batch_size)
        target_batches = [torch.tensor(labels, dtype=torch.long) for labels in split(asarray(self.labels), idxs)]

        shuffle(target_batches)
        return target_batches

    # endregion
