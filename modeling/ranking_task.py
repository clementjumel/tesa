from numpy.random import shuffle, choice
import torch


class RankingTask:
    """ Class for a single Ranking Task. """

    def __init__(self, queries, labelled_answers, ranking_size):
        """
        Initializes the RankingTask instance.

        Args:
            queries: list, initial Queries of the annotation (corresponding to different NYT articles).
            labelled_answers: dict, answers and their labels (0 for negative answers).
            ranking_size: int, number of choices to compute for each ranking task.
        """

        self.entities = self.get_entities(queries)
        self.entities_type_ = self.get_entities_type(queries)
        self.wiki_articles = self.get_wiki_articles(queries)
        self.nyt_titles = self.get_nyt_titles(queries)
        self.nyt_contexts = self.get_nyt_contexts(queries)

        labelled_answers = self.filter_answers(labelled_answers=labelled_answers, ranking_size=ranking_size)

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

    @staticmethod
    def filter_answers(labelled_answers, ranking_size):
        """ Return labelled_answers limited to self.ranking_size number of possible answers. If self.ranking_size is
        None, don't do anything; is it is 0, compute only the positive answers (for generation task). """

        if ranking_size is not None:
            negative_answers = sorted([key for key, value in labelled_answers.items() if value == 0])
            labelled_answers = {key: value for key, value in labelled_answers.items() if value > 0}

            if ranking_size:
                n = len(labelled_answers)
                if n > ranking_size:
                    raise Exception("Too small ranking size, some answers will be lost (should be at least %i)." % n)

                negative_answers = {answer: 0 for answer in choice(a=negative_answers,
                                                                   size=ranking_size - n,
                                                                   replace=False)}

                labelled_answers.update(negative_answers)

        return labelled_answers

    def inputs(self):
        """ Returns the inputs of the RakingTask in a dict. """

        return {'choices': self.choices,
                'entities': self.entities,
                'entities_type': self.entities_type_,
                'wiki_articles': self.wiki_articles,
                'nyt_titles': self.nyt_titles,
                'nyt_contexts': self.nyt_contexts}

    def targets(self):
        """ Returns the targets of the RankingTask in a line torch.Tensor of type torch.long. """

        return torch.tensor(self.labels, dtype=torch.long)

    # endregion
