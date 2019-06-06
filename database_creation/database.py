from database_creation.utils import BaseClass
from database_creation.article import Article

from copy import copy
from numpy import random
from glob import glob
from collections import defaultdict
from numpy import histogram

import matplotlib.pyplot as plt


class Database(BaseClass):
    # region Class initialization

    to_print = ['articles']
    print_attributes, print_lines, print_offsets = False, 2, 0
    limit_print, random_print = 50, True
    count_modulo = 1000

    @BaseClass.Verbose("Initializing the database...")
    def __init__(self, max_size=None, root='databases/nyt_jingyun', year='2000'):
        """
        Initializes an instance of Database.

        Args:
            max_size: int, maximum number of articles in the database; if None, takes all articles.
            root: str, relative path to the root of the project.
            year: str, year of the database to analyse.
        """

        self.max_size = max_size
        self.root = root
        self.year = str(year)

        self.articles = None
        self.size = None

        self.tuples = None
        self.tuples_ids = None
        self.stats = None

        self.compute_articles()
        self.compute_size()

        self.clean(Article.criterion_data)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Database.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets, limit_print, random_print = self.get_parameters()[:6]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'articles' else ''
            string += self.prefix(print_attribute, print_lines if string else 0, print_offsets, attribute) + s if s \
                else ''

        ids = self.get_ids()
        if random_print:
            random.shuffle(ids)

        string += self.prefix(True, print_lines if string else 0, print_offsets, 'articles') if print_attribute else ''

        count = 0
        for id_ in ids:
            s = self.to_string(self.articles[id_])

            if s:
                string += self.prefix(print_attribute, print_lines if string else 0, print_offsets,
                                      'article ' + str(id_)) + s
                count += 1
                if count == limit_print:
                    break

        return string

    @classmethod
    def get_parameters(cls):
        """
        Fetch the print attributes of the class.

        Returns:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            cls.limit_print: int, limit number of articles to print; if -1, prints all.
            cls.random_print: bool, whether or not to select randomly the articles printed.
        """

        to_print, print_attribute, print_lines, print_offsets = super(Database, cls).get_parameters()

        return to_print, print_attribute, print_lines, print_offsets, cls.limit_print, cls.random_print

    @classmethod
    def set_parameters(cls, to_print=None, print_attribute=None, print_lines=None, print_offsets=None, limit_print=None,
                       random_print=None):
        """
        Changes the parameter attributes of the class.

        Args:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            limit_print: int, limit number of articles to print; if -1, prints all.
            random_print: bool, whether or not to select randomly the articles printed.
        """

        super(Database, cls).set_parameters(to_print=to_print, print_attribute=print_attribute, print_lines=print_lines,
                                            print_offsets=print_offsets)

        cls.limit_print = limit_print if limit_print is not None else cls.limit_print
        cls.random_print = random_print if random_print is not None else cls.random_print

    # endregion

    # region Main methods

    @BaseClass.Verbose("Preprocessing the database...")
    def preprocess_database(self):
        """ Performs the preprocessing of the database. """

        self.compute_entities()
        self.compute_tuples()
        self.clean(Database.criterion_tuples_ids)

    @BaseClass.Verbose("Preprocessing the articles...")
    def preprocess_articles(self):
        """ Performs the preprocessing of the articles. """

        count = 0
        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].preprocess()

    @BaseClass.Verbose("Processing the articles (find aggregation candidates)...")
    def process_candidates(self):
        """ Performs the processing of the database by computing the aggregation candidates. """

        count = 0
        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].process_candidates()

    @BaseClass.Verbose("Processing the articles (compute frequent entity tuples contexts)...")
    def process_contexts(self):
        """ Performs the processing of the database by computing the frequent entity tuples contexts. """

        count = 0
        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].process_contexts()

    @BaseClass.Verbose("Computing the entities tuples statistics...")
    def stats_tuples(self):
        """ Compute and display the entities tuples statistics of the database. """

        self.compute_stats_tuples()
        self.display_stats_tuples(print_data=True)

    @BaseClass.Verbose("Computing the contexts statistics...")
    def stats_contexts(self):
        """ Compute and display the contexts statistics of the database. """

        self.compute_stats_contexts()
        self.display_stats_contexts(print_data=True)

    @BaseClass.Verbose("Filtering the articles according to a threshold...")
    @BaseClass.Attribute('tuples', True)
    def filter_threshold(self, threshold):
        """
        Filter out the articles that doesn't respect the specified threshold on the entities tuple frequency.

        Args:
            threshold: int, minimal number of articles an entities tuple must appear in to be considered.
        """

        tuples, tuples_ids = [], set()

        for entities_tuple in self.tuples:
            if entities_tuple['frequency'] >= threshold:
                tuples.append(entities_tuple)
                tuples_ids.update(entities_tuple['ids'])

        self.tuples = tuples
        self.tuples_ids = tuples_ids

        self.clean(Database.criterion_tuples_ids)

    # TODO: finish
    def display_tuple(self, idx):

        entity_tuple = self.tuples[idx]['tuple_']
        ids = self.tuples[idx]['ids']

        length = 0

        print("Entity tuples: {}\n\n".format(self.to_string(entity_tuple)))

        for id_ in ids:
            article = self.articles[id_]

            for type_ in article.contexts:
                contexts = article.contexts[type_][entity_tuple]
                length += len(contexts)

                print(self.to_string(contexts) + '\n\n')

        print("\n{} samples out of {} articles".format(length, len(ids)))

    @BaseClass.Verbose("Writing the candidates...")
    def write_candidates(self, file_name):
        """
        Writes the candidates of the database. Overwrites an existing file.

        Args:
            file_name: str, name of the file (with folder and extension).
        """

        count = 0
        with open(file_name, "w+") as f:
            for id_ in self.articles:
                count = self.progression(count)
                self.articles[id_].write_candidates(f)

    # endregion

    # region Methods compute_

    def compute_articles(self):
        """ Computes and initializes the articles in the database. """

        articles = {}

        for original_path in self.paths(pattern=self.root + '/data/' + self.year + '/*/*/*.xml',
                                        limit=self.max_size):
            id_ = original_path.split('/')[-1].split('.')[0]
            annotated_path = self.root + '/content_annotated/' + self.year + 'content_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(original_path=original_path, annotated_path=annotated_path)

        self.articles = articles

    def compute_size(self):
        """ Computes the number of articles in the database. """

        self.size = len(self.get_ids())

    def compute_entities(self):
        """ Compute the entities of the articles. """

        for id_ in self.articles:
            self.articles[id_].compute_entities()

    def compute_tuples(self):
        """ Compute the entities tuples of the database as a sorted list of dictionaries with the tuple, the frequency
        (in number of articles) and the ids of the articles. """

        tuples_ids = defaultdict(set)

        for id_ in self.articles:
            for entity_type in ['location', 'person', 'organization']:
                entities = getattr(self.articles[id_], 'entities')[entity_type]

                if len(entities) >= 2:
                    for t in self.subtuples(entities):
                        tuples_ids[t].add(id_)

        sorted_tuples = sorted(tuples_ids, key=lambda k: len(tuples_ids[k]), reverse=True)
        tuples = [{'tuple_': t, 'frequency': len(tuples_ids[t]), 'ids': tuples_ids[t]} for t in sorted_tuples]
        tuples_ids = set([id_ for tuple_ in tuples for id_ in tuple_['ids']])

        self.tuples = tuples
        self.tuples_ids = tuples_ids

    def compute_stats_tuples(self):
        """ Compute the entities tuples statistics of the database. """

        if self.stats is None:
            self.stats = dict()

        self.stats['tuples_lengths'] = self.stat_tuples_lengths()
        self.stats['tuples_frequencies'] = self.stat_tuples_frequencies()
        self.stats['tuples_thresholds'] = self.stat_tuples_thresholds()

    def compute_stats_contexts(self):
        """ Compute the contexts statistics of the database. """

        if self.stats is None:
            self.stats = dict()

        self.stats['contexts_same_sent'] = self.stat_contexts('same_sent')
        self.stats['contexts_neigh_sent'] = self.stat_contexts('neigh_sent')
        self.stats['contexts_same_role'] = self.stat_contexts('same_role')

    # endregion

    # region Methods get_

    def get_ids(self):
        """
        Computes the IDs of the articles in the database.

        Returns:
            list, IDs of the articles.
        """

        return list(self.articles.keys())

    # endregion

    # region Methods criterion_

    def criterion_tuples_ids(self, id_):
        """
        Check if an article does not belong to the tuples ids.

        Args:
            id_: string, id of the article to analyze.

        Returns:
            bool, True iff the article does not belong to the tuples ids.
        """

        return True if id_ not in self.tuples_ids else False

    # endregion

    # region Methods stat_

    def stat_tuples_lengths(self):
        """
        Compute the histogram of the lengths of the tuples as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the lengths of the entities tuples, starting from 0.
        """

        data = [len(entities_tuple['tuple_']) for entities_tuple in self.tuples]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_tuples_frequencies(self):
        """
        Compute the histogram of the frequencies of the tuples as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        data = [entities_tuple['frequency'] for entities_tuple in self.tuples]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_tuples_thresholds(self):
        """
        Compute the histogram of the size of the database corresponding to each threshold over the entities tuples
        frequency, starting from 0 (no threshold), as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the number of articles for each threshold.
        """

        m = max([entities_tuple['frequency'] for entities_tuple in self.tuples])

        threshold_ids = [set() for _ in range(m + 1)]
        threshold_ids[0].update(set(self.get_ids()))

        for entities_tuple in self.tuples:
            for threshold in range(1, entities_tuple['frequency'] + 1):
                threshold_ids[threshold].update(entities_tuple['ids'])

        data = [i for i in range(m + 1) for _ in threshold_ids[i]]
        bins = m + 1
        range_ = (0, m + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_contexts(self, type_):
        """
        Compute the histogram of the number of contexts for the specified type_, as a numpy.histogram.

        Args:
            type_: str, type of the context, must be 'same_sent', 'neigh_sent', or 'same_role'.

        Returns:
            numpy.histogram, histogram of the number of contexts.
        """

        data = []
        for entities_tuple in self.tuples:
            length = 0
            for id_ in entities_tuple['ids']:
                length += len(self.articles[id_].contexts[type_][entities_tuple['tuple_']])

            data.append(length)

        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    # endregion

    # region Methods display_

    def display_stats_tuples(self, print_data):
        """
        Display the entities tuples statistics of the database.

        Args:
            print_data: bool, whether to print the data or not.
        """

        print("\nTotal number of tuples: {}".format(len(self.tuples)))
        print("10 most frequent tuples:")
        for entities_tuple in self.tuples[:10]:
            print("{} (in {} articles)".format(self.to_string(entities_tuple['tuple_']), entities_tuple['frequency']))
        print('\n')

        self.plot_hist(fig=1, data=self.stats['tuples_lengths'], title='Lengths of the entities tuples',
                       xlabel='lengths', log=True, print_data=print_data)

        self.plot_hist(fig=2, data=self.stats['tuples_frequencies'], title='Frequencies of the entities tuples',
                       xlabel='frequencies', log=True, print_data=print_data)

        self.plot_hist(fig=3, data=self.stats['tuples_thresholds'], title='Number of articles for each threshold',
                       xlabel='thresholds', log=True, print_data=print_data)

        print('\n')

    def display_stats_contexts(self, print_data):
        """
        Display the contexts statistics of the database.

        Args:
            print_data: bool, whether to print the data or not.
        """

        print('\n')

        self.plot_hist(fig=4, data=self.stats['contexts_same_sent'], title='Number of same-sentence contexts',
                       xlabel='contexts size', log=True, print_data=print_data)

        self.plot_hist(fig=5, data=self.stats['contexts_neigh_sent'], title='Number of neighboring-sentences contexts',
                       xlabel='contexts size', log=True, print_data=print_data)

        self.plot_hist(fig=6, data=self.stats['contexts_same_role'], title='Number of same-role contexts',
                       xlabel='contexts size', log=True, print_data=print_data)

        print('\n')

    # endregion

    # region Other methods

    @BaseClass.Verbose("Cleaning the database...")
    @BaseClass.Attribute('size')
    def clean(self, criterion):
        """
        Removes from the database the articles which meets the Article's or Database's criterion.

        Args:
            criterion: function, method from Article or Database, criterion that an article must meet to be removed.
        """

        to_del = []

        if criterion.__module__.split('.')[-1] == 'article':
            for id_ in self.articles:
                if criterion(self.articles[id_]):
                    to_del.append(id_)

        elif criterion.__module__.split('.')[-1] in ['__main__', 'database']:
            for id_ in self.articles:
                if criterion(self, id_):
                    to_del.append(id_)

        else:
            raise Exception("Wrong criterion module: {}".format(criterion.__module__))

        for id_ in to_del:
            del self.articles[id_]

        self.compute_size()

    def progression(self, count):
        """
        Prints progression's updates and update the articles' count.

        Args:
            count: int, current count of articles.

        Returns:
            int, incremented count of articles.
        """

        count += 1

        if self.verbose and count % self.count_modulo == 0:
            print("File {}/{}...".format(count, self.size))

        return count

    @staticmethod
    def paths(pattern, limit):
        """
        Compute the paths of the files following the given pattern, limited to the limit firsts if limit is not None.

        Args:
            pattern: str, pattern for the files to follow.
            limit: int, maximum number of article's paths to compute; if None, returns all of them.

        Returns:
            list, file paths.
        """

        paths = glob(pattern)
        paths.sort()

        return paths[:limit] if limit else paths

    def plot_hist(self, fig, data, title, xlabel, log=False, print_data=True):
        """
        Plot the data as a histogram using matplotlib.pyplot. Print the data as well.

        Args:
            fig: int, index of the figure.
            data: numpy.histogram, histogram of the data.
            title: str, title of the figure.
            xlabel: str, label of the x-axis.
            log: bool, whether to use a logarithmic scale or not.
            print_data: bool, whether to print the data or not.
        """

        plt.figure(fig)

        counts, bins = data
        plt.hist(bins[:-1], bins, weights=counts, align='left', rwidth=.8, log=log)
        plt.title(title)
        plt.xlabel(xlabel)

        if print_data:
            s = self.to_string([str(int(bins[i])) + ':' + str(int(counts[i])) for i in range(len(counts))])
            print(title + ': ' + s)

    # endregion


def main():
    d = Database(max_size=1000, root='../databases/nyt_jingyun')

    d.preprocess_database()
    d.stats_tuples()

    d.filter_threshold(threshold=2)
    d.preprocess_articles()

    d.process_contexts()
    d.stats_contexts()

    # Database.set_parameters(to_print=[], print_attribute=True)
    # print(d)


if __name__ == '__main__':
    main()
