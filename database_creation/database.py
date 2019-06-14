from database_creation.utils import BaseClass
from database_creation.article import Article

from copy import copy
from numpy import random
from glob import glob
from collections import defaultdict
from numpy import histogram
from wikipedia import search, page, PageError, DisambiguationError
from pickle import dump, load

import matplotlib.pyplot as plt


class Database(BaseClass):
    # region Class initialization

    to_print = ['articles']
    print_attributes, print_lines, print_offsets = False, 2, 0
    limit_print, random_print = 50, True
    modulo_articles, modulo_tuples = 1000, 100

    @BaseClass.Verbose("Initializing the database...")
    def __init__(self, max_size=None, threshold=1, year='2000', project_root=''):
        """
        Initializes an instance of Database.

        Args:
            max_size: int, maximum number of articles in the database; if None, takes all articles.
            threshold: int, minimum frequency of the tuples of entities we consider.
            year: str, year of the database to analyse.
            project_root: str, relative path to the root of the project.
        """

        self.max_size = max_size
        self.threshold = threshold
        self.year = str(year)
        self.project_root = project_root

        self.articles = None

        self.tuples = None
        self.tuples_ids = None

        self.wikipedia = None
        self.not_wikipedia = None
        self.ambiguous = None

        self.stats = None

        self.samples = None

        self.compute_articles()
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

        if 'articles' in attributes:
            ids = self.get_ids()
            if random_print:
                random.shuffle(ids)

            string += self.prefix(True, print_lines if string else 0, print_offsets, 'articles') \
                if print_attribute else ''

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

        self.compute_metadata()
        self.compute_tuples()
        self.clean(Database.criterion_tuples_ids)

    @BaseClass.Verbose("Preprocessing the articles...")
    def preprocess_articles(self):
        """ Performs the preprocessing of the articles. """

        self.compute_annotations()

    @BaseClass.Verbose("Processing the wikipedia information...")
    def process_wikipedia(self, load):
        """
        Performs the processing of the wikipedia information of the database.

        Args:
            load: bool, if True, load an existing file (don't add new entry).
        """

        if load:
            self.load_attribute(attribute_name='wikipedia', file_name='wikipedia/wikipedia')
            self.load_attribute(attribute_name='not_wikipedia', file_name='wikipedia/not_wikipedia')
            self.load_attribute(attribute_name='ambiguous', file_name='wikipedia/ambiguous')

        else:
            self.compute_wikipedia()

            self.save_attribute(attribute_name='wikipedia', file_name='wikipedia/wikipedia')
            self.save_attribute(attribute_name='not_wikipedia', file_name='wikipedia/not_wikipedia')
            self.save_attribute(attribute_name='ambiguous', file_name='wikipedia/ambiguous')

    @BaseClass.Verbose("Processing the articles contexts...")
    def process_contexts(self):
        """ Performs the processing of the contexts of the database. """

        self.compute_contexts()

    @BaseClass.Verbose("Processing the aggregation samples...")
    def process_samples(self, load):
        """
        Performs the processing of the aggregation samples of the database.

        Args:
            load: bool, if True, load an existing file.
        """

        if load:
            self.load_attribute(attribute_name='samples', file_name='results/samples')

        else:
            self.compute_samples()
            self.save_attribute(attribute_name='samples', file_name='results/samples')

    @BaseClass.Verbose("Computing and displaying statistics...")
    def process_stats(self, type_):
        """
        Compute and display the statistics of the database of the given type.

        Args:
            type_: str, type of the statistics, must be 'tuples', 'wikipedia' or 'contexts'.
        """

        getattr(self, 'compute_stats_' + type_)()
        getattr(self, 'display_stats_' + type_)()

    @BaseClass.Verbose("Filtering the articles according to a threshold...")
    @BaseClass.Attribute('tuples')
    def filter_threshold(self, threshold=None):
        """
        Filter out the articles that doesn't respect the specified threshold on the entities tuple frequency.

        Args:
            threshold: int, minimal number of articles an entities tuple must appear in to be considered; if None,
            doesn't change the threshold.
        """

        self.threshold = threshold if threshold else self.threshold

        tuples, tuples_ids = [], set()

        for entities_tuple in self.tuples:
            if entities_tuple['frequency'] >= self.threshold:
                tuples.append(entities_tuple)
                tuples_ids.update(entities_tuple['ids'])

        self.tuples = tuples
        self.tuples_ids = tuples_ids

        self.clean(Database.criterion_tuples_ids)

    # endregion

    # region Methods compute_

    def compute_articles(self):
        """ Computes and initializes the articles in the database. """

        articles = {}
        root = self.project_root + 'databases/nyt_jingyun/'

        for original_path in self.paths(pattern=root + 'data/' + self.year + '/*/*/*.xml', limit=self.max_size):
            id_ = original_path.split('/')[-1].split('.')[0]
            annotated_path = root + 'content_annotated/' + self.year + 'content_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(original_path=original_path, annotated_path=annotated_path)

        self.articles = articles

    @BaseClass.Verbose("Computing the articles' metadata...")
    def compute_metadata(self):
        """ Computes the metadata of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_metadata()

    @BaseClass.Verbose("Computing the articles' annotations...")
    def compute_annotations(self):
        """ Computes the annotations of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_annotations()

    @BaseClass.Verbose("Computing the entity tuples...")
    def compute_tuples(self):
        """ Compute the entities tuples of the database as a sorted list of dictionaries with the tuple, the frequency
        (in number of articles) and the ids of the articles. """

        tuples_ids, tuples_type = defaultdict(set), defaultdict(str)
        count, size = 0, len(self.articles)

        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            for type_ in ['location', 'person', 'organization']:
                entities = getattr(self.articles[id_], 'entities')[type_]

                if len(entities) >= 2:
                    for t in self.subtuples(entities):
                        tuples_ids[t].add(id_)
                        tuples_type[t] = type_

        sorted_tuples = sorted(tuples_ids, key=lambda k: len(tuples_ids[k]), reverse=True)
        tuples = [{'tuple_': t, 'rank': rank + 1, 'frequency': len(tuples_ids[t]),
                   'ids': tuples_ids[t], 'type_': tuples_type[t]}
                  for rank, t in enumerate(sorted_tuples)]
        tuples_ids = set([id_ for tuple_ in tuples for id_ in tuple_['ids']])

        self.tuples = tuples
        self.tuples_ids = tuples_ids

    @BaseClass.Verbose("Computing the contexts...")
    def compute_contexts(self):
        """ Compute the contexts of the articles for each tuple of entities. """

        count, size = 0, len(self.tuples)

        for entities_tuple in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            for id_ in entities_tuple['ids']:
                self.articles[id_].compute_contexts(tuple_=entities_tuple['tuple_'], type_='neigh_sent')

    @BaseClass.Verbose("Computing the wikipedia information...")
    def compute_wikipedia(self):
        """ Compute the wikipedia information about the entities from self.tuples. """

        wikipedia, not_wikipedia, ambiguous = dict(), dict(), dict()
        count, size = 0, len(self.tuples)

        for entities_tuple in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            for entity in entities_tuple['tuple_']:
                if entity not in wikipedia and entity not in not_wikipedia:

                    raw_entities = self.get_raw_entities(entity, entities_tuple['ids'], entities_tuple['type_'])

                    if len(raw_entities) > 1:
                        print("Ambiguous case, first one chosen: {}".format(self.to_string(raw_entities)))
                        ambiguous[entity] = raw_entities

                    raw_entity = raw_entities[0]
                    p = self.wikipedia_page(entity, raw_entity, entities_tuple['type_'])

                    if p:
                        wikipedia[entity] = p.summary
                    else:
                        not_wikipedia[entity] = raw_entity

        self.wikipedia = wikipedia
        self.not_wikipedia = not_wikipedia
        self.ambiguous = ambiguous

    @BaseClass.Verbose("Computing the aggregation samples...")
    def compute_samples(self):
        """ Compute the aggregation samples of the database. """

        samples = dict()
        count, size = 0, len(self.tuples)

        for entities_tuple in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            info = dict([(entity, self.wikipedia[entity])
                         for entity in entities_tuple['tuple_'] if entity in self.wikipedia])

            for article_id_ in entities_tuple['ids']:
                article_samples = self.articles[article_id_].get_samples(entities_tuple['tuple_'], info)

                for context_id_ in article_samples:
                    sample_id_ = str(entities_tuple['rank']) + '_' + article_id_ + '_' + context_id_
                    samples[sample_id_] = article_samples[context_id_]

        self.samples = samples

    # endregion

    # region Methods get_

    def get_ids(self):
        """
        Computes the ids of the articles in the database.

        Returns:
            list, ids of the articles.
        """

        return list(self.articles.keys())

    def get_raw_entities(self, entity, ids, type_):
        """
        Return the raw entities of the entity from the articles ids.

        Args:
            entity: str, entity to analyse.
            ids: set, ids of the articles to scan.
            type_: str, type of the entity.

        Returns:
            list, raw entities of the entity.
        """

        raw_entities = sorted(
            set([self.articles[id_].raw_entities[entity] for id_ in ids if self.articles[id_].raw_entities[entity]]),
            key=len,
            reverse=True
        )

        standardized = getattr(self, 'standardize_' + type_)(entity)
        if len(raw_entities) > 1 and standardized in raw_entities:
            raw_entities.remove(standardized)

        if len(raw_entities) > 1 and all([raw_entities[i] in raw_entities[0] for i in range(1, len(raw_entities))]):
            raw_entities = [raw_entities[0]]

        return raw_entities

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

    # region Statistics methods

    def compute_stats_tuples(self):
        """ Compute the entities tuples statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['tuples_lengths'] = self.stat_tuples_lengths()
        self.stats['tuples_frequencies'] = self.stat_tuples_frequencies()
        self.stats['tuples_thresholds'] = self.stat_tuples_thresholds()

    def compute_stats_wikipedia(self):
        """ Compute the wikipedia statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['wikipedia_length'] = self.stat_wikipedia_length('wikipedia')
        self.stats['notwikipedia_length'] = self.stat_wikipedia_length('not_wikipedia')
        self.stats['ambiguous_length'] = self.stat_wikipedia_length('ambiguous')
        self.stats['wikipedia_frequencies'] = self.stat_wikipedia_frequencies('wikipedia')
        self.stats['notwikipedia_frequencies'] = self.stat_wikipedia_frequencies('not_wikipedia')

    def compute_stats_contexts(self):
        """ Compute the contexts statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['contexts'] = self.stat_contexts()

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

    def stat_wikipedia_length(self, file):
        """
        Compute the number of entries in the file.

        Args:
            file: str, file to analyse, must be 'wikipedia', 'not_wikipedia' or 'ambiguous'.

        Returns:
            int, number of entries in the file.
        """

        file = getattr(self, file)

        if file is not None:
            return len(file)
        else:
            return 0

    def stat_wikipedia_frequencies(self, file):
        """
        Compute the histogram of the frequencies of the tuples where appear the entities from file as a numpy.histogram.

        Args:
            file: str, file to analyse, must be 'wikipedia', 'not_wikipedia' or 'ambiguous'.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        file = getattr(self, file)

        data = [entities_tuple['frequency'] for entities_tuple in self.tuples
                for entity in entities_tuple['tuple_'] if entity in file]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_contexts(self):
        """
        Compute the histogram of the number of contexts as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the number of contexts.
        """

        data = []
        for entities_tuple in self.tuples:
            length = 0
            for id_ in entities_tuple['ids']:
                length += len(self.articles[id_].contexts[entities_tuple['tuple_']])

            data.append(length)

        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def display_stats_tuples(self):
        """ Display the entities tuples statistics of the database. """

        print("\nTotal number of tuples: {}".format(len(self.tuples)))
        print("\n10 most frequent tuples:")
        for entities_tuple in self.tuples[:10]:
            print("{} (in {} articles)".format(self.to_string(entities_tuple['tuple_']), entities_tuple['frequency']))
        print()

        self.plot_hist(fig=1, data=self.stats['tuples_lengths'], xlabel='lengths', log=True,
                       title='Lengths of the tuples of entities')

        self.plot_hist(fig=2, data=self.stats['tuples_frequencies'], xlabel='frequencies', log=True,
                       title='Frequencies of the tuples of entities')

        self.plot_hist(fig=3, data=self.stats['tuples_thresholds'], xlabel='thresholds', log=True,
                       title='Number of articles for each threshold on the frequency')

    def display_stats_wikipedia(self):
        """ Display the wikipedia statistics of the database. """

        print("\nTotal number of wikipedia: {}/not_wikipedia: {}/ambiguous: {}"
              .format(self.stats['wikipedia_length'],
                      self.stats['notwikipedia_length'],
                      self.stats['ambiguous_length']))

        print("\nWikipedia info of 5 most frequent tuples:")
        for entities_tuple in self.tuples[:5]:
            for entity in entities_tuple['tuple_']:
                print('\n' + self.wikipedia[entity]) if entity in self.wikipedia \
                    else print("\nNo information on {}".format(entity))

        print("\n10 not_wikipedia examples:")
        for entity in list(self.not_wikipedia.keys())[:10]:
            print(entity + ' (' + self.not_wikipedia[entity] + ')')

        print("\n10 ambiguous examples:")
        for entity in list(self.ambiguous.keys())[:10]:
            print(self.to_string(self.ambiguous[entity]) + ' (' + entity + ')')
        print()

        self.plot_hist(fig=4, data=self.stats['wikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities found in wikipedia')

        self.plot_hist(fig=5, data=self.stats['notwikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities not found in wikipedia')

    def display_stats_contexts(self):
        """ Display the contexts statistics of the database. """

        self.plot_hist(fig=6, data=self.stats['contexts'], xlabel='number of contexts', log=True,
                       title="Number of contexts found for each tuple")

    # endregion

    # region Other methods

    @BaseClass.Verbose("Cleaning the database...")
    @BaseClass.Attribute('articles')
    def clean(self, criterion):
        """
        Removes from the database the articles which meets the Article's or Database's criterion.

        Args:
            criterion: function, method from Article or Database, criterion that an article must meet to be removed.
        """

        print("Criterion: {}".format([line for line in criterion.__doc__.splitlines() if line][0][8:]))
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

    def progression(self, count, modulo, size, text):
        """
        Prints progression's updates and update the count.

        Args:
            count: int, current count.
            modulo: int, how often to print updates.
            size: int, size of the element to count.
            text: str, what to print at the beginning of the updates.

        Returns:
            int, incremented count of articles.
        """

        count += 1

        if self.verbose and count % modulo == 0:
            print("  " + text + " {}/{}...".format(count, size))

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

    @staticmethod
    def plot_hist(fig, data, title, xlabel, log=False):
        """
        Plot the data as a histogram using matplotlib.pyplot. Print the data as well.

        Args:
            fig: int, index of the figure.
            data: numpy.histogram, histogram of the data.
            title: str, title of the figure.
            xlabel: str, label of the x-axis.
            log: bool, whether to use a logarithmic scale or not.
        """

        plt.figure(num=fig, figsize=(12, 4))

        counts, bins = data
        plt.hist(bins[:-1], bins, weights=counts, align='left', rwidth=.8, log=log)
        plt.title(title)
        plt.xlabel(xlabel)

    def save_attribute(self, attribute_name, file_name):
        """
        Save an attribute using pickle.

        Args:
            attribute_name: str, name of the attribute to save.
            file_name: str, folder and name of the file to load (without extension).
        """

        obj = getattr(self, attribute_name)
        if obj is None:
            raise Exception("Nothing to save, attribute is None.")

        prefix, suffix = self.prefix_suffix()
        file_name = prefix + file_name + suffix + '.pkl'

        with open(file_name, 'wb') as f:
            dump(obj=obj, file=f, protocol=-1)

        print("Attribute {} saved at {}".format(attribute_name, file_name))

    def load_attribute(self, attribute_name, file_name):
        """
        Load an attribute using pickle.

        Args:
            attribute_name: str, name of the attribute to load.
            file_name: str, folder and name of the file to load (without extension).
        """

        prefix, suffix = self.prefix_suffix()
        file_name = prefix + file_name + suffix + '.pkl'

        with open(file_name, 'rb') as f:
            obj = load(f)

        setattr(self, attribute_name, obj)
        print("Attribute {} loaded from {}".format(attribute_name, file_name))

    def prefix_suffix(self):
        """
        Returns the standard beginning and ending for a file path.

        Returns:
            prefix: str, beginning of the name of the file (until the name of the folder).
            suffix: str, ending of the name of the file (after the basic name of the file).
        """

        prefix = self.project_root
        suffix = '_' + self.year

        if self.max_size is not None:
            if self.max_size >= 1000:
                suffix += '_size' + str(self.max_size // 1000) + 'k_threshold' + str(self.threshold)
            else:
                suffix += '_size' + str(self.max_size) + '_threshold' + str(self.threshold)

        return prefix, suffix

    def wikipedia_page(self, entity, raw_entity, type_):
        """
        Compute the wikipedia page of the entity, or None is none found.

        Args:
            entity: str, standardized entity to look for.
            raw_entity: str, original entity to look for.
            type_: str, type of the entity, must be 'location', 'person' or organization'.

        Returns:
            wikipedia.page, page of the entity.
        """

        try:
            p = page(entity)
            return p

        except DisambiguationError:
            pass
        except PageError:
            pass

        for s in search(raw_entity)[0:5]:
            try:
                p = page(s)
                for e in [entity, raw_entity]:
                    if self.match(e, p.title, type_, flexible=False) or e in p.summary:
                        return p

            except DisambiguationError:
                pass
            except PageError:
                pass

        return None

    # endregion


def main():
    database = Database(max_size=1000, project_root='../')

    database.preprocess_database()
    database.filter_threshold()

    database.preprocess_articles()
    database.process_contexts()

    database.process_wikipedia(load=False)

    database.process_samples(load=False)


if __name__ == '__main__':
    main()
