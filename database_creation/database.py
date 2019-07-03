from database_creation.utils import BaseClass, Tuple, Query
from database_creation.article import Article

from os import remove as os_remove
from copy import copy
from numpy.random import shuffle, seed
from random import sample
from glob import glob
from collections import defaultdict
from numpy import histogram
from wikipedia import search, page, PageError, DisambiguationError
from pickle import dump, load, PicklingError
from textwrap import fill
from nltk import sent_tokenize
from pandas import DataFrame
from re import sub

import matplotlib.pyplot as plt


class Database(BaseClass):
    # region Class initialization

    to_print = ['articles']
    print_attributes, print_lines, print_offsets = False, 2, 0
    limit_print, random_print = 50, True
    modulo_articles, modulo_tuples = 1000, 100
    info_length = 600

    def __init__(self, year='2000', max_size=None, project_root='', verbose=True, min_articles=None, min_queries=None):
        """
        Initializes an instance of Database.

        Args:
            year: str, year of the database to analyse.
            max_size: int, maximum number o=f articles in the database; if None, takes all articles.
            project_root: str, relative path to the root of the project.
            verbose: bool, verbose option of the database.
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
        """

        self.year = str(year)
        self.max_size = max_size
        self.project_root = project_root
        self.verbose = verbose

        self.min_articles = min_articles
        self.min_queries = min_queries

        self.articles = None
        self.article_ids = None

        self.tuples = None

        self.wikipedia = None
        self.not_wikipedia = None
        self.ambiguous = None

        self.stats = None

        self.queries = None
        self.answers = None

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
            ids = self.article_ids
            if random_print:
                shuffle(ids)

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

        self.compute_articles()
        self.clean(Article.criterion_data)
        self.compute_metadata()
        self.compute_tuples()

    @BaseClass.Verbose("Preprocessing the articles...")
    def preprocess_articles(self):
        """ Performs the preprocessing of the articles. """

        self.compute_annotations()
        self.compute_contexts()

    @BaseClass.Verbose("Processing the wikipedia information...")
    def process_wikipedia(self, load=False, file_name=None):
        """
        Performs the processing of the wikipedia information of the database.

        Args:
            load: bool, if True, load an existing file.
            file_name: str, name of the wikipedia file to load; if None, load the standard files.
        """

        if load:
            if file_name is None:
                self.load(attribute_name='wikipedia', folder_name='wikipedia')
                self.load(attribute_name='not_wikipedia', folder_name='wikipedia')
                self.load(attribute_name='ambiguous', folder_name='wikipedia')

            else:
                self.load(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')

        else:
            self.compute_wikipedia()

            self.save(attribute_name='wikipedia', folder_name='wikipedia')
            self.save(attribute_name='not_wikipedia', folder_name='wikipedia')
            self.save(attribute_name='ambiguous', folder_name='wikipedia')

    @BaseClass.Verbose("Processing the aggregation queries...")
    def process_queries(self, load=False, file_name=None):
        """
        Performs the processing of the aggregation queries.

        Args:
            load: bool, if True, load an existing file.
            file_name: str, name of the queries file to load; if None, load the standard file.
        """

        if load:
            if file_name is None:
                self.load('queries')
            else:
                self.load(attribute_name='queries', file_name=file_name)

        else:
            self.compute_queries()

            self.save('queries')
            self.save_csv(attribute_name='queries', limit=None)
            self.save_csv(attribute_name='queries', limit=200)

    @BaseClass.Verbose("Computing and displaying statistics...")
    def process_stats(self, type_):
        """
        Compute and display the statistics of the database of the given type.

        Args:
            type_: str, type of the statistics, must be 'tuples', 'wikipedia' or 'contexts'.
        """

        getattr(self, 'compute_stats_' + type_)()
        getattr(self, 'display_stats_' + type_)()

    @BaseClass.Verbose("Filtering the articles...")
    @BaseClass.Attribute('tuples')
    def filter(self, min_articles=None, min_queries=None):
        """
        Filter out the articles that doesn't respect the specified threshold on the minimum number of articles or the
        minimum number of queries.

        Args:
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
        """

        if min_articles is not None:
            self.print("Minimum number of articles: {}".format(min_articles))

            if min_articles >= 1:
                tuples, article_ids = [], set()

                for tuple_ in self.tuples:
                    if len(tuple_.article_ids) >= min_articles:
                        tuples.append(tuple_)
                        article_ids.update(tuple_.article_ids)

                self.tuples = tuples
                self.article_ids = article_ids
                self.min_articles = min_articles

                self.clean(Database.criterion_article_ids)

        if min_queries is not None:
            self.print("Minimum number of queries: {}".format(min_queries))

            if min_queries >= 1:
                tuples, article_ids = [], set()

                for tuple_ in self.tuples:
                    if len(tuple_.query_ids) >= min_queries:
                        tuples.append(tuple_)
                        article_ids.update(tuple_.article_ids)

                self.tuples = tuples
                self.article_ids = article_ids
                self.min_queries = min_queries

                self.clean(Database.criterion_article_ids)

    @BaseClass.Verbose("Performing the annotation task..")
    def ask(self, n_queries=1):
        """
        Performs the annotation task by asking the specified number of queries.

        Args:
            n_queries: int, number of queries to ask.
        """

        answer = defaultdict(list)
        query_ids = sample(list(self.queries), n_queries)

        for query_id_ in query_ids:
            print(self.to_string(self.queries[query_id_]))

            a = input("Answer: ")
            answer[query_id_].append(a) if a else None
            print('\n')

        count = 1
        while True:
            try:
                self.load(file_name='answer_' + str(count), folder_name='answers')
                count += 1
            except FileNotFoundError:
                break

        self.save(obj=answer, file_name='answer_' + str(count), folder_name='answers')

    @BaseClass.Verbose("Gathering the answers..")
    @BaseClass.Attribute('answers')
    def gather(self):
        """ Gather the different answers into one file and load them. """

        try:
            self.load(attribute_name='answers', folder_name='answers')
        except FileNotFoundError:
            self.answers = defaultdict(list)

        count = 1
        while True:
            try:
                answer = self.pop_file(file_name='answer_' + str(count), folder_name='answers')
                count += 1

                for query_id_ in answer:
                    self.answers[query_id_].extend(answer[query_id_])

            except FileNotFoundError:
                break

        self.save(attribute_name='answers', folder_name='answers')

    # endregion

    # region Methods compute_

    @BaseClass.Verbose("Computing the database' article...")
    def compute_articles(self):
        """ Computes and initializes the articles in the database. """

        articles = {}
        root = self.project_root + 'databases/nyt_jingyun/'

        for original_path in self.paths(pattern=root + 'data/' + self.year + '/*/*/*.xml', limit=self.max_size):
            id_ = original_path.split('/')[-1].split('.')[0]
            annotated_path = root + 'content_annotated/' + self.year + 'content_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(original_path=original_path, annotated_path=annotated_path)

        self.articles = articles
        self.article_ids = set(self.articles.keys())

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
        """ Compute the entity tuples of the database as a sorted list of Tuples with its entities, its type and the
        ids of its articles. """

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

        tuples = [Tuple(id_=str(idx + 1), entities=t, type_=tuples_type[t], article_ids=tuples_ids[t])
                  for idx, t in enumerate(sorted_tuples)]

        self.tuples = tuples

    @BaseClass.Verbose("Computing the contexts...")
    def compute_contexts(self):
        """ Compute the contexts of the articles for each entity Tuple. """

        count, size = 0, len(self.tuples)

        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            query_ids = set()

            for article_id_ in tuple_.article_ids:
                self.articles[article_id_].compute_contexts(entities=tuple_.entities, type_='neigh_sent')

                query_ids.update({tuple_.id_ + '_' + article_id_ + '_' + context_id_
                                  for context_id_ in self.articles[article_id_].contexts[tuple_.entities]})

            tuple_.query_ids = query_ids

    @BaseClass.Verbose("Computing the wikipedia information...")
    def compute_wikipedia(self):
        """ Compute the wikipedia information about the entities from self.tuples. """

        wikipedia, not_wikipedia, ambiguous = dict(), dict(), dict()
        count, size = 0, len(self.tuples)

        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            for entity in tuple_.entities:
                if entity not in wikipedia and entity not in not_wikipedia:

                    raw_entities = self.get_raw_entities(entity, tuple_.article_ids, tuple_.type_)

                    if len(raw_entities) > 1:
                        self.print("Ambiguous case, first one chosen: {}".format(self.to_string(raw_entities)))
                        ambiguous[entity] = raw_entities

                    raw_entity = raw_entities[0]
                    p = self.wikipedia_page(entity, raw_entity, tuple_.type_)

                    if p:
                        wikipedia[entity] = {'summary': p.summary, 'url': p.url}
                    else:
                        not_wikipedia[entity] = raw_entity

        self.wikipedia = wikipedia
        self.not_wikipedia = not_wikipedia
        self.ambiguous = ambiguous

    @BaseClass.Verbose("Computing the aggregation queries...")
    def compute_queries(self):
        """ Compute the aggregation Queries of the database. """

        queries = dict()
        count, size = 0, len(self.tuples)

        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            info = self.get_info(tuple_.entities)

            for article_id_ in tuple_.article_ids:
                article_contexts = self.articles[article_id_].contexts[tuple_.entities]

                for context_id_ in article_contexts:
                    query_id_ = tuple_.id_ + '_' + article_id_ + '_' + context_id_

                    queries[query_id_] = Query(id_=query_id_,
                                               entities=tuple_.entities,
                                               title=self.articles[article_id_].title,
                                               date=self.articles[article_id_].date,
                                               abstract=self.articles[article_id_].abstract,
                                               info=info,
                                               context=article_contexts[context_id_])

        self.queries = queries

    # endregion

    # region Methods get_

    def get_raw_entities(self, entity, article_ids, type_):
        """
        Return the raw entities of the entity from the articles ids.

        Args:
            entity: str, entity to analyse.
            article_ids: set, ids of the articles to scan.
            type_: str, type of the entity.

        Returns:
            list, raw entities of the entity.
        """

        raw_entities = sorted(set([self.articles[id_].raw_entities[entity]
                                   for id_ in article_ids if self.articles[id_].raw_entities[entity]]),
                              key=len,
                              reverse=True)

        standardized = getattr(self, 'standardize_' + type_)(entity)
        if len(raw_entities) > 1 and standardized in raw_entities:
            raw_entities.remove(standardized)

        if len(raw_entities) > 1 and all([raw_entities[i] in raw_entities[0] for i in range(1, len(raw_entities))]):
            raw_entities = [raw_entities[0]]

        return raw_entities

    def get_info(self, entities):
        """
        Compute the wikipedia info of the entities.

        Args:
            entities: tuple, entities mentioned in the article.

        Returns:
            dict, wikipedia info of the tuple.
        """

        info = dict()

        for entity in entities:
            if entity in self.wikipedia:
                paragraph = self.wikipedia[entity]['summary'].split('\n')[0]

                if len(paragraph) > self.info_length:
                    sentences = sent_tokenize(paragraph)[0]

                    for sentence in sent_tokenize(paragraph)[1:]:
                        if len(sentences + sentence) <= self.info_length:
                            sentences += ' ' + sentence
                        else:
                            break

                    paragraph = sentences

                paragraph = sub(r'\([^)]*\)', '', paragraph).replace('  ', ' ')
                paragraph = paragraph.encode("utf-8", errors="ignore").decode()

                url = self.wikipedia[entity]['url']

                info[entity] = {'paragraph': paragraph, 'url': url}

            else:
                info[entity] = {}

        return info

    # endregion

    # region Methods criterion_

    def criterion_article_ids(self, id_):
        """
        Check if an article does not belong to the article ids attribute.

        Args:
            id_: string, id of the article to analyze.

        Returns:
            bool, True iff the article does not belong to the article ids.
        """

        return True if id_ not in self.article_ids else False

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

        data = [len(tuple_.entities) for tuple_ in self.tuples]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_tuples_frequencies(self):
        """
        Compute the histogram of the frequencies of the tuples as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        data = [len(tuple_.article_ids) for tuple_ in self.tuples]
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

        m = max([len(tuple_.article_ids) for tuple_ in self.tuples])

        threshold_ids = [set() for _ in range(m + 1)]
        threshold_ids[0].update(self.article_ids)

        for tuple_ in self.tuples:
            for threshold in range(1, len(tuple_.article_ids) + 1):
                threshold_ids[threshold].update(tuple_.article_ids)

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

        data = [len(tuple_.article_ids) for tuple_ in self.tuples for entity in tuple_.entities if entity in file]
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
        for tuple_ in self.tuples:
            length = 0
            for id_ in tuple_.article_ids:
                length += len(self.articles[id_].contexts[tuple_.entities])

            data.append(length)

        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def display_stats_tuples(self):
        """ Display the entities tuples statistics of the database. """

        print("\nTotal number of tuples: {}".format(len(self.tuples)))
        print("\n10 most frequent tuples:")
        for tuple_ in self.tuples[:10]:
            print("{} (in {} articles)".format(self.to_string(tuple_.entities), len(tuple_.article_ids)))
        print()

        self.plot_hist(fig=1, data=self.stats['tuples_lengths'], xlabel='lengths', log=True,
                       title='Lengths of the tuples of entities')

        self.plot_hist(fig=2, data=self.stats['tuples_frequencies'], xlabel='frequencies', log=True,
                       title='Frequencies of the tuples of entities')

        self.plot_hist(fig=3, data=self.stats['tuples_thresholds'], xlabel='thresholds', log=True,
                       title='Number of articles for each threshold on the frequency')

    def display_stats_contexts(self):
        """ Display the contexts statistics of the database. """

        self.plot_hist(fig=6, data=self.stats['contexts'], xlabel='number of contexts', log=True,
                       title="Number of contexts found for each tuple")

    def display_stats_wikipedia(self):
        """ Display the wikipedia statistics of the database. """

        print("\nTotal number of wikipedia: {}/not_wikipedia: {}/ambiguous: {}"
              .format(self.stats['wikipedia_length'],
                      self.stats['notwikipedia_length'],
                      self.stats['ambiguous_length']))

        print("\nWikipedia info of 10 most frequent tuples:\n")
        for tuple_ in self.tuples[:10]:
            print(self.to_string(self.get_info(tuple_.entities)) + '\n')

        print("\nEntities not found in wikipedia:")
        for entity in self.not_wikipedia:
            print(entity + ' (' + self.not_wikipedia[entity] + ')')

        print("\nAmbiguous cases:")
        for entity in self.ambiguous:
            print(self.to_string(self.ambiguous[entity]) + ' (' + entity + ')')
        print()

        self.plot_hist(fig=4, data=self.stats['wikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities found in wikipedia')

        self.plot_hist(fig=5, data=self.stats['notwikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities not found in wikipedia')

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

        self.print("Criterion: {}".format([line for line in criterion.__doc__.splitlines() if line][0][8:]))
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

        self.article_ids = set(self.articles.keys())

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

        if count % modulo == 0:
            self.print("  " + text + " {}/{}...".format(count, size))

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

    def save(self, attribute_name=None, obj=None, file_name=None, folder_name='queries'):
        """
        Save an attribute (designated by its name) or an object into a file using pickle.

        Args:
            attribute_name: str, name of the attribute to save; if None, save an object instead.
            obj: unk, object saved if no attribute name is provided.
            file_name: str, name of the file; if None, save an attribute with the standard name.
            folder_name: str, name of the folder to save in.
        """

        if attribute_name is not None:
            obj = getattr(self, attribute_name)
        if obj is None:
            raise Exception("Nothing to save, object is None.")

        prefix, suffix = self.prefix_suffix()
        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + suffix + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to save the object.")

        try:
            with open(file_name, 'wb') as f:
                dump(obj=obj, file=f, protocol=-1)

            if attribute_name is not None:
                self.print("Attribute {} saved at {}.".format(attribute_name, file_name))
            else:
                self.print("Object saved at {}.".format(file_name))

        except PicklingError:
            print("Could not save (PicklingError.")

    def load(self, attribute_name=None, file_name=None, folder_name='queries'):
        """
        Load an attribute (designated by its name) or an object from a file using pickle.

        Args:
            attribute_name: str, name of the attribute to load; if None, returns the object.
            file_name: str, name of the file to load; if None, load the file with the corresponding standard name.
            folder_name: str, name of the folder to load from.
        """

        prefix, suffix = self.prefix_suffix()
        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + suffix + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to load the object.")

        with open(file_name, 'rb') as f:
            obj = load(f)

        if attribute_name is not None:
            self.print("Attribute {} loaded from {}.".format(attribute_name, file_name))
            setattr(self, attribute_name, obj)
        else:
            self.print("Object loaded from {}".format(file_name))
            return obj

    def pop_file(self, file_name=None, folder_name='queries'):
        """
        Remove and returns a pickle file.

        Args:
            file_name: str, name of the file to delete.
            folder_name: str, name of the folder of the file to delete.
        """

        prefix, suffix = self.prefix_suffix()
        file_name = prefix + folder_name + '/' + file_name + suffix + '.pkl'

        with open(file_name, 'rb') as f:
            obj = load(f)

        os_remove(file_name)

        self.print("Object loaded from {} and file deleted.".format(file_name))
        return obj

    def save_csv(self, attribute_name=None, folder_name='queries', limit=None):
        """
        Save a dictionary attribute to a .csv using pandas DataFrame.

        Args:
            attribute_name: str, name of the attribute to save.
            folder_name: str, name of the folder to save in.
            limit: int, maximum number of data to save; if None, save all of them.
        """

        obj = getattr(self, attribute_name)
        ids = list(obj.keys())

        if limit is not None:
            seed(seed=42)
            shuffle(ids)
            ids = ids[:limit]

        data = [obj[id_].to_dict() for id_ in ids]
        df = DataFrame.from_records(data=data)

        prefix, suffix = self.prefix_suffix()
        file_name = attribute_name if limit is None else attribute_name + '_short'
        file_name = prefix + folder_name + '/' + file_name + suffix + '.csv'

        df.to_csv(file_name, index_label='idx')

        self.print("Attribute {} saved at {}".format(attribute_name, file_name))

    def prefix_suffix(self):
        """
        Returns the standard beginning and ending for a file path.

        Returns:
            prefix: str, beginning of the name of the file (until the name of the folder).
            suffix: str, ending of the name of the file (after the basic name of the file).
        """

        prefix, suffix = self.project_root + 'results/' + self.year + '/', ''

        if self.max_size is None:
            suffix += '_sizemax'
        elif self.max_size >= 1000:
            suffix += '_size' + str(self.max_size // 1000) + 'k'
        else:
            suffix += '_size' + str(self.max_size)

        if self.min_articles is not None:
            suffix += '_articles' + str(self.min_articles)
        if self.min_queries is not None:
            suffix += '_queries' + str(self.min_queries)

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


# region Annotation task
n_queries = 10
project_root = ''
max_size = 10000
min_articles = 1
min_queries = 1


def create_queries():
    """ Run the pipeline for the creation of a queries file for the database. """

    database = Database(max_size=max_size, project_root=project_root,
                        min_articles=min_articles, min_queries=min_queries)
    database.preprocess_database()
    database.filter(min_articles=min_articles)
    database.preprocess_articles()
    database.filter(min_queries=min_queries)
    database.process_wikipedia(load=False)
    database.process_queries(load=False)


def instructions():
    """ Show the instructions of the task. """

    with open(project_root + 'results/instructions.txt', 'r') as f:
        for line in f:
            print(fill(line, BaseClass.text_width))


def annotation_task():
    """ Run the annotation task. """

    database = Database(max_size=max_size, project_root=project_root, min_articles=min_articles,
                        min_queries=min_queries, verbose=False)
    database.process_queries(load=True)
    database.ask(n_queries)


def gather_answers():
    """ Gather the answers file. """

    database = Database(max_size=max_size, project_root=project_root, min_articles=min_articles,
                        min_queries=min_queries)
    database.gather()
# endregion


def main():
    return


if __name__ == '__main__':
    main()
