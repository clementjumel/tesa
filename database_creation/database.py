from database_creation.utils import Tuple, Wikipedia, Query
from database_creation.article import Article

from numpy.random import shuffle, seed
from time import time
from glob import glob
from collections import defaultdict
from numpy import histogram
from pickle import dump, load, PicklingError
from pandas import DataFrame
from wikipedia import WikipediaException

import matplotlib.pyplot as plt


class Database:
    # region Class initialization

    modulo_articles, modulo_tuples, modulo_entities = 500, 1000, 100

    def __init__(self, years=(2006, 2007), max_size=None, shuffle=False, project_root='',
                 min_articles=None, min_queries=None, random_seed=None):
        """
        Initializes an instance of Database.

        Args:
            years: list, years (int) of the database to analyse.
            max_size: int, maximum number of articles in the database; if None, takes all articles.
            shuffle: bool, whether to shuffle the articles selected in the database.
            project_root: str, relative path to the root of the project.
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
            random_seed: int, the seed to use for the random processes.
        """

        self.years = years
        self.max_size = max_size
        self.shuffle = shuffle
        self.project_root = project_root
        self.min_articles = min_articles
        self.min_queries = min_queries
        self.random_seed = random_seed

        self.articles = None
        self.entities = None
        self.tuples = None
        self.wikipedia = None
        self.queries = None
        self.stats = None

        seed(seed=self.random_seed)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Database.

        Returns:
            str, readable format of the instance.
        """

        s = "Years: " + ', '.join([str(year) for year in self.years]) + '\n'
        s += "Max size: " + str(self.max_size) + '\n'
        s += "Shuffle: " + str(self.shuffle) + '\n'
        s += "Min articles: " + str(self.min_articles) + '\n'
        s += "Min queries: " + str(self.min_queries)

        return s

    # endregion

    # region Decorators

    class Verbose:
        """ Decorator for the display of a simple message. """

        def __init__(self, message):
            """ Initializes the Verbose decorator message. """

            self.message = message

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                t0 = time()

                print(self.message)
                func(*args, **kwargs)
                print("Done (elapsed time: {}s).\n".format(round(time() - t0)))

            return f

    class Attribute:
        """ Decorator for monitoring the length of an attribute. """

        def __init__(self, attribute):
            """ Initializes the Attribute decorator attribute. """

            self.attribute = attribute

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                slf = args[0]

                attribute = getattr(slf, self.attribute)
                length = len(attribute) if attribute is not None else 0
                print("Initial length of {}: {}".format(self.attribute, length))

                func(*args, **kwargs)

                attribute = getattr(slf, self.attribute)
                length = len(attribute) if attribute is not None else 0
                print("Final length of {}: {}".format(self.attribute, length))

            return f

    # endregion

    # region Main methods

    @Verbose("Preprocessing the database...")
    def preprocess_database(self, debug=False):
        """
        Performs the preprocessing of the database.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        self.compute_articles(debug=debug)

        self.clean_articles(criterion=Article.criterion_content)

        self.compute_metadata(debug=debug)
        self.compute_entities(debug=debug)
        self.compute_tuples(debug=debug)

        self.filter(min_articles=self.min_articles)

    @Verbose("Preprocessing the articles...")
    def process_articles(self, debug=False):
        """
        Performs the preprocessing of the articles.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        self.compute_annotations(debug=debug)
        self.compute_contexts(debug=debug)

        self.filter(min_queries=self.min_queries)

    @Verbose("Processing the wikipedia information...")
    def process_wikipedia(self, load=False, file_name=None, debug=False):
        """
        Performs the processing of the wikipedia information of the database.

        Args:
            load: bool, if True, load an existing file.
            file_name: str, name of the wikipedia file to save or load; if None, deal with the standard files name.
            debug: bool, whether or not to perform the debugging of the database.
        """

        if load:
            self.load_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')
            self.compute_wikipedia(load=load, debug=debug)
            self.save_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')

        else:
            self.compute_wikipedia(load=load, debug=debug)
            self.save_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')

    @Verbose("Processing the aggregation queries...")
    def process_queries(self, load=False, check_changes=False, file_name=None, debug=False):
        """
        Performs the processing of the aggregation queries.

        Args:
            load: bool, if True, load an existing file.
            check_changes: bool, if not load, load the existing queries file and check if there are changes in the new.
            file_name: str, name of the wikipedia file to save or load; if None, deal with the standard files name.
            debug: bool, whether or not to perform the debugging of the database.
        """

        if load:
            self.load_pkl(attribute_name='queries', file_name=file_name)

        else:
            if check_changes:
                try:
                    self.load_pkl(attribute_name='queries', file_name=file_name)
                except FileNotFoundError:
                    check_changes = False
                    print("Unable to check the changes: the queries file is missing.")
            old_queries = self.queries

            self.compute_queries(debug=debug)
            self.save_pkl(attribute_name='queries', file_name=file_name)
            self.save_csv(attribute_name='queries', file_name=file_name, limit=100)

            if check_changes:
                if old_queries == self.queries:
                    print("\nNo change in the computed queries.")
                else:
                    print("\nThe queries have changed!")

    @Verbose("Computing and displaying statistics...")
    def process_stats(self, type_):
        """
        Compute and display the statistics of the database of the given type.

        Args:
            type_: str, type of the statistics, must be 'tuples', 'wikipedia' or 'contexts'.
        """

        getattr(self, 'compute_stats_' + type_)()
        getattr(self, 'display_stats_' + type_)()

    # endregion

    # region Methods compute_

    @Verbose("Computing the database' article...")
    @Attribute('articles')
    def compute_articles(self, debug=False):
        """
        Computes and initializes the articles in the database.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        articles = {}
        root = self.project_root + 'databases/nyt_jingyun/'

        for data_path in self.paths():
            id_ = data_path.split('/')[-1].split('.')[0]
            year = data_path.split('/')[-4]

            content_path = root + 'content_annotated/' + str(year) + 'content_annotated/' + id_ + '.txt.xml'
            summary_path = root + 'summary_annotated/' + str(year) + 'summary_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(data_path=data_path, content_path=content_path, summary_path=summary_path)

        self.articles = articles

        self.write_debug(field='articles', method='articles') if debug else None

    @Verbose("Computing the articles' metadata...")
    def compute_metadata(self, debug=False):
        """
        Computes the metadata of the articles.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_metadata()

        self.write_debug(field='articles', method='metadata') if debug else None

    @Verbose("Computing the database' entities...")
    @Attribute('entities')
    def compute_entities(self, debug=False):
        """
        Compute the entities of the database.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        self.entities = dict()

        count, size = 0, len(self.articles)
        for _, article in self.articles.items():
            count = self.progression(count, self.modulo_articles, size, 'article')

            try:
                entities = article.get_entities()
            except AssertionError:
                print("      Several entities have the same name ({}); ignoring them...".format(
                    '; '.join(article.get_vanilla_entities())
                ))
                entities = []

            for entity in entities:
                if str(entity) in self.entities:
                    try:
                        self.entities[str(entity)].update_info(entity)
                    except AssertionError:
                        print("      {} corresponds to both {} and {}, ignoring the later...".format(
                            str(entity), entity.type_, self.entities[str(entity)].type_
                        ))
                else:
                    self.entities[str(entity)] = entity

            article.entities = [self.entities[name] for name in [str(entity) for entity in entities]]

        self.write_debug(field='articles', method='article_entities') if debug else None
        self.write_debug(field='entities', method='entities') if debug else None

    @Verbose("Computing the entity tuples...")
    @Attribute('tuples')
    def compute_tuples(self, debug=False):
        """
        Compute the Tuples of the database as a sorted list of Tuples (by number of articles).

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        ids = defaultdict(set)

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')

            entities = defaultdict(set)
            for entity in self.articles[id_].entities:
                entities[entity.type_].add(entity.name)

            for type_ in entities:
                for tuple_ in self.subtuples(entities[type_]):
                    ids[tuple_].add(id_)

        ranking = sorted(ids, key=lambda k: (len(ids[k]), str(k)), reverse=True)

        self.tuples = [Tuple(id_=str(rank + 1),
                             entities=tuple([self.entities[name] for name in tuple_]),
                             article_ids=ids[tuple_])
                       for rank, tuple_ in enumerate(ranking)]

        self.write_debug(field='tuples', method='tuples') if debug else None

    @Verbose("Computing the articles' annotations...")
    def compute_annotations(self, debug=False):
        """
        Computes the annotations of the articles.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_annotations()

        self.write_debug(field='articles', method='annotations') if debug else None

    @Verbose("Computing the contexts...")
    def compute_contexts(self, debug=False):
        """
        Compute the contexts of the articles for each Tuple.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')
            query_ids = set()

            for article_id_ in tuple_.article_ids:
                self.articles[article_id_].compute_contexts(tuple_=tuple_)

                query_ids.update({tuple_.id_ + '_' + article_id_ + '_' + context_id_
                                  for context_id_ in self.articles[article_id_].contexts[str(tuple_)]})

            tuple_.query_ids = query_ids

        self.write_debug(field='articles', method='contexts') if debug else None

    @Verbose("Computing the Wikipedia information...")
    def compute_wikipedia(self, load, debug=False):
        """
        Compute the wikipedia information about the entities from self.tuples.

        Args:
            load: bool, if True, load an existing file.
            debug: bool, whether or not to perform the debugging of the database.
        """

        wikipedia = {'found': dict(), 'not_found': set()} if not load else self.wikipedia
        print("Initial found entries: {}/not found: {}".format(len(wikipedia['found']), len(wikipedia['not_found'])))

        try:
            count, size = 0, len(self.entities)
            for name, entity in self.entities.items():
                count = self.progression(count, self.modulo_entities, size, 'entity')

                if not load:
                    wiki = entity.get_wiki()
                    if wiki.summary is not None:
                        wikipedia['found'][name] = wiki
                    else:
                        wikipedia['not_found'].add(name)

                else:
                    if name in wikipedia['found']:
                        wiki = wikipedia['found'][name]
                    elif name in wikipedia['not_found']:
                        wiki = Wikipedia()
                    else:
                        wiki = entity.get_wiki()
                        if wiki.summary is not None:
                            wikipedia['found'][name] = wiki
                        else:
                            wikipedia['not_found'].add(name)

                entity.wiki = wiki

        except (KeyboardInterrupt, WikipediaException) as e:
            print("A known error occurred, saving the loaded information ({})...".format(e))

        print("Final found entries: {}/not found: {}".format(len(wikipedia['found']), len(wikipedia['not_found'])))
        self.wikipedia = wikipedia

        self.write_debug(field='wikipedia', method='wikipedia') if debug else None

    @Verbose("Computing the Queries...")
    @Attribute('queries')
    def compute_queries(self, debug=False):
        """
        Compute the Queries of the database.

        Args:
            debug: bool, whether or not to perform the debugging of the database.
        """

        queries = dict()

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            for article_id_ in sorted(tuple_.article_ids):
                article_contexts = self.articles[article_id_].contexts[str(tuple_)]

                for context_id_, context in article_contexts.items():
                    query_id_ = '_'.join([article_id_, tuple_.id_, context_id_])
                    queries[query_id_] = Query(id_=query_id_,
                                               tuple_=tuple_,
                                               article=self.articles[article_id_],
                                               context=context)

        self.queries = queries

        self.write_debug(field='queries', method='queries') if debug else None

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

        self.stats['wikipedia_length'] = self.stat_wikipedia_length('found')
        self.stats['notwikipedia_length'] = self.stat_wikipedia_length('not_found')
        self.stats['wikipedia_frequencies'] = self.stat_wikipedia_frequencies('found')
        self.stats['notwikipedia_frequencies'] = self.stat_wikipedia_frequencies('not_found')

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
        threshold_ids[0].update(set(self.articles.keys()))

        for tuple_ in self.tuples:
            for threshold in range(1, len(tuple_.article_ids) + 1):
                threshold_ids[threshold].update(tuple_.article_ids)

        data = [i for i in range(m + 1) for _ in threshold_ids[i]]
        bins = m + 1
        range_ = (0, m + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_wikipedia_length(self, dictionary):
        """
        Compute the number of entries in the corresponding dict of wikipedia.

        Args:
            dictionary: str, name of the dict, must be  'found' or 'not_found'.

        Returns:
            int, number of entries in the dict.
        """

        d = self.wikipedia[dictionary]

        if d is not None:
            return len(d)
        else:
            return 0

    def stat_wikipedia_frequencies(self, dictionary):
        """
        Compute the histogram of the frequencies of the tuples where appear the entities from the corresponing dict
         as a numpy.histogram.

        Args:
            dictionary: str, name of the dict, must be  'found' or 'not_found'.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        d = self.wikipedia[dictionary]

        data = [len(tuple_.article_ids) for tuple_ in self.tuples for entity in tuple_.entities if entity.name in d]
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
                length += len(self.articles[id_].contexts[str(tuple_)])

            data.append(length)

        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def display_stats_tuples(self):
        """ Display the entities tuples statistics of the database. """

        print("\nTotal number of tuples: {}".format(len(self.tuples)))
        print("\n10 most frequent tuples:")
        for tuple_ in self.tuples[:10]:
            print("{} (in {} articles)".format(str(tuple_), len(tuple_.article_ids)))
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

        print("\nTotal number of wikipedia entries found: {}/not found: {}"
              .format(self.stats['wikipedia_length'],
                      self.stats['notwikipedia_length']))

        print("\nWikipedia info of 10 most frequent tuples:\n")
        for tuple_ in self.tuples[:10]:
            for entity in tuple_.entities:
                print(entity.name, '; ', entity.wiki)
            print()

        print("\nEntities not found in wikipedia:")
        for entity in self.wikipedia['not_found']:
            print(entity)

        print()

        self.plot_hist(fig=4, data=self.stats['wikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities found in wikipedia')

        self.plot_hist(fig=5, data=self.stats['notwikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities not found in wikipedia')

    # endregion

    # region Cleaning methods

    @Verbose("Cleaning the database's articles...")
    @Attribute('articles')
    def clean_articles(self, criterion=None, to_del=None, to_keep=None):
        """
        Removes from the database the articles which meet the Article's criterion or whose ids are in to_del or are not
        in to_keep.

        Args:
            criterion: function, criterion that an article must meet to be removed.
            to_del: set, ids of the articles that must be removed.
            to_keep: set, ids of the articles that must be kept.
        """

        if criterion is not None and to_del is None and to_keep is None:
            print("Criterion: {}".format([line for line in criterion.__doc__.splitlines() if line][0][8:]))
            to_del = set()

            for id_ in self.articles:
                if criterion(self.articles[id_]):
                    to_del.add(id_)

        elif criterion is None and to_del is not None and to_keep is None:
            print("Criterion: remove the designated articles.")

        elif criterion is None and to_del is None and to_keep is not None:
            print("Criterion: keep only the designated articles.")
            to_del = set()

            for id_ in self.articles:
                if id_ not in to_keep:
                    to_del.add(id_)

        else:
            raise Exception("Either a criterion or to_del or to_keep must be specified.")

        for id_ in to_del:
            del self.articles[id_]

    @Verbose("Cleaning the database's tuples...")
    @Attribute('tuples')
    def clean_tuples(self, to_del=None, to_keep=None):
        """
        Removes from the database the tuples whose names are in to_del or are not in to_keep.

        Args:
            to_del: set, names of the tuples that must be removed.
            to_keep: set, names of the tuples that must be kept.
        """

        if to_del is not None and to_keep is None:
            print("Criterion: remove the designated tuples.")
            tuples = self.tuples
            self.tuples = []

            for tuple_ in tuples:
                if str(tuple_) not in to_del:
                    self.tuples.append(tuple_)

        elif to_del is None and to_keep is not None:
            print("Criterion: keep only the designated tuples.")
            tuples = self.tuples
            self.tuples = []

            for tuple_ in tuples:
                if str(tuple_) in to_keep:
                    self.tuples.append(tuple_)

        else:
            raise Exception("Either to_del or to_keep must be specified.")

    @Verbose("Cleaning the database's entities...")
    @Attribute('entities')
    def clean_entities(self, to_del=None, to_keep=None):
        """
        Removes from the database the entities whose names are in to_del or are not in to_keep.

        Args:
            to_del: set, names of the entities that must be removed.
            to_keep: set, names of the entities that must be kept.
        """

        if to_del is not None and to_keep is None:
            print("Criterion: remove the designated entities.")

        elif to_del is None and to_keep is not None:
            print("Criterion: keep only the designated entities.")
            to_del = set()

            for name in self.entities:
                if name not in to_keep:
                    to_del.add(name)

        else:
            raise Exception("Either to_del or to_keep must be specified.")

        for name in to_del:
            del self.entities[name]

    @Verbose("Filtering the articles, tuples and entities...")
    def filter(self, min_articles=None, min_queries=None):
        """
        Filter out the articles that doesn't respect the specified threshold on the minimum number of articles or the
        minimum number of queries.

        Args:
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
        """

        to_keep_articles, to_keep_tuples, to_keep_entities = set(), set(), set()

        if min_articles is not None and min_queries is None:
            print("Minimum number of articles: {}".format(min_articles))
            threshold = min_articles
            attribute = 'article_ids'
        elif min_articles is None and min_queries is not None:
            print("Minimum number of queries: {}".format(min_queries))
            threshold = min_queries
            attribute = 'query_ids'
        else:
            raise Exception("Either min_articles or min_queries must be specified.")

        for tuple_ in self.tuples:
            if len(getattr(tuple_, attribute)) >= threshold:
                to_keep_tuples.add(str(tuple_))
                to_keep_articles.update(tuple_.article_ids)
                to_keep_entities.update([str(entity) for entity in tuple_.entities])

        self.clean_tuples(to_keep=to_keep_tuples)
        self.clean_articles(to_keep=to_keep_articles)
        self.clean_entities(to_keep=to_keep_entities)

        self.min_articles = min_articles if min_articles is not None else self.min_articles
        self.min_queries = min_queries if min_queries is not None else self.min_queries

    # endregion

    # region File methods

    def prefix_suffix(self):
        """
        Returns the standard beginning and ending for a file path.

        Returns:
            prefix: str, beginning of the name of the file (until the name of the folder).
            suffix: str, ending of the name of the file (after the basic name of the file).
        """

        year = str(self.years[0]) if len(self.years) == 1 else str(self.years[0]) + '-' + str(self.years[-1])[2:4]
        prefix, suffix = self.project_root + 'results/' + year + '/', ''

        if self.max_size is None:
            suffix += '_sizemax'
        elif self.max_size >= 1000:
            suffix += '_size' + str(self.max_size // 1000) + 'k'
        else:
            suffix += '_size' + str(self.max_size)

        if self.shuffle:
            suffix += '_shuffle'

        if self.min_articles is not None:
            suffix += '_articles' + str(self.min_articles)
        if self.min_queries is not None:
            suffix += '_queries' + str(self.min_queries)

        if self.random_seed is not None:
            suffix += '_seed' + str(self.random_seed)

        return prefix, suffix

    def save_pkl(self, attribute_name=None, obj=None, file_name=None, folder_name='queries'):
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
            file_name = prefix + folder_name + '/' + file_name + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to save the object.")

        try:
            with open(file_name, 'wb') as f:
                dump(obj=obj, file=f, protocol=-1)

            if attribute_name is not None:
                print("Attribute {} saved at {}.".format(attribute_name, file_name))
            else:
                print("Object saved at {}.".format(file_name))

        except PicklingError:
            print("Could not save (PicklingError).")

    def load_pkl(self, attribute_name=None, file_name=None, folder_name='queries'):
        """
        Load an attribute (designated by its name) or an object from a file using pickle.

        Args:
            attribute_name: str, name of the attribute to load; if None, returns the object.
            file_name: str, name of the file to load; if None, load the file with the corresponding standard name.
            folder_name: str, name of the folder to load from.
        """

        prefix, suffix = self.prefix_suffix()

        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to load the object.")

        with open(file_name, 'rb') as f:
            obj = load(f)

        if attribute_name is not None:
            print("Attribute {} loaded from {}.".format(attribute_name, file_name))
            setattr(self, attribute_name, obj)
        else:
            print("Object loaded from {}".format(file_name))
            return obj

    def combine_pkl(self, current=True, in_names=tuple(['wikipedia_global']), out_name='wikipedia_global'):
        """
        Combines current wikipedia information and some other wikipedia files into a single file.

        Args:
            current: bool, whether to use the current wikipedia information.
            in_names: list, names of the file to combine.
            out_name: str, name of the file to write in.
        """

        out_wikipedia = {'found': dict(), 'not_found': set()}

        if current:
            print("Current wikipedia information: {} found/{} not_found...".format(len(self.wikipedia['found']),
                                                                                   len(self.wikipedia['not_found'])))

            for type_ in ['found', 'not_found']:
                out_wikipedia[type_].update(self.wikipedia[type_])

            print("Global file updated: {} found/{} not_found.\n".format(len(out_wikipedia['found']),
                                                                         len(out_wikipedia['not_found'])))

        for in_name in in_names:
            in_wikipedia = self.load_pkl(file_name=in_name, folder_name='wikipedia')

            print("File {}: {} found/{} not_found...".format(in_name,
                                                             len(in_wikipedia['found']),
                                                             len(in_wikipedia['not_found'])))

            for type_ in ['found', 'not_found']:
                out_wikipedia[type_].update(in_wikipedia[type_])

            print("Global file updated: {} found/{} not_found.\n".format(len(out_wikipedia['found']),
                                                                         len(out_wikipedia['not_found'])))

        self.save_pkl(obj=out_wikipedia, file_name=out_name, folder_name='wikipedia')

    def save_csv(self, attribute_name=None, file_name=None, folder_name='queries', limit=None):
        """
        Save a dictionary attribute to a .csv using pandas DataFrame.

        Args:
            attribute_name: str, name of the attribute to save.
            file_name: str, name of the file; if None, save an attribute with the standard name.
            folder_name: str, name of the folder to save in.
            limit: int, maximum number of data to save; if None, save all of them.
        """

        obj = getattr(self, attribute_name)
        ids = list(obj.keys())

        if limit is not None:
            shuffle(ids)
            ids = ids[:limit]

        data = [obj[id_].to_dict() for id_ in ids]
        df = DataFrame.from_records(data=data)

        prefix, suffix = self.prefix_suffix()

        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + '.pkl'
        else:
            file_name = attribute_name if limit is None else attribute_name + '_short'
            file_name = prefix + folder_name + '/' + file_name + suffix + '.csv'

        df.to_csv(file_name, index=False)

        print("Attribute {} saved at {}".format(attribute_name, file_name))

    def write_debug(self, field, method):
        """
        Write the debugging of a method into a text file.

        Args:
            field: str, field of the database we want to debug.
            method: str, name of the method to debug.
        """

        if field == 'articles':
            lines = [[id_, getattr(article, 'debug_' + method)()] for id_, article in self.articles.items()]

        elif field == 'entities':
            lines = [[name, entity.debug_entities()] for name, entity in self.entities.items()]

        elif field == 'tuples':
            lines = [[str(tuple_), tuple_.debug_tuples()] for tuple_ in self.tuples]

        elif field == 'wikipedia':
            lines = [[name, wikipedia.debug_wikipedia()] for name, wikipedia in self.wikipedia['found'].items()] \
                    + [[name, ': not found'] for name in self.wikipedia['not_found']]

        elif field == 'queries':
            lines = [[id_, query.debug_queries()] for id_, query in self.queries.items()]

        else:
            raise Exception("Wrong field/method specified: {}/{}.".format(field, method))

        lines = [line[0] + line[1] + '\n' for line in lines if line[1]]

        if lines:
            prefix, _ = self.prefix_suffix()
            file_name = prefix + 'debug/' + method + '.txt'

            with open(file_name, 'w') as f:
                f.writelines(lines)

            print("Debugging Written in {}...".format(file_name))

    # endregion

    # region Other methods

    @staticmethod
    def progression(count, modulo, size, text):
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
            print("   " + text + " {}/{}...".format(count, size))

        return count

    def paths(self):
        """
        Compute the paths of the data files of the database.

        Returns:
            list, sorted file paths of the data of the articles.
        """

        patterns = [self.project_root + 'databases/nyt_jingyun/data/' + str(year) + '/*/*/*.xml' for year in self.years]

        paths = []
        for pattern in patterns:
            paths.extend(glob(pattern))
        paths.sort()

        if self.shuffle:
            shuffle(paths)
            paths = paths[:self.max_size] if self.max_size is not None else paths
            paths.sort()

        else:
            paths = paths[:self.max_size] if self.max_size is not None else paths

        return paths

    @staticmethod
    def subtuples(l):
        """
        Compute all the possible sorted subtuples of len > 1 from a list.

        Args:
            l: list, original list.

        Returns:
            set, all the possible subtuples of len > 1 of l.
        """

        if len(l) < 2:
            return set()

        elif len(l) == 2 or len(l) > 10:
            return {tuple(sorted(l))}

        else:
            res = {tuple(sorted(l))}
            for x in l:
                res = res.union(Database.subtuples([y for y in l if y != x]))

            return res

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

    # endregion


def main():
    max_size = 1000
    min_articles = 1
    min_queries = 1

    database = Database(project_root='../', max_size=max_size, min_articles=min_articles, min_queries=min_queries)

    database.preprocess_database()
    database.process_articles()

    database.process_wikipedia(load=False)
    database.process_queries(load=False)
    return


if __name__ == '__main__':
    main()
