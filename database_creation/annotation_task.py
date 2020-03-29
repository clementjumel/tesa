from database_creation.nyt_article import Article
from database_creation.utils import Tuple, Wikipedia, Query, Annotation

from numpy import split as np_split
from numpy.random import seed, choice
from time import time
from glob import glob
from collections import defaultdict
from pickle import dump, load, PicklingError
from pandas import DataFrame, read_csv
from unidecode import unidecode
from wikipedia import search, page, WikipediaException, DisambiguationError
from xml.etree.ElementTree import ParseError
from itertools import chain, combinations
from re import findall


class AnnotationTask:

    def __init__(self, years, max_tuple_size, short, random, debug, random_seed, save, silent, corpus_path,
                 results_path, root=''):
        """
        Initializes an instance of AnnotationTask, which creates the queries asked to the annotation workers and gathers
        their answers.

        Args:
            years: it, years (int) of the database to analyse.
            max_tuple_size: int, maximum size of the entities tuple to compute.
            short: bool, if True, limit the dataset to 10 000 initial articles.
            random: bool, if short, whether to pick the 10 000 at random or take the first ones.
            debug: bool, whether to, for each step, write its effect in a text file.
            random_seed: int, the seed to use for the random processes of numpy.
            save: bool, saving option.
            silent: bool, silence option.
            corpus_path: str, path to the NYT annotated corpus.
            results_path: str, path to the results folder
            root: str, path to the root of the project.
        """

        self.years = years
        self.max_tuple_size = max_tuple_size
        self.short = short
        self.random = random
        self.debug = debug
        self.random_seed = random_seed
        self.save = save
        self.silent = silent
        self.corpus_path = corpus_path
        self.results_path = results_path
        self.root = root

        self.articles = None
        self.entities = None
        self.tuples = None
        self.wikipedia = None
        self.queries = None
        self.task = None
        self.annotations = None

        self.modulo_articles = 500
        self.modulo_tuples = 1000
        self.modulo_entities = 100

        seed(random_seed)

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

                slf = args[0]
                silent = getattr(slf, "silent")

                t0 = time()
                print(self.message) if not silent else None

                res = func(*args, **kwargs)

                print("Done; elapsed time: %is.\n".format(round(time() - t0))) if not silent else None

                return res

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
                silent = getattr(slf, "silent")

                attribute = getattr(slf, self.attribute)
                length = len(attribute) if attribute is not None else 0

                print("Initial length of %s: %i." % (self.attribute, length)) if not silent else None

                res = func(*args, **kwargs)

                attribute = getattr(slf, self.attribute)
                length = len(attribute) if attribute is not None else 0

                print("Final length of %s: %i." % (self.attribute, length)) if not silent else None

                return res

            return f

    # endregion

    # region Main methods

    @Verbose("Preprocessing the database...")
    def preprocess_database(self):
        """ Performs the preprocessing of the database. """

        self.compute_articles()
        self.clean_articles(criterion=Article.criterion_content, to_keep=None)
        self.compute_metadata()
        self.compute_entities()
        self.compute_tuples()
        self.filter_no_tuple()

    @Verbose("Preprocessing the articles...")
    def process_articles(self):
        """ Performs the preprocessing of the articles. """

        self.compute_annotations()
        self.compute_contexts()
        self.filter_no_query()

    @Verbose("Processing the wikipedia information...")
    def process_wikipedia(self, load, file_name):
        """
        Performs the processing of the wikipedia information of the database, or load it.

        Args:
            load: bool, if True, load an existing file, else, computes it.
            file_name: str, name of the wikipedia file to save or load; if None, deal with the standard files name.
        """

        if load:
            self.load_attr_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')

        self.compute_wikipedia(load=load)

        if self.save:
            self.save_attr_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')

    @Verbose("Processing the aggregation queries...")
    def process_queries(self, load):
        """
        Performs the processing of the annotation queries.

        Args:
            load: bool, if True, load an existing file.
        """

        if load:
            self.load_attr_pkl(attribute_name='queries', file_name=None, folder_name='queries')

        else:
            self.compute_queries()

            if self.save:
                self.save_attr_pkl(attribute_name='queries', file_name=None, folder_name='queries')

    @Verbose("Processing the annotations batches...")
    def process_annotation_batches(self, batches, batch_size, exclude_pilot):
        """
        Saves a csv file for each batch of annotation task in queries/. If some csv files are already in queries/,
        generate different queries than those already in there.

        Args:
            batches: int, number of batches to create.
            batch_size: int, size of the batches.
            exclude_pilot: bool, whether or not to take into account the pilot annotations.
        """

        existing_ids, existing_batches = self.read_existing_batches(exclude_pilot=exclude_pilot)

        if self.save:
            self.save_annotation_batches(batches=batches,
                                         batch_size=batch_size,
                                         existing_ids=existing_ids,
                                         existing_batches=existing_batches)

    @Verbose("Processing the modeling task...")
    def process_task(self, exclude_pilot):
        """
        Process the annotations and the corresponding queries.

        Args:
            exclude_pilot: whether or not to exclude the data from the pilot.
        """

        self.compute_annotated_queries(exclude_pilot=exclude_pilot)
        self.compute_annotations(exclude_pilot=exclude_pilot)

    @Verbose("Combining the wikipedia files...")
    def combine_wiki(self, current, in_names, out_name):
        """
        Combines current wikipedia information and some other wikipedia files into a single file. Note that the
        most up to date information should come from the last file form in_names.

        Args:
            current: bool, whether to use the current wikipedia information.
            in_names: list, names of the file to combine.
            out_name: str, name of the file to write in.
        """

        out_wikipedia = {'found': dict(), 'not_found': set()}

        if current:
            if not self.silent:
                print("Current wikipedia information: %i found/%i not_found..." % (len(self.wikipedia['found']),
                                                                                   len(self.wikipedia['not_found'])))

            for type_ in ['found', 'not_found']:
                out_wikipedia[type_].update(self.wikipedia[type_])

            if not self.silent:
                print("Global file updated: %i found/%i not_found.\n" % (len(out_wikipedia['found']),
                                                                         len(out_wikipedia['not_found'])))

        for in_name in in_names:
            in_wikipedia = self.load_obj_pkl(file_name=in_name, folder_name='wikipedia')

            if not self.silent:
                print("File %s: %i found/%i not_found..." % (in_name,
                                                             len(in_wikipedia['found']),
                                                             len(in_wikipedia['not_found'])))

            for type_ in ['found', 'not_found']:
                out_wikipedia[type_].update(in_wikipedia[type_])

            if not self.silent:
                print("Global file updated: %i found/%i not_found.\n" % (len(out_wikipedia['found']),
                                                                         len(out_wikipedia['not_found'])))

        if self.save:
            self.save_obj_pkl(obj=out_wikipedia, file_name=out_name, folder_name='wikipedia')

        else:
            if not self.silent:
                print("Warning, the changes were not saved.")

        self.wikipedia = out_wikipedia

    @Verbose("Solving manually the wikipedia issues...")
    def correct_wiki(self, step, out_name):
        """
        Run the manual correction of the wikipedia tricky cases.

        Args:
            step: int, step of the correction to perform, between 1 and 4.
            out_name: str, name of the wikipedia file to save; if None, deal with the standard files name.
        """

        self.correction(step=step)

        if self.save:
            self.save_attr_pkl(attribute_name='wikipedia', file_name=out_name, folder_name='wikipedia')

        else:
            if not self.silent:
                print("Warning, the changes were not saved.")

    # endregion

    # region Methods compute_

    @Verbose("Computing the database' article...")
    @Attribute('articles')
    def compute_articles(self):
        """ Computes and initializes the articles in the database. """

        corpus_root = self.root + self.corpus_path
        patterns = [corpus_root + 'data/' + str(year) + '/*/*/*.xml' for year in self.years]
        paths = [path for pattern in patterns for path in glob(pattern)]

        if self.short:
            max_size = 10000

            if not self.random:
                paths.sort()
                paths = paths[:max_size]
            else:
                paths = choice(a=paths, size=max_size, replace=False)
                paths.sort()
        else:
            paths.sort()

        articles = {}

        for data_path in paths:
            id_ = data_path.split('/')[-1].split('.')[0]
            year = data_path.split('/')[-4]

            content_path = corpus_root + 'content_annotated/' + str(year) + 'content_annotated/' + id_ + '.txt.xml'
            summary_path = corpus_root + 'summary_annotated/' + str(year) + 'summary_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(data_path=data_path, content_path=content_path, summary_path=summary_path)

        self.articles = articles

        if self.debug:
            self.write_debug(field='articles', method='articles')

    @Verbose("Computing the articles' metadata...")
    def compute_metadata(self):
        """ Computes the metadata of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count=count, modulo=self.modulo_articles, size=size, text='article')

            self.articles[id_].compute_metadata()

        if self.debug:
            self.write_debug(field='articles', method='metadata')

    @Verbose("Computing the database' entities...")
    @Attribute('entities')
    def compute_entities(self):
        """ Compute the entities of the database. """

        self.entities = dict()

        count, size = 0, len(self.articles)
        for _, article in self.articles.items():
            count = self.progression(count=count, modulo=self.modulo_articles, size=size, text='article')

            try:
                entities = article.get_entities()
            except AssertionError:
                entities = []
                if not self.silent:
                    print("Several entities have the same name (%s); ignoring them..." %
                          '; '.join(article.get_vanilla_entities()))

            for entity in entities:
                if str(entity) in self.entities:
                    try:
                        self.entities[str(entity)].update_info(entity)
                    except AssertionError:
                        if not self.silent:
                            print("%s corresponds to both %s and %s, ignoring the later..." %
                                  (str(entity), entity.type_, self.entities[str(entity)].type_))

                else:
                    self.entities[str(entity)] = entity

            article.entities = [self.entities[name] for name in [str(entity) for entity in entities]]

        if self.debug:
            self.write_debug(field='articles', method='article_entities')
            self.write_debug(field='entities', method='entities')

    @Verbose("Computing the entity tuples...")
    @Attribute('tuples')
    def compute_tuples(self):
        """ Compute the Tuples of the database as a sorted list of Tuples (by number of articles). """

        def subtuples(s, max_size):
            """
            Compute all the possible sorted subtuples of len between 2 and max_size from a set s.

            Args:
                s: set, original set.
                max_size: int, maximal size of the tuples.

            Returns:
                set, all the possible sorted subtuples.
            """

            s = sorted(s)
            min_len, max_len = 2, min(len(s), max_size)

            return set(chain.from_iterable(combinations(s, r) for r in range(min_len, max_len + 1)))

        ids = defaultdict(set)

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count=count, modulo=self.modulo_articles, size=size, text='article')

            entities = defaultdict(set)
            for entity in self.articles[id_].entities:
                entities[entity.type_].add(entity.name)

            for type_ in entities:
                for tuple_ in subtuples(s=entities[type_], max_size=self.max_tuple_size):
                    ids[tuple_].add(id_)

        ranking = sorted(ids, key=lambda k: (len(ids[k]), str(k)), reverse=True)

        self.tuples = [Tuple(id_=str(rank + 1),
                             entities=tuple([self.entities[name] for name in tuple_]),
                             article_ids=ids[tuple_])
                       for rank, tuple_ in enumerate(ranking)]

        if self.debug:
            self.write_debug(field='tuples', method='tuples')

    @Verbose("Computing the articles' annotations from the corpus...")
    def compute_corpus_annotations(self):
        """ Computes the corpus annotations of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count=count, modulo=self.modulo_articles, size=size, text='article')

            try:
                self.articles[id_].compute_corpus_annotations()

            except ParseError:
                print("Data is not clean, remove data %s and start again." % id_)
                raise Exception

        if self.debug:
            self.write_debug(field='articles', method='annotations')

    @Verbose("Computing the contexts...")
    def compute_contexts(self):
        """ Compute the contexts of the articles for each Tuple. """

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count=count, modulo=self.modulo_tuples, size=size, text='tuple')

            query_ids = set()

            for article_id_ in tuple_.article_ids:
                self.articles[article_id_].compute_contexts(tuple_=tuple_)

                query_ids.update({tuple_.id_ + '_' + article_id_ + '_' + context_id_
                                  for context_id_ in self.articles[article_id_].contexts[str(tuple_)]})

            tuple_.query_ids = query_ids

        if self.debug:
            self.write_debug(field='articles', method='contexts')

    @Verbose("Computing the Wikipedia information...")
    def compute_wikipedia(self, load):
        """
        Compute the wikipedia information about the entities from self.tuples.

        Args:
            load: bool, if True, load an existing file.
        """

        wikipedia = self.wikipedia if load else {'found': dict(), 'not_found': set()}

        if not self.silent:
            print("Initial entries: %i found/%i not found." % (len(wikipedia['found']), len(wikipedia['not_found'])))

        try:
            count, size = 0, len(self.entities)
            for name, entity in self.entities.items():
                count = self.progression(count=count, modulo=self.modulo_entities, size=size, text='entity')

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

        except (KeyboardInterrupt, WikipediaException) as err:
            if not self.silent:
                print("An error occurred, saving the loaded information and leaving... (%s)" % err)

        if not self.silent:
            print("Final entries: %i found/%i not found." % (len(wikipedia['found']), len(wikipedia['not_found'])))

        self.wikipedia = wikipedia

        if self.debug:
            self.write_debug(field='wikipedia', method='wikipedia')

    @Verbose("Computing the Queries...")
    @Attribute('queries')
    def compute_queries(self):
        """ Compute the Queries of the database. """

        queries = dict()

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count=count, modulo=self.modulo_tuples, size=size, text='tuple')

            for article_id_ in sorted(tuple_.article_ids):
                article_contexts = self.articles[article_id_].contexts[str(tuple_)]

                for context_id_, context in article_contexts.items():
                    query_id_ = '_'.join([article_id_, tuple_.id_, context_id_])
                    queries[query_id_] = Query(id_=query_id_,
                                               tuple_=tuple_,
                                               article=self.articles[article_id_],
                                               context=context)

        self.queries = queries

        if self.debug:
            self.write_debug(field='queries', method='queries')

    @Verbose("Computing the annotated queries...")
    @Attribute('queries')
    def compute_annotated_queries(self, exclude_pilot):
        """
        Compute the queries corresponding to the annotations.

        Args:
            exclude_pilot: whether or not to exclude the data from the pilot.
        """

        queries = dict()

        for path in sorted(glob(self.results_path + 'annotations/*/task/*.pkl')):
            path = path.split(self.results_path)[1]
            version = path.split('/')[1]

            if 'pilot' not in version or not exclude_pilot:
                folder_name = '/'.join(path.split('/')[:-1])
                file_name = path.split('/')[-1].split('.pkl')[0]

                queries.update(self.load_obj_pkl(file_name=file_name, folder_name=folder_name))

        self.queries = queries

    @Verbose("Computing the annotations...")
    @Attribute('annotations')
    def compute_annotations(self, exclude_pilot):
        """
        Compute the annotations of the Mechanical Turks.

        Args:
            exclude_pilot: whether or not to exclude the data from the pilot.
        """

        annotations = defaultdict(list)

        for path in sorted(glob(self.results_path + 'annotations/*/results/*.csv')):
            path = path.split(self.results_path)[1]
            version = path.split('/')[1]
            batch = path.split('/')[-1].replace('_complete.csv', '')

            if 'pilot' not in version or not exclude_pilot:
                df = read_csv(self.results_path + path)

                if not self.silent:
                    print("%s loaded from %s" % (batch, path))

                for _, row in df.iterrows():
                    id_ = row.get('Input.id_')
                    annotations[id_].append(Annotation(id_=id_,
                                                       version=version,
                                                       batch=batch,
                                                       row=row,
                                                       silent=self.silent))

        self.annotations = annotations

    # endregion

    # region Cleaning methods

    @Verbose("Cleaning the database's articles...")
    @Attribute('articles')
    def clean_articles(self, criterion, to_keep):
        """
        Removes from the database the articles which meet the Article's criterion or whose ids are not in to_keep.

        Args:
            criterion: function, criterion that an article must meet to be removed.
            to_keep: set, ids of the articles that must be kept.
        """

        to_del = set()

        if criterion is not None and to_keep is None:
            if not self.silent:
                print("Criterion: %s" % [line for line in criterion.__doc__.splitlines() if line][0][8:])

            for id_ in self.articles:
                if criterion(self.articles[id_]):
                    to_del.add(id_)

        elif criterion is None and to_keep is not None:
            if not self.silent:
                print("Criterion: keep only the designated articles.")

            for id_ in self.articles:
                if id_ not in to_keep:
                    to_del.add(id_)

        else:
            raise Exception("Either a criterion or to_keep must be specified.")

        for id_ in to_del:
            del self.articles[id_]

    @Verbose("Cleaning the database's tuples...")
    @Attribute('tuples')
    def clean_tuples(self, to_keep):
        """
        Removes from the database the tuples whose names are not in to_keep.

        Args:
            to_keep: set, names of the tuples that must be kept.
        """

        if not self.silent:
            print("Criterion: keep only the designated tuples.")

        tuples = self.tuples
        self.tuples = []

        for tuple_ in tuples:
            if str(tuple_) in to_keep:
                self.tuples.append(tuple_)

    @Verbose("Cleaning the database's entities...")
    @Attribute('entities')
    def clean_entities(self, to_keep):
        """
        Removes from the database the entities whose names are not in to_keep.

        Args:
            to_keep: set, names of the entities that must be kept.
        """

        if not self.silent:
            print("Criterion: keep only the designated entities.")

        to_del = set()

        for name in self.entities:
            if name not in to_keep:
                to_del.add(name)

        for name in to_del:
            del self.entities[name]

    @Verbose("Filtering the articles and entities that correspond to no tuple...")
    def filter_no_tuple(self):
        """ Filter out the articles and entities that correspond to no tuple. """

        to_keep_articles, to_keep_entities = set(), set()

        for tuple_ in self.tuples:
            if len(tuple_.article_ids) >= 1:
                to_keep_articles.update(tuple_.article_ids)
                to_keep_entities.update([str(entity) for entity in tuple_.entities])

        self.clean_articles(criterion=None, to_keep=to_keep_articles)
        self.clean_entities(to_keep=to_keep_entities)

    @Verbose("Filtering the articles, tuples and entities that correspond to no query...")
    def filter_no_query(self):
        """ Filter out the articles that correspond to no query. """

        to_keep_articles, to_keep_tuples, to_keep_entities = set(), set(), set()

        for tuple_ in self.tuples:
            if len(tuple_.query_ids) >= 1:
                to_keep_tuples.add(str(tuple_))
                to_keep_articles.update(tuple_.article_ids)
                to_keep_entities.update([str(entity) for entity in tuple_.entities])

        self.clean_tuples(to_keep=to_keep_tuples)
        self.clean_articles(criterion=None, to_keep=to_keep_articles)
        self.clean_entities(to_keep=to_keep_entities)

    # endregion

    # region File methods

    def file_name_suffix(self):
        """
        Returns a standardized ending for a file name.

        Returns:
            str, ending of the name of the file (after the basic name of the file).
        """

        suffix = ['_short' if self.short else None,
                  '_random' if self.random else None,
                  '_seed_' + str(self.random_seed) if self.random_seed is not None else None]

        suffix = [s for s in suffix if s is not None]
        suffix = ''.join(suffix)

        return suffix

    def save_attr_pkl(self, attribute_name, file_name, folder_name):
        """
        Save an attribute designated by its name using pickle.

        Args:
            attribute_name: str, name of the attribute to save.
            file_name: str, name of the file; if None, save an attribute with the attribute_name.
            folder_name: str, name of the folder to save in.
        """

        file_name = file_name or attribute_name + self.file_name_suffix()
        obj = getattr(self, attribute_name)

        self.save_obj_pkl(obj=obj, file_name=file_name, folder_name=folder_name)

    def save_obj_pkl(self, obj, file_name, folder_name):
        """
        Save an object using pickle.

        Args:
            obj: unknown type, object to save.
            file_name: str, name of the file.
            folder_name: str, name of the folder to save in.
        """

        file_name = self.results_path + folder_name + "/" + file_name + ".pkl"

        try:
            with open(file_name, "wb") as file:
                dump(obj=obj, file=file, protocol=-1)

            if not self.silent:
                print("Object saved at %s." % file_name)

        except PicklingError as err:
            print("Could not save (PicklingError), moving on:", err)

    def load_attr_pkl(self, attribute_name, file_name, folder_name):
        """
        Load an attribute designated by its name using pickle.

        Args:
            attribute_name: str, name of the attribute to load.
            file_name: str, name of the file to load; if None, load the file with the corresponding attribute_name.
            folder_name: str, name of the folder to load from.
        """

        file_name = file_name or attribute_name + self.file_name_suffix()
        obj = self.load_obj_pkl(file_name=file_name, folder_name=folder_name)

        setattr(self, attribute_name, obj)

    def load_obj_pkl(self, file_name, folder_name):
        """
        Load an object using pickle.

        Args:
            file_name: str, name of the file to load.
            folder_name: str, name of the folder to load from.
        """

        file_name = self.results_path + folder_name + "/" + file_name + ".pkl"

        with open(file_name, 'rb') as file:
            obj = load(file=file)

        if not self.silent:
            print("Object loaded from %s." % file_name)

        return obj

    @Verbose("Reading the existing annotation batches...")
    def read_existing_batches(self, exclude_pilot):
        """
        Read in the folder queries and annotations the query ids and the batch indexes of the existing annotation
        batches (in .csv files).

        Args:
            exclude_pilot: bool, whether or not to take into account the pilot annotations.

        Returns:
            existing_ids: set, ids in the existing annotation batches.
            existing_batches: set, indexes of the existing annotation batches.
        """

        ids, idxs = set(), set()

        for path in glob(self.results_path + "queries/*.csv"):
            batch = path.split("/")[-1].split(".")[0]

            if not exclude_pilot or "pilot" not in batch:
                df = read_csv(path)
                df_ids = set([row.get('id_') for _, row in df.iterrows()])

                ids.update(df_ids)

                if batch.split("_")[0] == "batch":
                    idx = int(batch.split("_")[-1])
                    idxs.add(idx)

                if not self.silent:
                    print("Reading %s from results/queries/ folder (%i queries)." % (batch, len(df_ids)))

        for path in glob(self.results_path + "task_annotation/*/task/*.csv"):
            version = path.split("/")[-3]
            batch = path.split("/")[-1].split(".")[0]

            if not exclude_pilot or 'pilot' not in version:
                df = read_csv(path)
                df_ids = set([row.get('id_') for _, row in df.iterrows()])

                ids.update(df_ids)

                if batch.split("_")[0] == "batch":
                    idx = int(batch.split("_")[-1])
                    idxs.add(idx)

                if not self.silent:
                    print("Reading existing batch %s from %s (%i queries)." % (batch, version, len(df_ids)))

        return ids, idxs

    @Verbose("Saving new annotation batches...")
    def save_annotation_batches(self, batches, batch_size, existing_ids, existing_batches):
        """
        Save annotation batches in .csv files. Don't save queries that have been already saved.

        Args:
            batches: int, number of batches to create.
            batch_size: int, size of the batches.
            existing_ids: set, ids in the existing annotation batches.
            existing_batches: set, indexes of the existing annotation batches.
        """

        all_ids = set(self.queries.keys())
        ids = all_ids.difference(existing_ids)

        if not self.silent:
            print("Removing %i existing queries from the %i total queries; %i remaining queries." %
                  (len(existing_ids), len(all_ids), len(ids)))

        batches_ids = choice(a=sorted(ids), size=batches*batch_size, replace=False)
        batches_ids = np_split(batches_ids, batches)

        starting_idx = max(existing_batches) + 1

        for batch in range(batches):
            batch_ids = batches_ids[batch]
            data = [self.queries[id_].to_html() for id_ in batch_ids]

            df = DataFrame.from_records(data=data)

            batch_idx = starting_idx + batch
            batch_idx = "0" + str(batch_idx) if 0 <= batch_idx < 10 else str(batch_idx)
            file_name = self.results_path + "queries/batch_" + batch_idx + ".csv"

            df.to_csv(file_name, index=False)

            if not self.silent:
                print("batch_%s saved at %s." % (batch_idx, file_name))

    def write_debug(self, field, method):
        """
        Write the debugging of a method into a text file.

        Args:
            field: str, field of the database we want to debug.
            method: str, name of the method to debug.
        """

        if field == "articles":
            lines = [[id_, getattr(article, "debug_" + method)()] for id_, article in self.articles.items()]

        elif field == "entities":
            lines = [[name, entity.debug_entities()] for name, entity in self.entities.items()]

        elif field == "tuples":
            lines = [[str(tuple_), tuple_.debug_tuples()] for tuple_ in self.tuples]

        elif field == "wikipedia":
            lines = [[name, wikipedia.debug_wikipedia()] for name, wikipedia in self.wikipedia['found'].items()] \
                    + [[name, ": not found"] for name in self.wikipedia['not_found']]

        elif field == "queries":
            lines = [[id_, query.debug_queries()] for id_, query in self.queries.items()]

        else:
            raise Exception("Wrong field/method specified: %s/%s." % (field, method))

        lines = [line[0] + line[1] + '\n' for line in lines if line[1]]

        if lines:
            file_name = self.results_path + "debug/" + method + ".txt"

            if self.save:
                with open(file_name, "w") as f:
                    f.writelines(lines)

            if not self.silent:
                print("Debugging Written in %s..." % file_name)

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

    def correction(self, step):
        """
        Performs the manual correction of the wikipedia information.

        Args:
            step: int, step of the correction to perform, between 1 and 4.
        """

        to_correct = set([name for name, wiki in self.wikipedia['found'].items() if not wiki.exact])
        corrected = set()

        if not to_correct:
            if not self.silent:
                print("All the %i entities are exact, no correction to be made." % len(self.wikipedia['found']))
            return

        print("Entities to correct: %i/%i." % (len(to_correct), len(self.wikipedia['found'])))

        try:
            if step == 1:
                count, size = 0, len(to_correct)
                for name in sorted(to_correct):
                    count = self.progression(count=count, modulo=self.modulo_entities, size=size,
                                             text="to correct entity")

                    preprocessed_name_1 = unidecode(name).lower().replace(".", "")
                    preprocessed_name_2 = " ".join([word for word in preprocessed_name_1.split() if len(word) > 1])

                    title = self.wikipedia['found'][name].title
                    before_parenthesis = findall(r'(.*?)\s*\(', title)
                    before_parenthesis = before_parenthesis[0] if before_parenthesis and before_parenthesis[0] \
                        else title

                    preprocessed_title_1 = unidecode(before_parenthesis).lower().replace(".", "")
                    preprocessed_title_2 = " ".join([word for word in preprocessed_title_1.split() if len(word) > 1])

                    if preprocessed_name_1 == preprocessed_title_1 or preprocessed_name_2 == preprocessed_title_2:
                        self.wikipedia['found'][name].exact = True
                        corrected.add(name)

                to_correct, corrected = to_correct.difference(corrected), set()
                print("First step over, remaining: %i/%i." % (len(to_correct), len(self.wikipedia['found'])))

            elif step == 2:
                count, size = 0, len(to_correct)
                for name in sorted(to_correct):
                    count = self.progression(count=count, modulo=self.modulo_entities, size=size,
                                             text="to correct entity")

                    while True:
                        answer = input(name + "/" + self.wikipedia['found'][name].title + ": is this good? [y/n/o/d]")
                        if answer in ["y", "n", "o", "d"]:
                            break
                        else:
                            print('Answer should be "y" (yes), "n" (no), "o" (open) or "d" (discard), try again.')

                    if answer == "o":
                        while True:
                            answer = input(self.wikipedia['found'][name].get_info() + ": is this good? [y/n/d]")
                            if answer in ["y", "n", "d"]:
                                break
                            else:
                                print('Answer should be "y" (yes), "n" (no) or "d" (discard), try again.')

                    if answer == "y":
                        self.wikipedia['found'][name].exact = True
                        corrected.add(name)

                    elif answer == "d":
                        del self.wikipedia['found'][name]
                        self.wikipedia['not_found'].add(name)
                        corrected.add(name)

                to_correct, corrected = to_correct.difference(corrected), set()
                print("Second step over, remaining: %i/%i." % (len(to_correct), len(self.wikipedia['found'])))

            elif step == 3:
                count, size = 0, len(to_correct)
                for name in sorted(to_correct):
                    count = self.progression(count=count, modulo=self.modulo_entities, size=size,
                                             text='to correct entity')

                    wiki_search = search(name)
                    print("Wikipedia search for %s:" % name)
                    for cmpt, title in enumerate(wiki_search):
                        print("%s: %s" % (str(cmpt + 1), + title))

                    while True:
                        try:
                            answer = int(input("Which number is the good one? (0 for giving up this example)"))
                            if answer in range(len(wiki_search) + 1):
                                break
                            else:
                                print("Answer should be between 0 and the length of the wikipedia search, try again.")
                        except ValueError:
                            print("Answer should be an int, try again.")

                    if answer == 0:
                        del self.wikipedia['found'][name]
                        self.wikipedia['not_found'].add(name)
                        corrected.add(name)
                        print("Considered not found.")

                    else:
                        try:
                            p = page(wiki_search[answer - 1])
                            self.wikipedia['found'][name] = Wikipedia(p)

                        except DisambiguationError:
                            print("Search is still ambiguous, moving on to the next one...")

                to_correct, corrected = to_correct.difference(corrected), set()
                print("Third step over, remaining: %i/%i." % (len(to_correct), len(self.wikipedia['found'])))

            elif step == 4:
                count, size = 0, len(to_correct)
                for name in sorted(to_correct):
                    count = self.progression(count=count, modulo=self.modulo_entities, size=size,
                                             text='to correct entity')

                    del self.wikipedia['found'][name]
                    self.wikipedia['not_found'].add(name)
                    corrected.add(name)

                to_correct, corrected = to_correct.difference(corrected), set()
                print("Fifth step over, remaining: %i/%i.".format(len(to_correct), len(self.wikipedia['found'])))

            else:
                raise Exception("Wrong step specified.")

        except KeyboardInterrupt:
            print("Keyboard interruption, saving the results...")

    # endregion
