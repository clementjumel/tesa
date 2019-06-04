from database_creation.utils import BaseClass
from database_creation.article import Article

from copy import copy
from numpy import random
from glob import glob
from collections import defaultdict


class Database(BaseClass):
    # region Class initialization

    to_print,  = ['articles']
    print_attributes, print_lines, print_offsets = False, 2, 0
    limit_print, random_print = 50, True
    count_modulo = 1000

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

        # TODO: change to stats
        self.entity_tuples = None
        self.entity_tuples_articles = None

        self.compute_articles()
        self.compute_size()

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

    @BaseClass.Verbose("Preprocessing the articles (standard)...")
    def preprocess_standard(self):
        """ Performs the standard preprocessing of the database. """

        self.clean(Article.criterion_data)

        count = 0

        for id_ in self.articles:
            count = self.progression(count)

            try:
                self.articles[id_].preprocess()
            except AssertionError:
                print(id_)
                raise AssertionError

        self.clean(Article.criterion_entity)

    @BaseClass.Verbose("Preprocessing the articles (most frequent tuples)...")
    def preprocess_contexts(self, limit=100, display=False):
        """
        Performs a preprocessing of the database to isolate articles with frequent entity tuples..

        Args:
            limit: int, maximum number of tuples.
            display: bool, whether or not to display the ranking of the tuples.
        """

        self.clean(Article.criterion_data)

        count = 0

        for id_ in self.articles:
            count = self.progression(count)

            self.articles[id_].compute_entities()

        self.clean(Article.criterion_entity)

        self.compute_entity_tuples(limit=limit, display=display)
        self.clean(Database.criterion_rare_entities)

        self.preprocess_standard()

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

        self.clean(Article.criterion_context)

    # TODO: finish
    def process_tuple(self, idx):
        entity_tuple, ids = self.entity_tuples[idx]
        count = 0

        print("Entity tuples: {}\n\n".format(self.to_string(list(entity_tuple))))

        for id_ in ids:
            try:
                article = self.articles[id_]
            except KeyError:
                continue

            if article.tuple_contexts is not None:
                try:
                    context = article.tuple_contexts[entity_tuple]
                    count += len(list(context.keys())) - 1
                    print(self.to_string(context) + '\n\n')

                except KeyError:
                    pass

        print("\n{} samples out of {} articles".format(count, len(ids)))

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

    @BaseClass.Verbose("Computing the most frequent entities...")
    def compute_entity_tuples(self, limit, display):
        """
        Compute the most frequent tuples of entities.

        Args:
            limit: int, maximum number of tuples.
            display: bool, whether or not to display the ranking of the tuples.
        """

        entities_dict, count = defaultdict(set), 0

        for id_ in self.articles:
            count = self.progression(count)

            for entity_type in ['locations', 'persons', 'organizations']:
                entities = getattr(self.articles[id_], 'entities_' + entity_type)

                if entities and len(entities) >= 2:
                    entities.sort()

                    for t in self.subtuples(entities):
                        entities_dict[t].add(id_)

        sorted_tuples = sorted(entities_dict, key=lambda k: len(entities_dict[k]), reverse=True)[0:limit]

        if display:
            print('\nMost frequent entity tuples:')
            for t in sorted_tuples:
                print(self.to_string(list(t)) + ' (' + str(len(entities_dict[t])) + ')')
            print('')

        self.entity_tuples = [[t, entities_dict[t]] for t in sorted_tuples]
        self.entity_tuples_articles = \
            set([item for subset in [entities_dict[t] for t in sorted_tuples] for item in subset])

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

    def criterion_rare_entities(self, id_):
        """
        Check if an article's entities contain the most frequent tuples of entities.

        Args:
            id_: string, ID of the article to analyze.

        Returns:
            bool, True iff the article's entities does not contain frequent entities.
        """

        return True if id_ not in self.entity_tuples_articles else False

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

    # endregion


def main():
    d = Database(max_size=1000, root='../databases/nyt_jingyun')

    d.preprocess_contexts(limit=10, display=True)
    d.process_contexts()

    # Database.set_parameters(to_print=[], print_attribute=True)
    print(d)


if __name__ == '__main__':
    main()
