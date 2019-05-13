from database_creation.utils import BaseClass
from database_creation.article import Article

from copy import copy
from numpy import random
from time import time
from glob import glob


class Database(BaseClass):
    # region Class initialization

    to_print, print_attributes, print_lines, print_offsets = ['articles'], False, 2, 0
    limit_print, random_print = 50, False
    verbose, count_modulo = True, 10000

    def __init__(self, max_size=None):
        """
        Initializes an instance of Database.

        Args:
            max_size: int, maximum number of articles in the database; if None, takes all articles.
        """

        self.max_size = max_size

        self.year = '2000'
        self.root = '../databases/nyt_jingyun'

        self.articles = None
        self.size = None

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
                string += self.prefix(print_attribute, print_lines if string else 0, print_offsets, 'id ' + str(id_)) \
                          + s
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

    # region Decorator

    class Decorator:

        def __init__(self, message):
            """ Initializes the Verbose decorator message. """

            self.message = message

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                slf = args[0]

                if slf.verbose:
                    t0 = time()
                    print(self.message)

                func(*args, **kwargs)

                if slf.verbose:
                    print("Done ({} articles, elapsed time: {}s).\n".format(slf.size, round(time() - t0)))

            return f

    # endregion

    # region Main methods

    @Decorator("Preprocessing the articles...")
    def preprocess(self):
        """ Performs the preprocessing of the database. """

        self.clean(Article.criterion_data)

        count = 0

        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].preprocess()

        self.clean(Article.criterion_entity)

    @Decorator("Processing the articles...")
    def process(self):
        """ Performs the processing of the database by calling the equivalent Article method. """

        count = 0

        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].process()

        self.clean(Article.criterion_similarity)

    @Decorator("Writing the candidates...")
    def write(self, file_name):
        """
        Writes the candidates of the database. Overwrites an existing file.

        Args:
            file_name: str, name of the file (without folder and extension).
        """

        file_name = 'results/' + file_name + ".txt"
        count = 0

        with open(file_name, "w+") as f:
            for id_ in self.articles:
                count = self.progression(count)
                self.articles[id_].write(f)

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

    # region Other methods

    @Decorator("Cleaning the articles...")
    def clean(self, criterion):
        """
        Removes from the database the articles which meets the criterion.

        Returns:
            function, criterion that an article must meet to be removed.
        """

        count, to_del = 0, []

        for id_ in self.articles:
            count = self.progression(count)

            if criterion(self.articles[id_]):
                to_del.append(id_)

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
    database = Database(max_size=100)

    database.preprocess()
    database.process()

    return


if __name__ == '__main__':
    main()
