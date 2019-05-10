from database_creation.utils import BaseClass
from database_creation.articles import Article

from copy import copy
from numpy import random
from glob import glob


class Database(BaseClass):
    # region Class initialization

    to_print = ['articles']
    print_attribute = False
    print_lines = 2
    print_offsets = 0

    limit_print = 50
    random_print = False

    verbose = True
    display_count = 10000

    def __init__(self, year, limit_articles=None):
        """
        Initializes an instance of Database, then initializes, cleans and preprocess the articles.

        Args:
            year: int, year of the database to take into account.
            limit_articles: int, maximum number of articles; if None, takes all articles.
        """

        self.year = str(year)
        self.limit_articles = limit_articles

        self.db_root = 'databases/nyt_jingyun'

        self.articles = None
        self.n_articles = None

        self.initialize()
        self.clean()
        self.preprocess()

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Database.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets, limit_print, random_print = \
            self.get_print_parameters()[:6]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'articles' else ''
            string += self.prefix(print_attribute, print_lines if string else 0, print_offsets, attribute) + s if s \
                else ''

        article_ids = self.get_articles_ids()
        if random_print:
            random.shuffle(article_ids)

        string += self.prefix(True, print_lines if string else 0, print_offsets, 'articles') if print_attribute else ''

        count = 0
        for article_id in article_ids:
            s = self.to_string(self.articles[article_id])

            if s:
                string += self.prefix(print_attribute, print_lines if string else 0, print_offsets,
                                      'id ' + str(article_id)) + s
                count += 1
                if count == limit_print:
                    break

        return string

    @classmethod
    def set_print_parameters(cls, to_print=None, print_attribute=None, print_lines=None, print_offsets=None,
                             limit_print=None, random_print=None):
        """
        Changes the print attributes of the class.

        Args:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            limit_print: int, limit number of articles to print; if -1, prints all.
            random_print: bool, whether or not to select randomly the articles printed.
        """

        super(Database, cls).set_print_parameters(to_print=to_print, print_attribute=print_attribute,
                                                  print_lines=print_lines, print_offsets=print_offsets)

        cls.limit_print = limit_print if limit_print is not None else cls.limit_print
        cls.random_print = random_print if random_print is not None else cls.random_print

    @classmethod
    def get_print_parameters(cls):
        """
        Computes the print attribute of the class.

        Returns:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            cls.limit_print: int, limit number of articles to print; if -1, prints all.
            cls.random_print: bool, whether or not to select randomly the articles printed.
        """

        to_print, print_attribute, print_lines, print_offsets = super(Database, cls).get_print_parameters()

        return to_print, print_attribute, print_lines, print_offsets, cls.limit_print, cls.random_print

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
                    print(self.message)

                func(*args, **kwargs)

                if slf.verbose:
                    print("Done ({} articles).\n".format(slf.n_articles))

            return f

    # endregion

    # region Main methods

    @Decorator("Initializing the articles...")
    def initialize(self):
        """ Initializes the articles of the database. """

        def get_original_paths(pattern, limit):
            """ Computes the sorted file paths corresponding to the pattern. """

            file_paths = glob(pattern)
            file_paths.sort()

            return file_paths[:limit] if limit else file_paths

        articles = {}

        for original_path in get_original_paths(pattern=self.db_root + '/data/' + self.year + '/*/*/*.xml',
                                                limit=self.limit_articles):
            article_id = original_path.split('/')[-1].split('.')[0]
            annotated_path = \
                self.db_root + '/content_annotated/' + self.year + 'content_annotated/' + article_id + '.txt.xml'

            articles[article_id] = Article(article_id=article_id,
                                           original_path=original_path,
                                           annotated_path=annotated_path)

        self.articles = articles
        self.n_articles = self.get_n_articles()

    @Decorator("Cleaning the articles...")
    def clean(self):
        """ Removes from the database the articles with incomplete data. """

        count_articles, to_del = 0, []

        for article_id in self.articles:
            count_articles = self.progression_update(count_articles)

            if self.articles[article_id].to_clean():
                to_del.append(article_id)

        for article_id in to_del:
            del self.articles[article_id]

        self.n_articles = self.get_n_articles()

    @Decorator("Preprocessing the articles...")
    def preprocess(self):
        """ Performs the preprocessing of the database by calling the equivalent Article method. """

        count_articles = 0

        for article_id in self.articles:
            count_articles = self.progression_update(count_articles)
            self.articles[article_id].preprocess()

    @Decorator("Processing the articles...")
    def process(self):
        """ Performs the processing of the database by calling the equivalent Article method. """

        count_articles = 0

        for article_id in self.articles:
            count_articles = self.progression_update(count_articles)
            self.articles[article_id].process()

    @Decorator("Writing the candidates...")
    def write_candidates(self, file_name):
        """
        Writes the candidates of the database. Overwrites an existing file.

        Args:
            file_name: str, name of the file (without folder and extension).
        """

        file_name = 'results/' + file_name + ".txt"
        count_articles = 0

        with open(file_name, "w+") as f:
            for article_id in self.articles:
                count_articles = self.progression_update(count_articles)
                self.articles[article_id].write_candidates(f)

    # endregion

    # region Methods get

    def get_articles_ids(self):
        """
        Computes the ids of the articles as a list.

        Returns:
            list, ids of the articles.
        """

        return list(self.articles.keys())

    def get_n_articles(self):
        """
        Computes the updated number of articles in the database.

        Returns:
            int, number of articles.
        """

        return len(self.get_articles_ids())

    # endregion

    # region Other methods

    def progression_update(self, count_articles):
        """
        Prints the progression update and update the articles' count.

        Args:
            count_articles: int, current count of articles.

        Returns:
            int, incremented count of articles.
        """

        count_articles += 1

        if self.verbose and count_articles % self.display_count == 0:
            print("File {}/{}...".format(count_articles, self.n_articles))

        return count_articles

    # endregion


def main():
    db = Database(year='2000', limit_articles=100)
    db.process()
    db.write_candidates('out')
    return


if __name__ == '__main__':
    main()
