from database_creation.utils import BaseClass
from database_creation.article import Article

from copy import copy
from numpy import random
from glob import glob
from collections import defaultdict


class Database(BaseClass):
    # region Class initialization

    to_print, print_attributes, print_lines, print_offsets = ['articles'], False, 2, 0
    limit_print, random_print = 50, True
    count_modulo = 1000

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
        self.most_frequent_entities = None

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

    # region Main methods

    @BaseClass.Verbose("Preprocessing the articles...")
    def preprocess_candidates(self):
        """ Performs the preprocessing of the database. """

        self.clean(Article.criterion_data)

        count = 0

        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].preprocess_candidates()

        self.clean(Article.criterion_entity)

    @BaseClass.Verbose("Processing the articles candidates...")
    def process_candidates(self):
        """ Performs the processing of the database by calling the equivalent Article method. """

        count = 0

        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].process_candidates()

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

    @BaseClass.Verbose("Preprocessing the articles...")
    def preprocess_tuples(self):
        """ Performs the preprocessing of the database. """

        self.clean(Article.criterion_data)

        count = 0

        for id_ in self.articles:
            count = self.progression(count)
            self.articles[id_].preprocess_tuples()

        self.clean(Article.criterion_entity)

    @BaseClass.Verbose("Processing the articles aggregation tuples...")
    def process_tuples(self):
        """ Performs the processing of the possible aggregation tuples of the database. """

        self.compute_most_frequent_entities(20)

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
    def compute_most_frequent_entities(self, limit):
        """
        Compute the most frequent tuple of entities.

        Args:
            limit: int, maximum number of tuples.
        """

        count, entities_dict = 0, defaultdict(int)

        for id_ in self.articles:
            count = self.progression(count)

            for entity_type in ['locations', 'persons', 'organizations']:
                entities = getattr(self.articles[id_], 'entities_' + entity_type)

                if entities and len(entities) >= 2:
                    entities.sort()

                    for t in self.tuples(entities):
                        entities_dict[t] += 1

        self.most_frequent_entities = \
            [(t, entities_dict[t]) for t in sorted(entities_dict, key=entities_dict.get, reverse=True)[0:limit]]

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

    @BaseClass.Verbose("Cleaning the database...")
    @BaseClass.Attribute('size')
    def clean(self, criterion):
        """
        Removes from the database the articles which meets the criterion.

        Returns:
            function, criterion that an article must meet to be removed.
        """

        to_del = []

        for id_ in self.articles:
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

    @staticmethod
    def tuples(l):
        """
        Compute all the possible tuples from l of len >= 2. Note that the element inside a tuple will appear in the same
        order as in l.

        Args:
            l: list, original list.

        Returns:
            set, all the possible tuples of len >= 2 of l.
        """

        if len(l) == 2 or len(l) > 10:
            return {tuple(l)}

        else:
            res = {tuple(l)}
            for x in l:
                res = res | Database.tuples([y for y in l if y != x])

            return res

    # endregion


# def main():
#
#     database = Database(max_size=1000)
#
#     database.preprocess_candidates()
#     database.process_candidates()
#
#     database.write_candidates('../results/out.txt')
#
#     return

def main():

    d = Database(max_size=10000)

    d.preprocess_tuples()
    d.process_tuples()

    print(d.most_frequent_entities)


if __name__ == '__main__':
    main()
