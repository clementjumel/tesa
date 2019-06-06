from copy import copy
from time import time
from numpy import int64
from re import findall
from gensim.models import KeyedVectors


class BaseClass:
    # region Class base methods

    verbose = True
    # TODO: add a print_lines before
    to_print, print_attribute, print_lines, print_offsets = None, None, None, None

    def __str__(self):
        """
        Overrides the builtin str method for the instances of BaseClass.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute))
            string += self.prefix(print_attribute, print_lines, print_offsets, attribute) + s if s else ''

        return string

    @classmethod
    def get_parameters(cls):
        """
        Fetch the print attribute of the class.

        Returns:
            cls.to_print: list, attributes to print; if [], print all the attributes.
            cls.print_attribute: bool, whether or not to print the attributes' names.
            cls.print_lines: int, whether or not to print line breaks (and how many).
            cls.print_offsets: int, whether or not to print an offset (and how many).
        """

        return cls.to_print, cls.print_attribute, cls.print_lines, cls.print_offsets

    @classmethod
    def set_parameters(cls, to_print=None, print_attribute=None, print_lines=None, print_offsets=None):
        """
        Changes the print attributes of the class.

        Args:
            to_print: list, attributes to print; if [], print all the attributes.
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
        """

        cls.to_print = to_print if to_print is not None else cls.to_print
        cls.print_attribute = print_attribute if print_attribute is not None else cls.print_attribute
        cls.print_lines = print_lines if print_lines is not None else cls.print_lines
        cls.print_offsets = print_offsets if print_offsets is not None else cls.print_offsets

    @classmethod
    def set_verbose(cls, verbose):
        """
        Changes the verbose attribute of the class.

        Args:
            verbose: bool, new verbose value.
        """

        cls.verbose = verbose

    @staticmethod
    def to_string(item):
        """
        Converts an item of any type into a string with a easily readable format.

        Args:
            item: unknown type, item to convert into string, can be of any type.

        Returns:
            str, readable format of item.
        """

        # Cases corresponding to an empty string
        if item is None or item == [] or item == () or item == {}:
            return ''

        # Case of strings
        elif isinstance(item, str):
            return item

        # Case of numbers
        elif isinstance(item, (float, int, int64)):
            return str(round(item, 2))

        # Case of instances of custom objects
        elif isinstance(item, (BaseClass, Similarity, Dependency, Mention, Context)):
            return str(item)

        # Case of lists
        elif isinstance(item, list):
            strings = [BaseClass.to_string(ite) for ite in item]

            # List of custom objects
            if isinstance(item[0], BaseClass):
                return ' '.join([s for s in strings if s])

            # List of lists
            elif isinstance(item[0], list):
                return '\n'.join([s for s in strings if s])

            # Other lists
            else:
                return '|'.join([s for s in strings if s])

        # Case of sets and tuples
        elif isinstance(item, (set, tuple)):
            strings = [BaseClass.to_string(ite) for ite in item]
            return '|'.join([s for s in strings if s])

        # Case of dictionaries
        elif isinstance(item, dict):
            strings = [(BaseClass.to_string(ite), BaseClass.to_string(item[ite])) for ite in item]

            # The keys are int (long dictionaries)
            if isinstance(list(item.keys())[0], int):
                return ' '.join([s[1] for s in strings if s[1]])

            # The keys are strings or tuples (short dictionaries)
            elif isinstance(list(item.keys())[0], (str, tuple)):
                # Dictionary of dictionaries
                if isinstance(item[list(item.keys())[0]], dict):
                    return '\n'.join([s[0] + ':\n' + s[1] for s in strings if s[1]])

                # Other dictionaries
                else:
                    return '\n'.join([s[0] + ': ' + s[1] for s in strings if s[1]])

        else:
            raise Exception("Unsupported type: {}.".format(type(item)))

    @staticmethod
    def prefix(print_attribute=False, print_lines=0, print_offsets=0, attribute=None):
        """
        Returns a prefix corresponding to the parameters.

        Args:
            print_attribute: bool, whether or not to print the attributes' names.
            print_lines: int, whether or not to print line breaks (and how many).
            print_offsets: int, whether or not to print an offset (and how many).
            attribute: str, attribute to print (if relevant).

        Returns:
            str, prefix corresponding to the parameters.
        """

        prefix = ''

        if print_lines:
            for _ in range(print_lines):
                prefix += '\n'

        if print_offsets:
            for _ in range(print_offsets):
                prefix += '  '

        if print_attribute:
            if attribute is not None:
                prefix += attribute + ': '
            else:
                raise Exception("No attribute specified.")

        return prefix

    # endregion

    # region Decorator

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
                t0 = time()

                if slf.verbose:
                    print('\n' + self.message)

                func(*args, **kwargs)

                if slf.verbose:
                    print("Done (elapsed time: {}s).\n".format(round(time() - t0)))

            return f

    class Attribute:
        """ Decorator for monitoring an attribute. """

        def __init__(self, attribute, length=False):
            """ Initializes the Attribute decorator attribute. """

            self.attribute = attribute
            self.length = length

        def __call__(self, func):
            """ Performs the call to the decorated function. """

            def f(*args, **kwargs):
                """ Decorated function. """

                slf = args[0]

                if slf.verbose:
                    if not self.length:
                        print("Initial {}: {}".format(self.attribute, getattr(slf, self.attribute)))
                    else:
                        print("Initial length of {}: {}".format(self.attribute, len(getattr(slf, self.attribute))))

                func(*args, **kwargs)

                if slf.verbose:
                    if not self.length:
                        print("Final {}: {}".format(self.attribute, getattr(slf, self.attribute)))
                    else:
                        print("Final length of {}: {}".format(self.attribute, len(getattr(slf, self.attribute))))

            return f

    # endregion

    # region Methods standardize

    @staticmethod
    def standardize_location(entity):
        """
        Standardize a location entity (forget what is inside parenthesis).

        Args:
            entity: str, location entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        return entity

    @staticmethod
    def standardize_person(entity):
        """
        Standardize a person entity (forget what is inside parenthesis, inverse last name and first name when necessary, 
        remove middle name/letter).

        Args:
            entity: str, person entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        split = entity.split(', ')
        entity = ' '.join([split[1], split[0]]) if len(split) == 2 else entity  # inverse last name, first names

        split = entity.split()
        entity = ' '.join([split[0], split[2]]) if len(split) == 3 else entity  # remove middle name

        return entity

    @staticmethod
    def standardize_organization(entity):
        """
        Standardize an organization entity (forget what is inside parenthesis).

        Args:
            entity: str, entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis

        entity = before[0] if len(before) > 0 else entity

        return entity

    @staticmethod
    def standardize(entity):
        """
        Standardize an entity by returning all possible different standardizations.

        Args:
            entity: str, entity to standardize.

        Returns:
            set, strings representing the different standardizations of the entity.
        """

        standardization = {entity}

        for standardize_name in ['location', 'person', 'organization']:
            standardize = getattr(BaseClass, 'standardize_' + standardize_name)
            s = standardize(entity)

            standardization.add(s)

            if standardize_name == 'person' and len(s.split()) == 2:
                standardization.add(s.split()[1])  # add the case of only last name

        return standardization

    # endregion

    # region Other methods

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
                res = res.union(BaseClass.subtuples([y for y in l if y != x]))

            return res

    @staticmethod
    def match(entity1, entity2):
        """
        Check if the two entities match by checking the intersection of their standardization.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.

        Returns:
            bool, True iff the two entities match.
        """

        return True if BaseClass.standardize(entity1).intersection(BaseClass.standardize(entity2)) else False

    @classmethod
    @Verbose("Loading embeddings...")
    def load_embeddings(cls, type_):
        """
        Load the token or entity embeddings.

        Args:
            type_: str, must be 'token' or 'entity', type of the embeddings.
        """

        if type_ == 'token':
            cls.embeddings_token = KeyedVectors.load_word2vec_format(
                fname='../pre_trained_models/GoogleNews-vectors-negative300.bin',
                binary=True
            )

        elif type_ == 'entity':
            cls.embeddings_entity = KeyedVectors.load_word2vec_format(
                fname='../pre_trained_models/freebase-vectors-skipgram1000-en.bin',
                binary=True
            )

        else:
            raise Exception("Wrong embeddings' type: {}".format(type_))

    # endregion


class Similarity:
    # region Class base methods

    def __init__(self, score, items, similar_items):
        """
        Initializes the Similarity instance.

        Args:
            score: float, similarity score between the object and similar_to.
            items: list, items compared.
            similar_items: list, objects similar to the initial objects.
        """

        self.score = score
        self.items = items
        self.similar_items = similar_items

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Similarity.

        Returns:
            str, readable format of the instance.
        """

        string = BaseClass.to_string(self.score)
        string += ' (' + BaseClass.to_string(self.items) + '/' + BaseClass.to_string(self.similar_items) + ')'

        return string

    # endregion


class Dependency:
    # region Class base methods

    def __init__(self, type_, gov_word, gov_idx, dep_word, dep_idx):
        """
        Initializes the Dependency instance.

        Args:
            type_: str, type of the dependency.
            gov_word: str, word of the governor Token.
            gov_idx: int, index of the governor Token.
            dep_word: str, word of the dependent Token.
            dep_idx: int, index of the dependent Token.
        """

        self.type_ = type_
        self.gov_word = gov_word
        self.gov_idx = gov_idx
        self.dep_word = dep_word
        self.dep_idx = dep_idx

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Dependency.

        Returns:
            str, readable format of the instance.
        """

        string = BaseClass.to_string(self.type_) + ': '
        string += BaseClass.to_string(self.gov_word) + '[' + BaseClass.to_string(self.gov_idx) + ']/'
        string += BaseClass.to_string(self.dep_word) + '[' + BaseClass.to_string(self.dep_idx) + ']'

        return string

    # endregion


class Mention:
    # region Class base methods

    def __init__(self, text, sentence, start, end):
        """
        Initializes the Mention instance.

        Args:
            text: str, text of the mention.
            sentence: int, index of the sentence of the mention.
            start: int, index of the beginning of the mention in the sentence.
            end: int, index of the end of the mention in the sentence.
        """

        self.text = text
        self.sentence = sentence
        self.start = start
        self.end = end

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Mention.

        Returns:
            str, readable format of the instance.
        """

        string = BaseClass.to_string(self.text)
        string += ' (sentence [' + BaseClass.to_string(self.sentence) + '], '
        string += 'tokens [' + BaseClass.to_string(self.start) + '-' + BaseClass.to_string(self.end) + '])'

        return string

    # endregion


class Context:
    # region Class base methods

    def __init__(self, sentence_texts, sentence_idxs,
                 before_texts=None, before_idxs=None,
                 after_texts=None, after_idxs=None):
        """
        Initializes the Context instance.

        Args:
            sentence_texts: list, the context's sentences' texts.
            sentence_idxs: list, indexes of sentences' sentences.
            before_texts: list, sentences' texts before the actual context.
            before_idxs: list, indexes of before's sentences.
            after_texts: list, sentences' texts after the actual context.
            after_idxs: list, indexes of after's sentences.
        """

        self.sentence_texts = sentence_texts
        self.sentence_idxs = sentence_idxs
        self.before_texts = before_texts
        self.before_idxs = before_idxs
        self.after_texts = after_texts
        self.after_idxs = after_idxs

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Mention.

        Returns:
            str, readable format of the instance.
        """

        string = ''

        if self.before_texts is not None:
            string += ' '.join([
                BaseClass.to_string(self.before_texts[i]) + '[' + BaseClass.to_string(self.before_idxs[i]) + ']'
                if self.before_texts[i] else '[...]' for i in range(len(self.before_texts))
            ])
            string += '\n'

        string += ' '.join([BaseClass.to_string(self.sentence_texts[i]) + '[' +
                            BaseClass.to_string(self.sentence_idxs[i]) + ']' for i in range(len(self.sentence_texts))])

        if self.after_texts is not None:
            string += '\n'
            string += ' '.join([
                BaseClass.to_string(self.after_texts[i]) + '[' + BaseClass.to_string(self.after_idxs[i]) + ']'
                if self.after_texts[i] else '[...]' for i in range(len(self.after_texts))
            ])

        return string

    # endregion


def main():
    return


if __name__ == '__main__':
    main()
