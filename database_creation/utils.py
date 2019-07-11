from copy import copy
from time import time
from numpy import int64
from re import findall
from gensim.models import KeyedVectors
from textwrap import fill


class BaseClass:
    # region Class base methods

    verbose = True
    to_print, print_attribute, print_lines, print_offsets = None, None, None, None
    text_width = 100

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

    def print(self, string):
        """
        Alternative to the builtin method to take into account self.verbose.

        Args:
            string: str, message to print.
        """

        if self.verbose:
            print(string)

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
        elif isinstance(item, (BaseClass, Similarity, Dependency, Mention, Context, Query)):
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
            return '; '.join([s for s in strings if s])

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
            print(item)
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
                t0 = time()

                if slf.verbose:
                    print('\n' + self.message)

                func(*args, **kwargs)

                if slf.verbose:
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

                if slf.verbose:
                    attribute = getattr(slf, self.attribute)
                    if attribute is not None:
                        print("Initial length of {}: {}".format(self.attribute, len(attribute)))

                func(*args, **kwargs)

                if slf.verbose:
                    attribute = getattr(slf, self.attribute)
                    if attribute is not None:
                        print("Final length of {}: {}".format(self.attribute, len(attribute)))

            return f

    # endregion

    # region Methods standardize_

    @staticmethod
    def standardize_location(entity):
        """
        Standardize a location entity.

        Args:
            entity: str, location entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis
        entity = before[0] if len(before) > 0 else entity

        entity = entity.replace(' & ', ' and ')

        return entity

    @staticmethod
    def standardize_person(entity):
        """
        Standardize a person entity.

        Args:
            entity: str, person entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis
        entity = before[0] if len(before) > 0 else entity

        split = entity.split()
        if split[-1].lower() in ['jr', 'jr.']:
            entity = ' '.join(split[:-1])
            jr = ' Jr'
        elif split[-1].lower() in ['sr', 'sr.']:
            entity = ' '.join(split[:-1])
            jr = ' Sr'
        else:
            jr = ''

        split = entity.split(', ')
        entity = ' '.join([split[1], split[0]]) if len(split) == 2 else entity  # inverse last name, first names

        words = [s for s in entity.split() if len(s) > 1]
        entity = ' '.join(words) if len(words) >= 2 else entity  # remove single letters

        split = entity.split()
        entity = ' '.join([split[0], split[2]]) if len(split) == 3 else entity  # remove middle name

        entity += jr

        return entity

    @staticmethod
    def standardize_organization(entity):
        """
        Standardize an organization entity.

        Args:
            entity: str, entity to standardize.

        Returns:
            str, standardized entity.
        """

        before = findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis
        entity = before[0] if len(before) > 0 else entity

        entity = entity.replace(' & ', ' and ')

        return entity

    def standardize(self, entity, type_):
        """
        Standardize an entity of type type_.

        Args:
            entity: str, entity to standardize.
            type_: str, type of the entity, must be 'location', 'person' or organization'.

        Returns:
            str, standardized entity.
        """

        return getattr(self, 'standardize_' + type_)(entity)

    # endregion

    # region Methods match_

    def match_location(self, entity1, entity2, flexible):
        """
        Check if two location entities match.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.
            flexible: bool, if True, loosen the conditions for matching.

        Returns:
            bool, True iff the two strings refer to the same location entity.
        """

        standardized1, standardized2 = self.standardize_location(entity1), self.standardize_location(entity2)

        candidates1 = {entity1}
        candidates1.add(standardized1 or entity1)

        candidates2 = {entity2}
        candidates2.add(standardized2 or entity2)

        if flexible:
            pass

        return True if candidates1.intersection(candidates2) else False

    def match_person(self, entity1, entity2, flexible):
        """
        Check if two person entities match.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.
            flexible: bool, if True, loosen the conditions for matching.

        Returns:
            bool, True iff the two strings refer to the same person entity.
        """

        standardized1, standardized2 = self.standardize_person(entity1), self.standardize_person(entity2)

        candidates1 = {entity1}
        candidates1.add(standardized1 or entity1)

        candidates2 = {entity2}
        candidates2.add(standardized2 or entity2)

        if flexible:
            candidates1.add(standardized1.split()[-1])
            candidates2.add(standardized2.split()[-1])

        return True if candidates1.intersection(candidates2) else False

    def match_organization(self, entity1, entity2, flexible):
        """
        Check if two organization entities match.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.
            flexible: bool, if True, loosen the conditions for matching.

        Returns:
            bool, True iff the two strings refer to the same organization entity.
        """

        standardized1, standardized2 = self.standardize_organization(entity1), self.standardize_organization(entity2)

        candidates1 = {entity1}
        candidates1.add(standardized1 or entity1)
        candidates1.add(' '.join(standardized1.split()[:-1])
                        if standardized1.split()[-1].lower() in ['co', 'co.', 'company', 'foundation'] else entity1)

        candidates2 = {entity2}
        candidates2.add(standardized2 or entity2)
        candidates2.add(' '.join(standardized2.split()[:-1])
                        if standardized2.split()[-1].lower() in ['co', 'co.', 'company', 'foundation'] else entity2)

        if flexible:
            pass

        return True if candidates1.intersection(candidates2) else False

    def match(self, entity1, entity2, type_=None, flexible=True):
        """
        Check if two entities match.

        Args:
            entity1: str, first entity to compare.
            entity2: str, second entity to compare.
            type_: str, type of the entity, must be 'location', 'person' or organization'.
            flexible: bool, if True, loosen the conditions for matching.

        Returns:
            bool, True iff the two strings refer to the same entity of type type_, if mentioned.
        """

        if type_ is not None:
            return getattr(self, 'match_' + type_)(entity1, entity2, flexible)

        else:
            return any([getattr(self, 'match_' + type_)(entity1, entity2, flexible)
                        for type_ in ['location', 'person', 'organization']])

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


class Tuple:
    # region Class base methods

    def __init__(self, id_, entities, type_, article_ids, query_ids=None):
        """
        Initializes an entity Tuple instance.

        Args:
            id_: str, id of the Tuple.
            entities: tuple, entities of the Tuple.
            type_: str, type of the entities, must be 'location', 'person' or 'organization'.
            article_ids: set, ids of the articles where the entities are mentioned.
            query_ids: set, ids of the queries corresponding to the Tuple.
        """

        self.id_ = id_
        self.entities = entities
        self.type_ = type_
        self.article_ids = article_ids
        self.query_ids = query_ids

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Tuple.

        Returns:
            str, readable format of the instance.
        """

        return BaseClass.to_string(self.entities)

    # endregion


class Context:
    # region Class base methods

    def __init__(self, sentences, entity_coreferences):
        """
        Initializes the Context instance.

        Args:
            sentences: dict, the context's sentences (deep copy of the article's sentences), mapped by their indexes.
            entity_coreferences: dict, coreferences mapped by the indexes and then the entities.
        """

        self.sentences = sentences
        self.enhance_entities(entity_coreferences)

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Mention.

        Returns:
            str, readable format of the instance.
        """

        string = '[...] ' if list(self.sentences.keys())[0] != 1 else ''
        string += ' '.join([self.sentences[id_].text for id_ in self.sentences])
        string += ' [...]'

        return string

    def enhance_entities(self, entity_coreferences):
        """
        Enhance the entities mentioned in the context.

        Args:
            entity_coreferences: dict, coreferences mapped by the indexes and then the entities.
        """

        for idx in entity_coreferences:
            for entity in entity_coreferences[idx]:
                for coreference in entity_coreferences[idx][entity]:
                    for mention in [coreference.representative] + coreference.mentions:

                        if mention.sentence == idx:
                            if not BaseClass.match(self.sentences[idx], entity, mention.text):
                                self.sentences[idx].tokens[mention.end - 1].word += ' [' + entity + ']'

                            self.sentences[idx].tokens[mention.start].word = \
                                '<strong>' + self.sentences[idx].tokens[mention.start].word
                            self.sentences[idx].tokens[mention.end - 1].word += '</strong>'

            self.sentences[idx].compute_text()

    # endregion


class Query:
    # region Class base methods

    def __init__(self, id_, entities, title, date, abstract, info, context):
        """
        Initializes the aggregation Query instance.

        Args:
            id_, str, id of the query.
            entities: tuple, entities mentioned in the article.
            title: str, title of the article from where the query comes from.
            date: str, date of the article from where the query comes from.
            abstract: str, abstract of the article from where the query comes from.
            info: dict, wikipedia information of the entities.
            context: Context, context of the entities in the article.
        """

        self.id_ = id_

        self.entities = entities
        self.info = info

        self.title = title
        self.date = date
        self.abstract = abstract

        self.context = context

        self.html_entities = self.get_html_entities()
        self.html_info = self.get_html_info()
        self.html_context = self.get_html_context()

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Query.

        Returns:
            str, readable format of the instance.
        """

        string = fill("Entities: " + BaseClass.to_string(self.entities), BaseClass.text_width) + '\n\n'

        string += '\n\n'.join([fill(self.info[entity], BaseClass.text_width) for entity in self.info
                               if self.info[entity]]) + '\n\n'

        string += fill("Article: " + self.title + ' (' + self.date + ')', BaseClass.text_width) + '\n'
        string += fill(self.abstract, BaseClass.text_width) + '\n\n'

        string += fill(BaseClass.to_string(self.context), BaseClass.text_width) + '\n\n'

        return string

    def to_dict(self):
        """
        Return the object as a dictionary.

        Returns:
            dict, object as a dictionary.
        """

        d = {
            'id': self.id_,
            'html_entities': self.html_entities,
            'html_info': self.html_info,
            'title': self.title,
            'date': self.date,
            'abstract': self.abstract,
            'html_context': self.html_context,
        }

        return d

    def get_html_entities(self):
        """
        Returns the html version of the entities.

        Returns:
            str, html version of the entities.
        """

        return ', '.join([entity for entity in self.entities[:-1]]) + ' & ' + self.entities[-1]

    def get_html_info(self):
        """
        Returns the html version of the information.

        Returns:
            str, html version of the information.
        """

        string = '<br/>'.join([
            '<a href=' + self.info[entity]['url'] + '>' + entity + '</a>: ' + self.info[entity]['paragraph']
            if self.info[entity] else entity + ': No information found.' for entity in self.info
        ])

        return string

    def get_html_context(self):
        """
        Returns the html version of the Context.

        Returns:
            str, html version of the Context.
        """

        return str(self.context)

    # endregion


def main():
    return


if __name__ == '__main__':
    main()
