from time import time
from numpy import int64
from re import findall, sub
from textwrap import fill
from nltk import sent_tokenize
from unidecode import unidecode
from wikipedia import search, page, PageError, DisambiguationError


class BaseClass:
    # region Class base methods

    to_print, print_attribute, print_lines, print_offsets = None, None, None, None
    text_width = 100

    def __str__(self):
        """
        Overrides the builtin str method for the instances of BaseClass.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = to_print or list(self.__dict__.keys())

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
        elif isinstance(item, (BaseClass, Mention, Context, Entity, Wikipedia, Tuple, Query)):
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

    def __init__(self, sentences=None, type_=None):
        """
        Initializes the Context instance.

        Args:
            sentences: dict, the context's sentences (deep copy of the article's sentences), mapped by their indexes.
            type_: str, type_ of the context.
        """

        self.sentences = sentences
        self.type_ = type_

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Mention.

        Returns:
            str, readable format of the instance.
        """

        string = ''

        string += '[...] ' if self.type_ == 'content' else ''
        string += ' '.join([sentence.text for _, sentence in self.sentences.items()])
        string += ' [...]' if self.type_ == 'content' else ''

        return string

    # endregion


class Entity:
    # region Class base methods

    def __init__(self, original_name, type_):
        """
        Initializes the Entity instance.

        Args:
            original_name: str, original mention of the entity.
            type_: str, type of the entity.
        """

        self.original_name = original_name
        self.type_ = type_

        self.name = None
        self.plausible_names = None
        self.possible_names = None
        self.extra_info = None

        self.wiki = None

        self.compute_name()

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Entity.

        Returns:
            str, readable format of the instance.
        """

        return self.name

    # endregion

    # region Methods compute_

    def compute_name(self):
        """ Compute the name and possibly the plausible names and the extra info of the entity. """

        before_parenthesis = findall(r'(.*?)\s*\(', self.original_name)
        before_parenthesis = before_parenthesis[0] if before_parenthesis and before_parenthesis[0] \
            else self.original_name

        in_parenthesis = set(findall(r'\((.*?)\)', self.original_name))

        plausible_names, possible_names = set(), set()

        if self.type_ == 'person':
            before_parenthesis = before_parenthesis.replace('.', '')
            split = before_parenthesis.split()

            if len(split) > 1 and split[-1].lower().replace('.', '') in ['jr', 'junior']:
                name = ' '.join(split[:-1])
                suffix = ' Jr.'
            elif len(split) > 1 and split[-1].lower().replace('.', '') in ['sr', 'senior']:
                name = ' '.join(split[:-1])
                suffix = ' Sr.'
            else:
                name = before_parenthesis
                suffix = ''

            # Inverse the words when there is a comma
            split = name.split(', ')
            name = ' '.join([split[1], split[0]]) if len(split) == 2 else name

            # Add '.' to single letters
            split = name.split()
            for i in range(len(split)):
                if len(split[i]) == 1:
                    split[i] += '.'

            # Full name
            name = ' '.join(split)
            plausible_names.add(name)
            name += suffix
            plausible_names.add(name)

            # Name without single letters
            split_bis = [word for word in split if len(word.replace('.', '')) > 1]
            plausible_name = ' '.join(split_bis)
            plausible_names.add(plausible_name + suffix)
            plausible_names.add(plausible_name)

            # Combination of pairs of words
            if len(split_bis) > 2:
                for i in range(len(split_bis) - 1):
                    for j in range(i + 1, len(split_bis)):
                        possible_name = split_bis[i] + ' ' + split_bis[j]
                        possible_names.add(possible_name)

            # Last word of the name
            if len(split) > 0:
                possible_name = split[-1]
                possible_names.add(possible_name + suffix)
                possible_names.add(possible_name)

            # Fist and last part of the name
            if len(split) > 1:
                plausible_name = split[0] + ' ' + split[-1]
                plausible_names.add(plausible_name + suffix)
                plausible_names.add(plausible_name)

            # Write first letter for words in the middle
            if len(split) > 2:
                plausible_name = split[0] + ' ' + ' '.join([s[0] + '.' for s in split[1:-1]]) + ' ' + split[-1]
                plausible_names.add(plausible_name + suffix)
                plausible_names.add(plausible_name)

        elif self.type_ == 'location':
            name = before_parenthesis

            if in_parenthesis:
                info = ', '.join(in_parenthesis)
                plausible_names.add(name + ', ' + info)

            split = name.split()
            if len(split) > 1 and split[-1].lower() == 'city':
                plausible_names.add(' '.join(split[:-1]))

            if len(split) > 2:
                for i in range(len(split) - 1):
                    possible_name = split[i] + split[i + 1]
                    possible_names.add(possible_name)

        elif self.type_ == 'org':
            split = before_parenthesis.split()
            if split[-1].lower().replace('.', '') == 'co':
                name = ' '.join(split[:-1])
                suffix1, suffix2 = ' Co.', ' Company'
            elif split[-1].lower().replace('.', '') == 'company':
                name = ' '.join(split[:-1])
                suffix1, suffix2 = ' Company', ' Co.'
            elif split[-1].lower().replace('.', '') == 'foundation':
                name = ' '.join(split[:-1])
                suffix1, suffix2 = ' Foundation', ''
            else:
                name = ' '.join(split)
                suffix1, suffix2 = '', ''

            plausible_names.update([name, name + suffix1, name + suffix2])

            plausible_name = name.replace(' & ', ' and ')
            plausible_names.update([plausible_name, plausible_name + suffix1, plausible_name + suffix2])

            if len(split) > 2:
                for i in range(len(split) - 1):
                    possible_name = split[i] + split[i + 1]
                    possible_names.add(possible_name)

            name += suffix1

        else:
            raise Exception("Wrong type for an entity: {}".format(self.type_))

        if name in plausible_names:
            plausible_names.remove(name)
        for n in plausible_names.intersection(possible_names):
            possible_names.remove(n)

        extra_info = in_parenthesis

        self.name = name
        self.plausible_names = plausible_names
        self.possible_names = possible_names
        self.extra_info = extra_info

    # endregion

    # region Methods get_

    def get_wiki(self):
        """
        Returns the wikipedia information of the entity.

        Returns:
            Wikipedia, wikipedia page of the entity.
        """

        p, exact = self.match_page(self.name)

        if p is None:
            for query in search(self.name):
                p, exact = self.match_page(query)
                if p is not None:
                    break

        return Wikipedia(p, exact)

    # endregion

    # region Other methods

    def match(self, string, type_=None, flexible=False):
        """
        Check if the entity matches another entity represented as a string.

        Args:
            string: str, entity to check.
            type_: str, type of the entity; if None, takes the same as the other entity.
            flexible: bool, whether or not to check the possible names as well.

        Returns:
            bool, True iff the string matches the entity.
        """

        if type_ is not None:
            if self.type_ != type_:
                return False
        else:
            type_ = self.type_

        entity = Entity(string, type_)

        names1 = {self.name}.union(self.plausible_names)
        names2 = {entity.name}.union(entity.plausible_names)

        if flexible:
            names1.update(self.possible_names)
            names2.update(entity.possible_names)

        names1 = {unidecode(name) for name in names1}
        names2 = {unidecode(name) for name in names2}

        return self.name in names2 or entity.name in names1

    def is_in(self, string, flexible=False):
        """
        Check if the entity is in a text.

        Args:
            string: str, text to check.
            flexible: bool, whether or not to check the possible names as well.

        Returns:
            bool, True iff the text contains the entity.
        """

        string = unidecode(string)
        string = string.split()

        names = {self.name}.union(self.plausible_names)
        if flexible:
            names.update(self.possible_names)
        names = {unidecode(name) for name in names}

        for name in names:
            split = name.split()
            try:
                idxs = [string.index(word) for word in split]
            except ValueError:
                continue

            if len(split) == 1:
                return True

            else:
                differences = [idxs[i + 1] - idxs[i] for i in range(len(split) - 1)]
                if min(differences) > 0 and max(differences) <= 4:
                    return True

        return False

    def match_page(self, query):
        """
        Check if the entity matches the Wikipedia page found with a query and if the match is exact or if the page is
        only related to the entity.

        Args:
            query: str, query to perform to find the page.

        Returns:
            p: wikipedia.page, wikipedia page corresponding to the entity, or None.
            exact: bool, whether or not the match is exact or not.
        """

        try:
            p = page(query)
        except (PageError, DisambiguationError):
            return None, None

        if self.match(string=p.title, flexible=True):
            return p, True

        elif self.is_in(string=p.title) or self.is_in(string=p.summary):
            return p, False

        else:
            return None, None

    def update_info(self, entity):
        """
        Updates the information (plausible & possible names, extra info) of the current entity with another one.

        Args:
            entity: Entity, new entity to take into account.
        """

        e = self if len(self.name.split()) >= len(entity.name.split()) else entity

        self.name = e.name
        self.plausible_names = e.plausible_names
        self.possible_names = e.possible_names
        self.extra_info.update(entity.extra_info)

    # endregion


class Wikipedia:
    # region Class base methods

    info_length = 600

    def __init__(self, page=None, exact=None):
        """
        Initializes the Wikipedia instance; if the wikipedia entry is not found, the arguments are None.

        Args:
            page: wikipedia.page, wikipedia page of the entity; can be None.
            exact: bool, whether the page corresponds directly to an entity; can be None.
        """

        if page is not None:
            self.title = page.title
            self.summary = page.summary
            self.url = page.url

            self.exact = exact

            self.info = None
            self.compute_info()

        else:
            self.title, self.summary, self.url, self.exact, self.info = None, None, None, None, None

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Wikipedia.

        Returns:
            str, readable format of the instance.
        """

        if self.info is not None:
            return self.info

        else:
            return "No information found."

    # endregion

    # region Methods compute_

    def compute_info(self):
        """ Compute the information of the Wikipedia object. """

        paragraph = self.summary.split('\n')[0]

        if len(paragraph) <= self.info_length:
            info = paragraph

        else:
            sentences = sent_tokenize(paragraph)
            info = sentences[0]

            for sentence in sentences[1:]:
                new_info = info + ' ' + sentence
                if len(new_info) <= self.info_length:
                    info = new_info
                else:
                    break

        info = sub(r'\([^)]*\)', '', info).replace('  ', ' ')
        info = info.encode("utf-8", errors="ignore").decode()
        info = info + ' [This may be a related article.]' if not self.exact else info

        self.info = info

    # endregion


class Tuple:
    # region Class base methods

    def __init__(self, id_, entities, article_ids=None, query_ids=None):
        """
        Initializes the Tuple instance.

        Args:
            id_: str, id of the Tuple.
            entities: tuple, Entities of the Tuple.
            article_ids: set, ids of the articles where the entities are mentioned.
            query_ids: set, ids of the queries corresponding to the Tuple.
        """

        self.id_ = id_
        self.entities = entities
        self.article_ids = article_ids
        self.query_ids = query_ids

        self.type_ = self.get_type()

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Tuple.

        Returns:
            str, readable format of the instance.
        """

        return BaseClass.to_string(self.entities)

    # endregion

    # region Methods get_

    def get_type(self):
        """
        Returns the type of the Tuple.

        Returns:
            str, type of the tuple, must be 'location', 'person' or 'org'.
        """

        types = set([entity.type_ for entity in self.entities])
        assert len(types) == 1 and types.issubset({'location', 'person', 'org'})

        return types.pop()

    def get_name(self):
        """
        Returns the tuple of names of the Tuple.

        Returns:
            tuple, tuple of the names (str) of the Tuple.
        """

        name = tuple([entity.name for entity in self.entities])

        return name

    # endregion


class Query:
    # region Class base methods

    def __init__(self, id_, tuple_, article, context):
        """
        Initializes the Query instance.

        Args:
            id_: str, id of the Query.
            tuple_: Tuple, Tuple of entities mentioned in the article.
            article: Article, article from where the query comes from.
            context: Context, context of the entities in the article.
        """

        self.id_ = id_

        self.entities = tuple_.entities
        self.string_entities = self.get_string_entities(tuple_)
        self.info = self.get_info(tuple_)

        self.title = article.title
        self.date = article.date

        self.context = str(context)
        self.type_ = context.type_

        self.html_entities = self.get_html_entities()
        self.html_info = self.get_html_info()
        self.html_title = self.get_html_title()
        self.html_context = self.get_html_context()

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Query.

        Returns:
            str, readable format of the instance.
        """

        width = BaseClass.text_width

        string = fill("Entities: " + self.string_entities, width) + '\n\n'
        string += "Info:" + '\n\n'.join([fill(info, width) for info in self.info]) + '\n\n'
        string += fill("Article: " + self.title + ' (' + self.date + ')', width) + '\n\n'
        string += fill("Context: " + BaseClass.to_string(self.context), width) + '\n\n'

        return string

    # endregion

    # region Methods get_

    @staticmethod
    def get_string_entities(tuple_):
        """
        Returns the entities in a natural language string.

        Args:
            tuple_: Tuple, Tuple of entities mentioned in the article.

        Returns:
            str, list of the entities separated by ',' or 'and'.
        """

        names = tuple_.get_name()

        return ', '.join(names[:-1]) + ' and ' + names[-1]

    @staticmethod
    def get_info(tuple_):
        """
        Returns the wikipedia information as a list of information for each Entity.

        Args:
            tuple_: Tuple, Tuple of entities mentioned in the article.

        Returns:
            list, information for each Entity as a string.
        """

        info = []
        for entity in tuple_.entities:
            if entity.wiki is None:
                info.append("Entity not searched.")
            else:
                info.append(str(entity.wiki))

        return info

    def get_html_entities(self):
        """
        Returns the html version of the entities.

        Returns:
            str, html version of the entities.
        """

        string = ''.join(['<th>' + str(entity) + '</th>' for entity in self.entities])

        return string

    def get_html_info(self):
        """
        Returns the html version of the information.

        Returns:
            str, html version of the information.
        """

        string = ''.join(['<td>' + str(info) + '</td>' for info in self.info])

        return string

    def get_html_context(self):
        """
        Returns the html version of the Context.

        Returns:
            str, html version of the Context.
        """

        string = '<td colspan=' + str(len(self.entities)) + '>' + str(self.context) + '</td>'

        return string

    def get_html_title(self):
        """
        Returns the html version of the title.

        Returns:
            str, html version of the title.
        """

        string = '<th colspan=' + str(len(self.entities)) + '><strong_blue>'
        string += 'Title of the article: '
        string += '</strong_blue>' + str(self.title)
        string += ' (' + str(self.date) + ')'
        string += '</th>'

        return string

    # endregion

    # region Other methods

    def to_dict(self):
        """
        Return the object as a dictionary.

        Returns:
            dict, object as a dictionary.
        """

        d = {
            'id_': self.id_,
            'entities': self.html_entities,
            'string_entities': self.string_entities,
            'info': self.html_info,
            'title': self.html_title,
            'context': self.html_context,
        }

        return d

    # endregion


def main():
    for name in ['George Bush', 'George W Bush', 'George Walker Bush', 'George Bush Sr', 'George W Bush Sr',
                 'George Walker Bush Sr', 'Valerie Elise Plame Wilson', 'Sacco and Vanzetti']:
        e = Entity(name, 'person')
        print(e.name, e.plausible_names, e.possible_names)

    for pair in [('George Bush', 'George Walker Bush'),
                 ('George Walker Bush Sr', 'George Bush jr'),
                 ('George Walker Bush', 'George H W Bush'),
                 ('Valerie Plame', 'Valerie Elise Plame Wilson')]:
        print(Entity(pair[0], 'person').match(pair[1], 'person', False),
              Entity(pair[0], 'person').match(pair[1], 'person', True))

    return


if __name__ == '__main__':
    main()
