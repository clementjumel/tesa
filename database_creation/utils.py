from re import findall, sub
from numpy.random import shuffle, choice
from nltk import sent_tokenize
from unidecode import unidecode
from wikipedia import search, page, PageError, DisambiguationError
from collections import defaultdict


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

        return self.text

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

        s = '[...] ' if self.type_ == 'content' else ''
        s += ' '.join([str(sentence) for _, sentence in self.sentences.items()])
        s += ' [...]' if self.type_ == 'content' else ''

        return s

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

    def __eq__(self, obj):
        """
        Overrides the builtin equals method for the instances of Entity.

        Returns:
            bool, whether or not the two objects are equal.
        """

        if not isinstance(obj, Entity):
            return False

        if self.name != obj.name:
            return False

        if self.type_ != obj.type_:
            return False

        return True

    # endregion

    # region Methods compute_

    def compute_name(self):
        """ Compute the names and the extra info of the entity. """

        before_parenthesis = findall(r'(.*?)\s*\(', self.original_name)
        before_parenthesis = before_parenthesis[0] if before_parenthesis and before_parenthesis[0] \
            else self.original_name
        in_parenthesis = set(findall(r'\((.*?)\)', self.original_name))

        plausible_names, possible_names = set(), set()

        if self.type_ == 'person':
            before_parenthesis = before_parenthesis.replace('.', '')
            split = before_parenthesis.split()

            if len(split) > 1 and split[-1].lower() in ['jr', 'sr']:
                name = ' '.join(split[:-1])
                suffix = ' ' + split[-1].lower().capitalize() + '.'
            else:
                name, suffix = ' '.join(split), ''

            # Inverse the words when there is a comma
            split = name.split(', ')
            name = ' '.join([split[1], split[0]]) if len(split) == 2 else name

            # Add '.' to single letters
            split = name.split()
            for i in range(len(split)):
                if len(split[i]) == 1:
                    split[i] += '.'
            name = ' '.join(split)

            # Name without suffix
            plausible_names.add(name) if suffix else None

            # Name without single letters
            split = [word for word in name.split() if len(word.replace('.', '')) > 1]
            plausible_names.add(' '.join(split))
            plausible_names.add(' '.join(split) + suffix) if suffix else None

            # Combination of pairs of words
            split = [word for word in name.split() if len(word.replace('.', '')) > 1]
            if len(split) > 2:
                for i in range(len(split) - 1):
                    for j in range(i + 1, len(split)):
                        plausible_names.add(split[i] + ' ' + split[j])
                        plausible_names.add(split[i] + ' ' + split[j] + suffix) if suffix else None

            # Last word of the name
            split = name.split()
            if len(split) > 1:
                possible_names.add(split[-1])
                possible_names.add(split[-1] + suffix) if suffix else None

            # Fist and last part of the name
            split = name.split()
            if len(split) > 2:
                plausible_names.add(split[0] + ' ' + split[-1])
                plausible_names.add(split[0] + ' ' + split[-1] + suffix) if suffix else None

            # Write first letter for words in the middle
            split = name.split()
            if len(split) > 2:
                plausible_names.add(split[0] + ' ' + ' '.join([s[0] + '.' for s in split[1:-1]]) + ' ' + split[-1])
                plausible_names.add(split[0] + ' ' + ' '.join([s[0] + '.' for s in split[1:-1]]) + ' ' + split[-1]
                                    + suffix) if suffix else None

            name += suffix

        elif self.type_ == 'location':
            split = before_parenthesis.split()

            if len(split) > 1 and split[-1].lower() in ['city']:
                name = ' '.join(split[:-1])
                suffix = ' ' + split[-1].lower().capitalize()
            else:
                name, suffix = ' '.join(split), ''

            plausible_names.add(name)
            name += suffix

            if in_parenthesis:
                info = ', '.join(in_parenthesis)
                plausible_names.add(name + ', ' + info)

        elif self.type_ == 'org':
            split = before_parenthesis.split()

            if len(split) > 1 and split[-1].lower().replace('.', '') in ['co', 'corp', 'inc']:
                name = ' '.join(split[:-1])
                suffix = ' ' + split[-1].lower().replace('.', '').capitalize() + '.'
            elif len(split) > 1 and split[-1].lower() in ['company', 'university']:
                name = ' '.join(split[:-1])
                suffix = ' ' + split[-1].lower().replace('.', '').capitalize()
            else:
                name, suffix = ' '.join(split), ''

            plausible_names.add(name)
            name += suffix

        else:
            raise Exception("Wrong type for an entity: {}".format(self.type_))

        for n in plausible_names.intersection(possible_names):
            possible_names.remove(n)
        for n in {name}.intersection(plausible_names):
            plausible_names.remove(n)

        self.name = name
        self.plausible_names = plausible_names
        self.possible_names = possible_names
        self.extra_info = in_parenthesis

    # endregion

    # region Methods get_

    def get_wiki(self):
        """
        Returns the wikipedia information of the entity.

        Returns:
            Wikipedia, wikipedia page of the entity.
        """

        p = self.match_page(self.name)

        if p is None:
            for query in search(self.name):
                p = self.match_page(query)
                if p is not None:
                    break

        return Wikipedia(p)

    # endregion

    # region Methods debug_

    def debug_entities(self):
        """
        Returns a string showing the debugging of an entity.

        Returns:
            str, debugging of the entity.
        """

        s = ' (' + self.original_name + '): '
        s += '; '.join(self.plausible_names) + ' | ' + '; '.join(self.possible_names) + ' | '
        s += '; '.join(self.extra_info)

        return s

    # endregion

    # region Other methods

    def match_entity(self, entity, flexible=False):
        """
        Check if the entity matches another entity.

        Args:
            entity: Entity, entity to check.
            flexible: bool, whether or not to check the possible names as well.

        Returns:
            bool, True iff the entities match.
        """

        if self.__eq__(entity):
            return True

        elif not isinstance(entity, Entity) or not self.type_ == entity.type_:
            return False

        else:
            names1 = {self.name}.union(self.plausible_names)
            names2 = {entity.name}.union(entity.plausible_names)

            if flexible:
                names1.update(self.possible_names)
                names2.update(entity.possible_names)

            names1 = {unidecode(name) for name in names1}
            names2 = {unidecode(name) for name in names2}

            return self.name in names2 or entity.name in names1

    def match_string(self, string, flexible=False):
        """
        Check if the entity matches a string representing potentially an entity.

        Args:
            string: str, string to check.
            flexible: bool, whether or not to check the possible names as well.

        Returns:
            bool, True iff the entity matches the string.
        """

        if not self.not_match(string):
            words, strings = string.split(), [string]

            if len(words) > 1 and words[-1] in ["'", "'s"]:
                s = ' '.join(words[:-1])
                strings.append(s)
                words = s.split()

            if len(words) > 1:
                s = ' '.join(words[1:])
                strings.append(s)

            for s in strings:
                entity = Entity(original_name=s, type_=self.type_)
                if self.match_entity(entity=entity, flexible=flexible):
                    return True

        return False

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
        Check if the entity matches the Wikipedia page found with a query.

        Args:
            query: str, query to perform to find the page.

        Returns:
            wikipedia.page, wikipedia page corresponding to the entity, or None.
        """

        try:
            p = page(query)
        except (PageError, DisambiguationError):
            return None

        if self.match_string(string=p.title, flexible=False) \
                or self.is_in(string=p.title, flexible=False) \
                or self.is_in(string=p.summary, flexible=False):
            return p

        else:
            return None

    def update_info(self, entity):
        """
        Updates the information (plausible & possible names, extra info) of the current entity with another one.

        Args:
            entity: Entity, new entity to take into account.
        """

        assert self.__eq__(entity)

        if self.original_name != entity.original_name:
            names1 = set([self.name] + list(self.plausible_names) + list(self.possible_names))
            names2 = set([entity.name] + list(entity.plausible_names) + list(entity.possible_names))

            if len(names1) != len(names2):
                e = self if len(names1) > len(names2) else entity
                self.plausible_names = e.plausible_names
                self.possible_names = e.possible_names

            self.extra_info.update(entity.extra_info)

    def not_match(self, string):
        """
        Check if a string must not be compared to the entity.

        Args:
            string: str, string to check.

        Returns:
            bool, True iff the string must not be compared to the entity.
        """

        words = string.lower().split()

        if self.type_ == 'person':
            if 'and' in words:
                return True

        elif self.type_ == 'location':
            pass

        elif self.type_ == 'org':
            pass

        else:
            raise Exception("Wrong type of entity: {}".format(self.type_))

        return False

    # endregion


class Wikipedia:
    # region Class base methods

    info_length = 600

    def __init__(self, page=None):
        """
        Initializes the Wikipedia instance; if the wikipedia entry is not found, the arguments are None.

        Args:
            page: wikipedia.page, wikipedia page of the entity; can be None.
        """

        if page is not None:
            self.title = page.title
            self.summary = page.summary
            self.url = page.url

        else:
            self.title = None
            self.summary = None
            self.url = None

        self.exact = False

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Wikipedia.

        Returns:
            str, readable format of the instance.
        """

        return "No information found." if self.summary is None else self.get_info()

    # endregion

    # region Methods get_

    def get_info(self):
        """
        Returns the information of the Wikipedia object.

        Returns:
            str, info of the Wikipedia object.
        """

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

        return info

    # endregion

    # region Methods debug_

    def debug_wikipedia(self):
        """
        Returns a string showing the debugging of a wikipedia information.

        Returns:
            str, debugging of the wikipedia information.
        """

        return ' (' + self.title + '): ' + str(self)[:150] + '...'

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

        return ', '.join([str(entity) for entity in self.entities[:-1]]) + ' and ' + str(self.entities[-1])

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

    # endregion

    # region Methods debug_

    def debug_tuples(self):
        """
        Returns a string showing the debugging of a tuple.

        Returns:
            str, debugging of the tuple.
        """

        return ' (' + self.type_ + '): in ' + str(len(self.article_ids)) + ' articles'

    # endregion


class Query:
    """ Query to gather annotations from the Mechanical Turk workers. """

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

        self.entities = [str(entity) for entity in tuple_.entities]
        self.entities_type_ = tuple_.type_
        self.summaries = dict([(str(entity), str(entity.wiki)) for entity in tuple_.entities])
        self.urls = dict([(str(entity), entity.wiki.url) for entity in tuple_.entities])

        self.title = article.title
        self.date = article.date

        self.context = str(context)
        self.context_type_ = context.type_

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Query.

        Returns:
            str, readable format of the instance.
        """

        d = self.to_readable()

        return d['entities'] + ':\n' + d['context'] + '\n'

    def __eq__(self, obj):
        """
        Overrides the builtin equals method for the instances of Query.

        Returns:
            bool, whether or not the two objects are equal.
        """

        return isinstance(obj, Query) and self.to_html() == obj.to_html() and self.to_readable() == obj.to_readable()

    # endregion

    # region Methods get_

    def get_entities_phrase(self):
        """
        Returns a generic phrase such as "the two persons".

        Returns:
            str, generic phrase of the tuple.
        """

        type_ = self.entities_type_ + 's'
        if type_ == 'persons':
            type_ = 'people'
        elif type_ == 'locations':
            type_ = 'areas'
        elif type_ == 'orgs':
            type_ = 'organizations'
        else:
            raise Exception

        int_to_str = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
        number = int_to_str[len(self.entities)]

        return 'The ' + number + ' ' + type_

    def get_entities_html(self):
        """
        Returns the html version of the entities.

        Returns:
            str, html version of the entities.
        """

        s = ''

        for entity_name in self.entities:
            url = self.urls[entity_name]

            s += '<th>'
            if url is not None:
                s += '<a href="' + url + '" target="_blank">'

            s += entity_name

            if url is not None:
                s += '</a>'
            s += '</th>'

        return s

    def get_title_html(self):
        """
        Returns the html version of the title.

        Returns:
            str, html version of the title.
        """

        s = '<th colspan=' + str(len(self.entities)) + '><strong_blue>Title of the article:</strong_blue> '
        s += self.title + '</th>'

        return s

    def get_context_html(self):
        """
        Returns the html version of the context.

        Returns:
            str, html version of the context.
        """

        context = self.context.replace('-LRB- ', '(').replace(' -RRB-', ')')
        context = context.replace('-LRB-', '(').replace('-RRB-', ')')

        s = '<td colspan=' + str(len(self.entities)) + '>'
        s += context
        s += '</td>'

        return s

    def get_context_readable(self):
        """
        Returns the readable version of the context.

        Returns:
            str, readable version of the context.
        """

        s = self.context.replace('-LRB- ', '(').replace(' -RRB-', ')')
        s = s.replace('-LRB-', '(').replace('-RRB-', ')')

        s = sub(r'<span.*?>', ' [', s)
        s = sub(r'</span.*?>', ']', s)
        s = sub(r'<.*?>', '', s)

        return s

    # endregion

    # region Methods to_

    def to_html(self):
        """
        Return the html elements of the object as a dictionary.

        Returns:
            dict, object as a dictionary.
        """

        d = {
            'id_': self.id_,
            'entities': self.get_entities_html(),
            'generic_phrase': self.get_entities_phrase(),
            'context': self.get_context_html(),
            'entities_names': ', '.join(self.entities[:-1]) + ' and ' + self.entities[-1],
            'info': ''.join(['<td>' + self.summaries[entity_name] + '</td>' for entity_name in self.entities]),
            'title': self.get_title_html(),
        }

        return d

    def to_readable(self):
        """
        Return the readable elements of the object as a dictionary.

        Returns:
            dict, object as a dictionary.
        """

        d = {
            'entities': ', '.join(self.entities[:-1]) + ' and ' + self.entities[-1],
            'info': '\n'.join([self.summaries[entity_name] for entity_name in self.entities]),
            'context': self.get_context_readable(),
        }

        return d

    # endregion

    # region Methods debug_

    def debug_queries(self):
        """
        Returns a string showing the debugging of an query.

        Returns:
            str, debugging of the query.
        """

        return str(self)

    # endregion


class Annotation:
    """ Annotation from the Mechanical Turk workers. """

    # region Class base methods

    def __init__(self, id_, version, batch, row):
        """
        Initializes the Annotation instance.

        Args:
            id_: str, id of the corresponding Query.
            version: str, version of the Query.
            batch: str, batch of the Annotation.
            row: panda.Series, whole data of the annotation.
        """

        self.id_ = id_
        self.version = version
        self.batch = batch

        self.worker_id = str(row.get('WorkerId'))
        self.duration = int(row.get('WorkTimeInSeconds'))
        self.bug = bool(row.get('Answer.box_impossible.on'))

        self.answers, self.preprocessed_answers = self.get_answers(row)
        self.correct_inconsistencies()

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Annotation.

        Returns:
            str, readable format of the instance.
        """

        return '/'.join(self.answers) or '[bug]'

    # endregion

    # region Methods get_

    @staticmethod
    def get_answers(row):
        """
        Return both the cleaned answers (remove artifacts) and the preprocessed answers and return them as lists.

        Args:
            row: panda.Series, whole data of the annotation.

        Returns:
            answers: list, cleaned answers.
            preprocessed_answers: list, preprocessed answers.
        """

        unpreprocessed_answers, answers, preprocessed_answers = [], [], []
        articles, numbers = ['the'], ['two', 'three', 'four', 'five', 'six']
        standard_answers = ['people', 'areas', 'organizations']

        unpreprocessed_answers.append(str(row.get('Answer.utterance_1'))) \
            if str(row.get('Answer.utterance_1')) != 'nan' else None
        unpreprocessed_answers.append(str(row.get('Answer.utterance_2'))) \
            if str(row.get('Answer.utterance_2')) != 'nan' else None

        for unpreprocessed_answer in unpreprocessed_answers:
            old = unpreprocessed_answer

            if '.' in unpreprocessed_answer and ' are discussed' in unpreprocessed_answer.lower().split('.')[0]:
                unpreprocessed_answer = '.'.join(unpreprocessed_answer.split('.')[1:])

            if unpreprocessed_answer.isupper():
                unpreprocessed_answer = unpreprocessed_answer.capitalize()

            if old != unpreprocessed_answer:
                print('   Correcting "{}" to "{}"'.format(old, unpreprocessed_answer))

            while unpreprocessed_answer[-1] in ['.', ' ']:
                unpreprocessed_answer = unpreprocessed_answer[:-1]

            while unpreprocessed_answer[0] == ' ':
                unpreprocessed_answer = unpreprocessed_answer[1:]

            answer = ' '.join([unpreprocessed_answer.split()[0].capitalize()] + unpreprocessed_answer.split()[1:])
            words = answer.lower().split()

            words = words[1:] if words[0] in articles and len(words) > 1 else words
            words = words[1:] if words[0] in numbers and len(words) > 1 else words

            preprocessed_answer = ' '.join(words)

            if 'have' in preprocessed_answer.split() \
                    or 'are' in preprocessed_answer.split() \
                    or preprocessed_answer == 'the' \
                    or '/' in preprocessed_answer:
                print('   Discarding "{}"'.format(answer))
            elif preprocessed_answer not in preprocessed_answers and preprocessed_answer not in standard_answers:
                answers.append(answer)
                preprocessed_answers.append(preprocessed_answer)

        return answers, preprocessed_answers

    # endregion

    # region Other methods

    def correct_inconsistencies(self):
        """ Correct the NA reports inconsistencies by discarding answers when there is a reported bug. """

        if not self.answers and not self.bug:
            self.bug = True

        elif self.answers and self.bug:
            self.answers, self.preprocessed_answers = [], []

    # endregion


class Sample:
    """ Sample for the modeling Task. """

    # region Class base methods

    def __init__(self, query, positive_answers, negative_answers):
        """
        Initializes the Sample instance.

        Args:
            query: Query, initial Query of the annotation.
            positive_answers: list of str, gold standard answers.
            negative_answers: list of str, negative answers.
        """

        self.entities = query.entities
        self.entities_type_ = query.entities_type_
        self.summaries = query.summaries
        self.title = query.title
        self.context = query.context

        self.positive_answers = positive_answers
        self.negative_answers = negative_answers

        choices = positive_answers + negative_answers
        shuffle(choices)
        self.choices = choices

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Sample.

        Returns:
            str, readable format of the instance.
        """

        choices = []

        for choice in self.choices:
            if choice in self.positive_answers:
                choices.append(choice + '[y]')
            else:
                choices.append(choice)

        return '/'.join(self.entities) + ': ' + '; '.join(choices) + '?'

    # endregion

    # region Methods get_

    def get_x(self):
        """ Returns the x data of the Sample. """

        x = {'entities': self.entities,
             'summaries': self.summaries,
             'article': self.title + ': ' + self.context,
             'choices': self.choices}

        return x

    def get_y(self):
        """ Returns the y data of the Sample. """

        return [1 if choice in self.positive_answers else 0 for choice in self.choices]

    # endregion


class Task:
    """ Modeling task. """

    # region Class base methods

    def __init__(self, queries, annotations, answers_threshold):
        """
        Initializes the Task instance.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.
            answers_threshold: int, number of annotators that answers an annotation for it to be taken into account.
        """

        annotations = self.filter_annotations(annotations=annotations, answers_threshold=answers_threshold)

        self.answers = self.get_answers(queries, annotations)
        self.samples = self.get_samples(queries, annotations)

    def __str__(self):
        """
        Overrides the builtin str method for the instances of Task.

        Returns:
            str, readable format of the instance.
        """

        return '\n'.join([str(sample) for sample in self.samples])

    # endregion

    # region Methods get_

    @staticmethod
    def get_answers(queries, annotations):
        """
        Returns the answers and their probabilities in the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            list, tuples (probability, answer, entities type) corresponding to the answers.
        """

        answers = defaultdict(int)

        for id_, annotation_list in annotations.items():
            for annotation in annotation_list:
                for answer in annotation.preprocessed_answers:
                    answers[(answer, queries[id_].entities_type_)] += 1

        answers = sorted([(count, key[0], key[1]) for key, count in answers.items()], reverse=True)

        return answers

    def get_samples(self, queries, annotations):
        """
        Returns the samples of the Task.

        Args:
            queries: dict of Query, Queries of the annotations.
            annotations: dict of list of Annotations, Annotations from the MT workers.

        Returns:
            list, not randomized list of Samples.
        """

        samples = []

        for id_, annotation_list in annotations.items():
            query = queries[id_]
            positive_answers = sorted(set([answer for annotation in annotation_list
                                           for answer in annotation.preprocessed_answers]))

            answers = [(count, answer) for count, answer, entities_type_ in self.answers
                       if entities_type_ == query.entities_type_ and answer not in positive_answers]
            total_count = sum([count for count, _ in answers])

            negative_answers = list(choice(a=[answer for _, answer in answers],
                                           size=len(positive_answers),
                                           replace=False,
                                           p=[count/total_count for count, _ in answers]))

            samples.append(Sample(query=query, positive_answers=positive_answers, negative_answers=negative_answers))

        shuffle(samples)

        return samples

    def get_data(self):
        """
        Returns the modeling Task, to feed to a model.

        Returns:
            list, Task and its Samples, ready to be analyzed.
        """

        return [(sample.get_x(), sample.get_y()) for sample in self.samples]

    # endregion

    # region Other methods

    @staticmethod
    def filter_annotations(annotations, answers_threshold):
        """
        Returns the Annotations filtered from those which received less than answers_threshold answers.

        Args:
            annotations: dict of list of Annotations, Annotations from the MT workers.
            answers_threshold: int, number of annotators that answers an annotation for it to be taken into account.

        Returns:
            dict of list of Annotations, filtered Annotations from the MT workers.
        """

        return dict([(id_, annotation_list) for id_, annotation_list in annotations.items()
                     if len([annotation for annotation in annotation_list if not annotation.bug]) >= answers_threshold])

    # endregion


def main():
    return


if __name__ == '__main__':
    main()
