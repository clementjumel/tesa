from collections import defaultdict
from copy import deepcopy
from string import punctuation as string_punctuation
from xml.etree import ElementTree

from nltk.corpus import stopwords as nltk_stopwords

from tesa.database_creation.utils import Context, Entity, Mention


class Article:
    def __init__(self, data_path, content_path, summary_path):
        """
        Initializes an instance of Article.

        Args:
            data_path: str, path of the article's data.
            content_path: str, path of the article's annotated content.
            summary_path: str, path of the article's annotated summary.
        """

        self.data_path = data_path
        self.content_path = content_path
        self.summary_path = summary_path

        self.title = None
        self.date = None
        self.entities = None
        self.content = None
        self.summary = None
        self.contexts = None

    # region Methods compute_

    def compute_metadata(self):
        """Compute the metadata (title, date) of the article."""

        root = ElementTree.parse(self.data_path).getroot()

        title = self.get_title(root)
        date = self.get_date(root)

        self.title = title
        self.date = date

    def compute_corpus_annotations(self):
        """Compute the annotated texts (content or summary) of the article from the corpus."""

        content_root = ElementTree.parse(self.content_path).getroot()
        summary_root = ElementTree.parse(self.summary_path).getroot()

        self.content = Text(root=content_root, entities=self.entities)
        self.summary = Text(root=summary_root, entities=self.entities)

    def compute_contexts(self, tuple_):
        """
        Compute the contexts of the article for the Tuple of entities, according to the specified context types.

        Args:
            tuple_: Tuple, tuple of Entities mentioned in the article.
        """

        name = str(tuple_)
        self.contexts = self.contexts or dict()

        contexts = dict()

        contexts.update(self.content.contexts_neigh_sent(tuple_=tuple_, type_="content"))
        contexts.update(self.summary.contexts_all_sent(tuple_=tuple_, type_="summary"))

        self.contexts[name] = contexts

    # endregion

    # region Methods get_

    @staticmethod
    def get_title(root):
        """
        Returns the title of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            str, title of the article.
        """

        element = root.find("./head/title")

        return element.text if element is not None else "No title."

    @staticmethod
    def get_date(root):
        """
        Returns the date of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            str, date of the article.
        """

        d = root.find('./head/meta[@name="publication_day_of_month"]').get("content")
        m = root.find('./head/meta[@name="publication_month"]').get("content")
        y = root.find('./head/meta[@name="publication_year"]').get("content")

        d = "0" + d if len(d) == 1 else d
        m = "0" + m if len(m) == 1 else m

        return "/".join([y, m, d])

    def get_entities(self):
        """
        Returns the Entities of the article given the tree of its metadata.

        Returns:
            list, Entities of the article.
        """

        root = ElementTree.parse(self.data_path).getroot()

        person_elements = root.findall("./head/docdata/identified-content/person")
        location_elements = root.findall("./head/docdata/identified-content/location")
        org_elements = root.findall("./head/docdata/identified-content/org")

        elements = set(
            [("person", e.text) for e in person_elements if e.get("class") == "indexing_service"]
            + [
                ("location", e.text)
                for e in location_elements
                if e.get("class") == "indexing_service"
            ]
            + [("org", e.text) for e in org_elements if e.get("class") == "indexing_service"]
        )

        entities = [
            Entity(original_name=element[1], type_=element[0]) for element in sorted(elements)
        ]
        assert len(entities) == len(set([str(entity) for entity in entities]))

        return entities

    def get_vanilla_entities(self):
        """
        Returns the unpreprocessed Entities of the article given the tree of its metadata.

        Returns:
            list, name of the Entities of the article.
        """

        root = ElementTree.parse(self.data_path).getroot()

        person_elements = root.findall("./head/docdata/identified-content/person")
        location_elements = root.findall("./head/docdata/identified-content/location")
        org_elements = root.findall("./head/docdata/identified-content/org")

        entities = (
            [("person", e.text) for e in person_elements if e.get("class") == "indexing_service"]
            + [
                ("location", e.text)
                for e in location_elements
                if e.get("class") == "indexing_service"
            ]
            + [("org", e.text) for e in org_elements if e.get("class") == "indexing_service"]
        )

        entities = [pair[1] + " (" + pair[0] + ")" for pair in sorted(entities)]

        return entities

    # endregion

    def criterion_content(self):
        """
        Check if an article's content file exists.

        Returns:
            bool, True iff the file doesn't exist and must be deleted.
        """

        try:
            f = open(self.content_path, "r")
            f.close()
            return False

        except FileNotFoundError:
            return True

    # region Methods debug_

    def debug_articles(self):
        """
        Returns a string showing, for each article, the data path retrieved.

        Returns:
            str, debugging of the article.
        """

        return " -> " + self.data_path

    def debug_metadata(self):
        """
        Returns a string showing, for each article, the metadata retrieved.

        Returns:
            str, debugging of the article.
        """

        return " -> " + self.title + " (" + self.date + ")"

    def debug_article_entities(self):
        """
        Returns a string showing, for each article, the entities retrieved.

        Returns:
            str, debugging of the article.
        """

        entities1 = self.get_vanilla_entities()
        entities2 = [str(entity) for entity in self.entities]

        if len(entities1) != len(entities2):
            return ": " + ", ".join(entities1) + "\n      -> " + ", ".join(entities2)
        else:
            return ""

    def debug_annotations(self):
        """
        Returns a string showing, for each article, the coreference chains.

        Returns:
            str, debugging of the article.
        """

        s, empty = ": " + ", ".join([str(entity) for entity in self.entities]) + "\n", True

        for coreference in self.content.coreferences + self.summary.coreferences:
            mentions = [coreference.representative] + coreference.mentions
            matches = sorted(
                set(
                    [
                        str(entity)
                        for entity in self.entities
                        for mention in mentions
                        if entity.is_in(str(mention), flexible=True)
                    ]
                )
            )

            if matches:
                s += str(coreference) + " (" + ", ".join(matches) + ")" + "\n"
                empty = False

        return s if not empty else ""

    def debug_contexts(self):
        """
        Returns a string showing, for each article, the potential contexts retrieved.

        Returns:
            str, debugging of the article.
        """

        s, empty = ":", True

        for tuple_name, contexts in self.contexts.items():
            temp = "\n".join([str(context) for _, context in contexts.items()])

            if temp:
                s += "\n" + tuple_name + ":\n" + temp + "\n"
                empty = False

        return s if not empty else ""

    # endregion


class Text:
    def __init__(self, root, entities):
        """
        Initializes an instance of Text (content or summary).

        Args:
            root: ElementTree.root, root of the annotations of the Text.
            entities: set, Entities of the article.
        """

        self.sentences = self.get_sentences(root)
        self.coreferences = self.get_coreferences(root, entities)

    # region Methods get_

    @staticmethod
    def get_sentences(root):
        """
        Returns the Sentences of a Text given its tree.

        Args:
            root: ElementTree.root, root of the annotations of the Text.

        Returns:
            dict, Sentences of the article (mapped with their indexes).
        """

        elements = root.findall("./document/sentences/sentence")
        sentences = {int(element.attrib["id"]): Sentence(element) for element in elements}

        return sentences

    @staticmethod
    def get_coreferences(root, entities):
        """
        Returns the Coreferences of a Text given its tree.

        Args:
            root: ElementTree.root, root of the annotations of the Text.
            entities: set, Entities of the article.

        Returns:
            list, Coreferences of the article.
        """

        elements = root.findall("./document/coreference/coreference")
        coreferences = [Coreference(element, entities) for element in elements]

        return coreferences

    def get_entity_sentences(self, entity):
        """
        Returns the indexes of the sentences where there is a mention of the specified entity.

        Args:
            entity: Entity, entity we want to find mentions of.

        Returns:
            list, sorted list of sentences' indexes.
        """

        entity_sentences = set()

        for coreference in self.coreferences:
            if coreference.entity and str(entity) == coreference.entity:
                entity_sentences.update(coreference.sentences)

        return sorted(entity_sentences)

    # endregion

    # region Methods contexts_

    def contexts_neigh_sent(self, tuple_, type_):
        """
        Returns the neighboring-sentences Contexts for a Tuple (neighboring sentences where the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: str, type of the Text used for the Context.

        Returns:
            dict, Contexts mapped with the indexes of the first and last sentences separated by '_'.
        """

        contexts, context_length = {}, len(tuple_.entities)

        # Mapping between the name of the entities and the indexes of the mentions' sentences
        entities_sentences = dict(
            [(str(entity), self.get_entity_sentences(entity)) for entity in tuple_.entities]
        )
        all_entities_sentences = set(
            [idx for _, sentences in entities_sentences.items() for idx in sentences]
        )

        for start_idx in all_entities_sentences:
            seen = set()

            for idx in range(start_idx, start_idx + context_length):
                seen.update(
                    [name for name, sentences in entities_sentences.items() if idx in sentences]
                )

                if len(seen) == context_length:
                    if idx == start_idx:  # Case of one-sentence context
                        if start_idx == 1:  # Case of beginning of article
                            idx += 1
                        else:  # Case of middle of the article
                            start_idx -= 1

                    sentence_span = range(start_idx, idx + 1)
                    sentences = {i: deepcopy(self.sentences[i]) for i in sentence_span}
                    self.highlight(sentences=sentences, tuple_=tuple_)

                    id_ = str(start_idx) + "_" + str(idx)
                    contexts[id_] = Context(sentences=sentences, type_=type_)
                    break

        return contexts

    def contexts_all_sent(self, tuple_, type_):
        """
        Returns the all-sentences Contexts for a Tuple (all the sentences if all the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: type of the Text used for the Context.

        Returns:
            dict, Contexts mapped with the index '0'.
        """

        contexts, context_length = {}, len(tuple_.entities)

        for entity in tuple_.entities:
            if not self.get_entity_sentences(entity):
                return contexts

        sentences = deepcopy(self.sentences)
        self.highlight(sentences=sentences, tuple_=tuple_)

        contexts["0"] = Context(sentences=sentences, type_=type_)

        return contexts

    # endregion

    def highlight(self, sentences, tuple_):
        """
        Highlight (put in bold) the mentions of the entities of the Tuple in the sentences.

        Args:
            sentences: dict, sentences to highlight.
            tuple_: Tuple, entities to analyse.
        """

        entities_boundaries = defaultdict(set)

        idx = 0
        entity_to_color = dict()

        if len(tuple_.entities) > 8:
            raise Exception("Not enough colors, implement more of them!")

        for i in range(len(tuple_.entities)):
            entity = tuple_.entities[i]
            entity_to_color[entity.name] = "color" + str(i)

        for coreference in self.coreferences:
            if coreference.entity and coreference.entity in [
                str(entity) for entity in tuple_.entities
            ]:
                entity = [
                    entity for entity in tuple_.entities if entity.name == coreference.entity
                ][0]
                for mention in [coreference.representative] + coreference.mentions:
                    if mention.sentence in sentences:
                        entities_boundaries[mention.sentence].add(
                            (entity.name, mention.start, mention.end)
                        )

        for sentence_id, sentence in sentences.items():
            boundaries = sorted(entities_boundaries[sentence_id])
            to_remove = set()

            for boundary1 in boundaries:
                for boundary2 in boundaries:
                    if boundary1[0] == boundary2[0] and (
                        (boundary1[1] > boundary2[1] and boundary1[2] <= boundary2[2])
                        or (boundary1[1] >= boundary2[1] and boundary1[2] < boundary2[2])
                    ):
                        to_remove.add(boundary2)

            for boundary in to_remove:
                boundaries.remove(boundary)

            for name, start_idx, end_idx in boundaries:
                color = entity_to_color[name]
                start_tag = '<div class="popup" onclick="pop(' + str(idx) + ')"><' + color + ">"
                end_tag = (
                    "</"
                    + color
                    + '><span class="popuptext" id="'
                    + str(idx)
                    + '">'
                    + name
                    + "</span></div>"
                )

                if sentence.tokens[start_idx].start_tag is None:
                    sentence.tokens[start_idx].start_tag = start_tag
                else:
                    sentence.tokens[start_idx].start_tag += start_tag

                if sentence.tokens[end_idx - 1].end_tag is None:
                    sentence.tokens[end_idx - 1].end_tag = end_tag
                else:
                    sentence.tokens[end_idx - 1].end_tag += end_tag

                idx += 1


class Sentence:
    def __init__(self, sentence_element):
        """
        Initializes an instance of Sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.tokens = None

        self.compute_tokens(sentence_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Sentence.

        Returns:
            str, readable format of the instance.
        """

        s, start = "", True

        for _, token in self.tokens.items():
            if start:  # Beginning of the sentence
                if token.criterion_punctuation():
                    s += str(token)
                else:
                    s += str(token)[0].capitalize() + str(token)[1:]
                    start = False

            else:
                if not token.criterion_punctuation():
                    s += " "

                s += str(token)

        return s

    def compute_tokens(self, sentence_element):
        """
        Compute the tokens of the sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        tokens = {}

        for token_element in sentence_element.findall("./tokens/token"):
            tokens[int(token_element.attrib["id"])] = Token(token_element)

        self.tokens = tokens


class Coreference:
    def __init__(self, element, entities):
        """
        Initializes an instance of the Coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
            entities: set, Entities of the articles.
        """

        self.representative = None
        self.mentions = None
        self.entity = None
        self.sentences = None

        self.compute_annotations(element)
        self.compute_entity(entities)
        self.compute_sentences()

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Coreference.

        Returns:
            str, readable format of the instance.
        """

        entity = "[" + self.entity + "] " if self.entity is not None else ""

        return entity + "; ".join(
            [str(mention) for mention in [self.representative] + self.mentions]
        )

    # region Methods compute_

    def compute_annotations(self, element):
        """
        Compute the mentions and the representative of the coreference, and the corresponding sentences.

        Args:
            element: ElementTree.Element, annotations of the coreference.
        """

        element_mentions = element.findall("./mention")

        m = element_mentions[0]
        assert m.attrib and m.attrib["representative"] == "true"
        representative = Mention(
            text=m.find("text").text,
            sentence=int(m.find("sentence").text),
            start=int(m.find("start").text),
            end=int(m.find("end").text),
        )

        mentions = []
        for i in range(1, len(element_mentions)):
            m = element_mentions[i]

            text = m.find("text").text
            if len(text.split()) <= 10:
                mentions.append(
                    Mention(
                        text=text,
                        sentence=int(m.find("sentence").text),
                        start=int(m.find("start").text),
                        end=int(m.find("end").text),
                    )
                )

        self.representative = representative
        self.mentions = mentions

    def compute_entity(self, entities):
        """
        Compute the entity the coreference refer to, or None.

        Args:
            entities: set, Entities of the articles.
        """

        matches = set(
            [
                str(entity)
                for entity in entities
                if entity.match_string(string=self.representative.text, flexible=False)
            ]
        )

        if len(matches) == 1:
            self.entity = matches.pop()

    def compute_sentences(self):
        """Compute the indexes of the sentences of the coreference chain."""

        self.sentences = set([m.sentence for m in [self.representative] + self.mentions])

    # endregion


class Token:
    punctuation = [p for p in string_punctuation] + ["''", "``"]
    stopwords = set(nltk_stopwords.words("english"))

    def __init__(self, token_element):
        """
        Initializes an instance of Token.

        Args:
            token_element: ElementTree.Element, annotations of the token.
        """

        self.word = None
        self.lemma = None
        self.pos = None
        self.ner = None
        self.start_tag = None
        self.end_tag = None

        self.compute_annotations(token_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Token.

        Returns:
            str, readable format of the instance.
        """

        s = self.start_tag if self.start_tag is not None else ""
        s += self.word
        s += self.end_tag if self.end_tag is not None else ""

        return s

    def compute_annotations(self, token_element):
        """
        Compute the annotations (word, lemma, Part Of Speech tag and Named-Entity Recognition) of the token.

        Args:
            token_element: ElementTree.Element, annotations of the token.
        """

        self.word = token_element.find("word").text
        self.lemma = token_element.find("lemma").text
        self.pos = token_element.find("POS").text
        self.ner = token_element.find("NER").text

    # region Methods criterion_

    def criterion_punctuation(self):
        """Check if a token is a punctuation mark."""

        return True if self.word in self.punctuation and self.word != "--" else False

    def criterion_stopwords(self):
        """Check if a token is a stop word."""

        return True if self.lemma in self.stopwords else False

    # endregion
