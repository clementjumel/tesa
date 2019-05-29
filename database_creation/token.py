from database_creation.utils import BaseClass

from copy import copy
from string import punctuation as string_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import KeyedVectors


class Token(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], False, 0, 0

    punctuation = [p for p in string_punctuation] + ["''", '``']
    stopwords = set(nltk_stopwords.words('english'))
    embedding_token = None

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

        self.similarity = None
        self.similar_entities = None

        self.compute_annotations(token_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Token.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string1 = self.prefix(print_lines=print_lines, print_offsets=print_offsets) + str(self.word)
        string2 = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'word' else ''
            string2 += ', ' if s and string2 else ''
            string2 += self.prefix(print_attribute=print_attribute, attribute=attribute) + s if s else ''

        return string1 + ' (' + string2 + ')' if string2 else string1

    # endregion

    # region Methods compute_

    def compute_annotations(self, token_element):
        """
        Compute the annotations (word, lemma, Part Of Speech tag and Named-Entity Recognition) of the token.

        Args:
            token_element: ElementTree.Element, annotations of the token.
        """

        self.word = token_element.find('word').text

        if self.word not in self.punctuation:
            ner = token_element.find('NER').text

            self.lemma = token_element.find('lemma').text
            self.pos = token_element.find('POS').text
            self.ner = ner if ner != 'O' else None

    # TODO: implement Similarity
    def compute_similarity(self, entities):
        """
        Compute the similarity of the token to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        if self.embedding_token is None:
            self.load_embedding()

        if self.criterion_exclude():
            return

        token = self.find_vocab()
        if not token:
            return

        similarities = []

        for entity in entities:
            similarities.append(
                max([self.embedding_token.similarity(token, entity_token)
                     if entity_token in self.embedding_token.vocab else -1. for entity_token in entity.split()])
            )

        if max(similarities) != -1:
            self.similarity = max(similarities)
            self.similar_entities = \
                [entities[i] for i in [idx for idx, val in enumerate(similarities) if val == self.similarity]]

    # endregion

    # region Methods criterion_

    def criterion_punctuation(self):
        """ Check if a token is a punctuation mark. """

        return True if self.word in self.punctuation else False

    def criterion_stopwords(self):
        """ Check if a token is a stop word. """

        return True if self.lemma in self.stopwords else False

    def criterion_determiner(self):
        """ Check if a token is a determiner. """

        return True if self.pos == 'DT' else False

    def criterion_number(self):
        """ Check if a token is a number. """

        return True if self.pos == 'CD' else False

    def criterion_adjective(self):
        """ Check if a token is an adjective. """

        return True if self.pos == 'JJ' else False

    def criterion_possessive(self):
        """ Check if a token is a possessive mark. """

        return True if self.pos == 'POS' else False

    def criterion_exclude(self):
        """ Check if a token must not be analyzed by similarity methods. """

        if self.criterion_punctuation() \
                or self.criterion_stopwords() \
                or self.criterion_determiner() \
                or self.criterion_number():
            return True

    # endregion

    # region Methods load_

    @classmethod
    @BaseClass.Verbose("Loading token embeddings...")
    def load_embedding(cls):
        """ Load the token embedding. """

        cls.embedding_token = KeyedVectors.load_word2vec_format(
            fname='../pre_trained_models/GoogleNews-vectors-negative300.bin',
            binary=True
        )

    # endregion

    # region Methods find_

    def find_vocab(self):
        """
        Finds a string that matches the embedding vocabulary. Otherwise, returns None.

        Returns:
            str, token that matches the embedding vocabulary or None.
        """

        if self.word in self.embedding_token.vocab:
            return self.word

        if self.word.lower() in self.embedding_token.vocab:
            return self.word.lower()

        elif self.lemma.lower() in self.embedding_token.vocab:
            return self.lemma.lower()

        else:
            return None

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = ['James Joyce', 'Richard Bernstein']
    tokens = [Token(token_element) for token_element in root.find('./document/sentences/sentence/tokens')]

    for token in tokens:
        token.compute_similarity(entities)

    Token.set_parameters(to_print=[], print_attribute=True)
    print(Token.to_string(tokens))


if __name__ == '__main__':
    main()
