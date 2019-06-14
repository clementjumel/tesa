from database_creation.utils import BaseClass, Similarity

from copy import copy
from string import punctuation as string_punctuation
from nltk.corpus import stopwords as nltk_stopwords


class Token(BaseClass):
    # region Class initialization

    to_print = ['word', 'pos', 'ner', 'similarity']
    print_attribute, print_lines, print_offsets = False, 0, 0

    punctuation = [p for p in string_punctuation] + ["''", '``']
    stopwords = set(nltk_stopwords.words('english'))
    embeddings_token = None

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

        word = token_element.find('word').text

        assert word
        self.word = word

        if self.word not in self.punctuation:
            ner = token_element.find('NER').text

            self.lemma = token_element.find('lemma').text
            self.pos = token_element.find('POS').text
            self.ner = ner if ner != 'O' else None

    def compute_similarity(self, entities):
        """
        Compute the similarity of the token to the entities in the article.

        Args:
            entities: dict, full entities of the sentence.
        """

        if self.embeddings_token is None:
            self.load_embeddings(type_='token')

        token = self.find_embedding()
        if not token:
            return

        sim = None

        for entity in entities['all']:
            token_sim = [self.embeddings_token.similarity(token, entity_token) for entity_token in entity.split()
                         if entity_token in self.embeddings_token.vocab]
            score = max(token_sim) if token_sim else None

            if score:
                if sim is None or score > sim.score:
                    sim = Similarity(score=score, items=[self.word], similar_items=[entity])

                elif sim is not None and sim.score == score:
                    sim.items.append(self.word)
                    sim.similar_items.append(entity)

        self.similarity = sim

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

        if self.criterion_punctuation() or self.criterion_stopwords() \
                or self.criterion_determiner() or self.criterion_number():
            return True

    # endregion

    # region Other methods

    def find_embedding(self):
        """
        Finds a string that matches the embeddings' vocabulary. Otherwise, returns None.

        Returns:
            str, token that matches the embedding vocabulary or None.
        """

        if self.criterion_exclude():
            return None

        else:
            if self.word in self.embeddings_token.vocab:
                return self.word

            if self.word.lower() in self.embeddings_token.vocab:
                return self.word.lower()

            elif self.lemma.lower() in self.embeddings_token.vocab:
                return self.lemma.lower()

            else:
                return None

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = {'person': ['James Joyce', 'Richard Bernstein'], 'location': [], 'organization': [],
                'all': ['James Joyce', 'Richard Bernstein']}
    tokens = [Token(token_element) for token_element in root.find('./document/sentences/sentence/tokens')]

    for token in tokens:
        token.compute_similarity(entities)

    print(Token.to_string(tokens))


if __name__ == '__main__':
    main()
