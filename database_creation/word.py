from database_creation.utils import BaseClass

from copy import copy
from string import punctuation as string_punctuation
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import KeyedVectors


class Word(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], False, 0, 0

    punctuation = [p for p in string_punctuation] + ["''", '``']
    stopwords = set(nltk_stopwords.words('english'))
    embedding_word = None

    def __init__(self, text, pos):
        """
        Initializes an instance of Word.

        Args:
            text: str, the word itself.
            pos: str, Part Of Speech tag of the word.
        """

        assert text

        self.text = text
        self.pos = pos if text not in self.punctuation else None

        self.similarity = None
        self.similar_entities = None

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Word.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string1 = self.prefix(print_lines=print_lines, print_offsets=print_offsets) + str(self.text)
        string2 = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'text' else ''
            string2 += ', ' if s and string2 else ''
            string2 += self.prefix(print_attribute=print_attribute, attribute=attribute) + s if s else ''

        return string1 + ' (' + string2 + ')' if string2 else string1

    # endregion

    # region Methods compute_

    def compute_similarity(self, entities):
        """
        Compute the similarity of the word to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        if self.embedding_word is None:
            self.load_embedding()

        if self.criterion_exclude():
            return

        word = self.find_vocab()
        if not word:
            return

        similarities = []

        for entity in entities:
            similarities.append(
                max([self.embedding_word.similarity(word, entity_word)
                     if entity_word in self.embedding_word.vocab else -1. for entity_word in entity.split()])
            )

        if max(similarities) != -1:
            self.similarity = max(similarities)
            self.similar_entities = \
                [entities[i] for i in [idx for idx, val in enumerate(similarities) if val == self.similarity]]

    # endregion

    # region Methods criterion_

    def criterion_punctuation(self):
        """ Check if a word is a punctuation mark. """

        return True if self.pos is None else False

    def criterion_stopwords(self):
        """ Check if a word is a stop word. """

        return True if self.text.lower() in self.stopwords else False

    def criterion_determiner(self):
        """ Check if a word is a determiner. """

        return True if self.pos == 'DT' else False

    def criterion_number(self):
        """ Check if a word is a number. """

        return True if self.pos == 'CD' else False

    def criterion_adjective(self):
        """ Check if a word is an adjective. """

        return True if self.pos == 'JJ' else False

    def criterion_possessive(self):
        """ Check if a word is a possessive mark. """

        return True if self.pos == 'POS' else False

    def criterion_exclude(self):
        """ Check if a word must not be analyzed by similarity methods. """

        if self.criterion_punctuation() \
                or self.criterion_stopwords() \
                or self.criterion_determiner() \
                or self.criterion_number():

            return True

    # endregion

    # region Methods load_

    @classmethod
    @BaseClass.Verbose("Loading word embeddings...")
    def load_embedding(cls):
        """ Load the word embedding. """

        cls.embedding_word = KeyedVectors.load_word2vec_format(
            fname='../pre_trained_models/GoogleNews-vectors-negative300.bin',
            binary=True
        )

    # endregion

    # region Methods find_

    def find_vocab(self):
        """
        Finds a string that matches the embedding vocabulary. Otherwise, returns None.

        Returns:
            str, word that matches the embedding vocabulary or None.
        """

        if self.text in self.embedding_word.vocab:
            return self.text

        elif self.text.lower() in self.embedding_word.vocab:
            return self.text.lower()

        else:
            return None

    # endregion


def main():

    words = tuple([
        Word('the', 'DT'),
        Word('urban', 'JJ'),
        Word('city', 'NN'),
    ])

    entities = ['New York', 'San Francisco', 'town']

    for word in words:
        word.compute_similarity(entities)

    print(Word.to_string(words))

    return


if __name__ == '__main__':
    main()
