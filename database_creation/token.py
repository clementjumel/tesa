from database_creation.utils import BaseClass

from string import punctuation as string_punctuation
from nltk.corpus import stopwords as nltk_stopwords


class Token(BaseClass):
    # region Class initialization

    to_print = ['word', 'pos', 'ner']
    print_attribute, print_lines, print_offsets = False, 0, 0

    punctuation = [p for p in string_punctuation] + ["''", '``']
    stopwords = set(nltk_stopwords.words('english'))

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

        self.compute_annotations(token_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Token.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        attributes = to_print or list(self.__dict__.keys())

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

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    tokens = [Token(token_element) for token_element in root.find('./document/sentences/sentence/tokens')]

    print(Token.to_string(tokens))
    return


if __name__ == '__main__':
    main()
