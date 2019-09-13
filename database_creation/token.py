from string import punctuation as string_punctuation
from nltk.corpus import stopwords as nltk_stopwords


class Token:
    # region Class initialization

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
        self.start_tag = None
        self.end_tag = None

        self.compute_annotations(token_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Token.

        Returns:
            str, readable format of the instance.
        """

        s = self.start_tag if self.start_tag is not None else ''
        s += self.word
        s += self.end_tag if self.end_tag is not None else ''

        return s

    # endregion

    # region Methods compute_

    def compute_annotations(self, token_element):
        """
        Compute the annotations (word, lemma, Part Of Speech tag and Named-Entity Recognition) of the token.

        Args:
            token_element: ElementTree.Element, annotations of the token.
        """

        self.word = token_element.find('word').text
        self.lemma = token_element.find('lemma').text
        self.pos = token_element.find('POS').text
        self.ner = token_element.find('NER').text

    # endregion

    # region Methods criterion_

    def criterion_punctuation(self):
        """ Check if a token is a punctuation mark. """

        return True if self.word in self.punctuation else False

    def criterion_stopwords(self):
        """ Check if a token is a stop word. """

        return True if self.lemma in self.stopwords else False

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml')
    root = tree.getroot()

    tokens = [Token(token_element) for token_element in root.find('./document/sentences/sentence/tokens')]

    for token in tokens:
        print(', '.join([token.word, token.ner]))

    return


if __name__ == '__main__':
    main()
