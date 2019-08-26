from database_creation.utils import BaseClass
from database_creation.token import Token


class Sentence(BaseClass):
    # region Class initialization

    to_print = ['text', 'tokens']
    print_attribute, print_lines, print_offsets = False, 1, 2

    def __init__(self, sentence_element):
        """
        Initializes an instance of Sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.tokens = None
        self.text = None

        self.compute_tokens(sentence_element)
        self.compute_text()

    # endregion

    # region Methods compute_

    def compute_tokens(self, sentence_element):
        """
        Compute the tokens of the sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        tokens = {}

        for token_element in sentence_element.findall('./tokens/token'):
            tokens[int(token_element.attrib['id'])] = Token(token_element)

        self.tokens = tokens

    def compute_text(self):
        """ Compute the text defined by the tokens. """

        text = ''
        for idx in self.tokens:
            token = self.tokens[idx]

            text += '' if token.criterion_punctuation() or not text else ' '
            text += token.start_tag if token.start_tag else ''
            text += token.word
            text += ' [' + token.entity + ']' if token.entity else ''
            text += token.end_tag if token.end_tag else ''

        self.text = text

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml')
    root = tree.getroot()

    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    print(Sentence.to_string(sentences))
    return


if __name__ == '__main__':
    main()
