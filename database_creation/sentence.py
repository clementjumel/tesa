from database_creation.utils import BaseClass
from database_creation.np import Np
from database_creation.token import Token

from nltk import tree


class Sentence(BaseClass):
    # region Class initialization

    to_print = ['text', 'nps']
    print_attribute, print_lines, print_offsets = False, 1, 2

    def __init__(self, sentence_element):
        """
        Initializes an instance of Sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.tokens = None
        self.parse = None
        self.text = None
        self.nps = None

        self.compute_tokens(sentence_element)
        self.compute_parse(sentence_element)
        self.compute_text()
        self.compute_nps()

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

    def compute_parse(self, sentence_element):
        """
        Compute the parsing of the sentence from its element.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.parse = sentence_element.find('./parse').text

    def compute_text(self):
        """ Compute the text defined by the tokens. """

        text = ''
        for idx in self.tokens:
            token = self.tokens[idx]

            text += '' if token.criterion_punctuation() or not text else ' '
            text += token.word

        self.text = text

    def compute_nps(self):
        """ Compute the NPs defined by self.parse. """

        t = tree.Tree.fromstring(self.parse)
        idx = 0
        for position in t.treepositions('leaves'):

            if position[-1] == 0:
                idx += 1
                t[position] += '|' + str(idx)

            # Case of non-breaking spaces inside a token
            else:
                old_position = tuple(list(position[:-1]) + [0])
                # Insert a non-breaking space
                t[old_position] = t[old_position].split('|')[0] + ' ' + t[position] + '|' + str(idx)
                t[position] = ''

        nps = []
        for leaves in [subtree.leaves() for subtree in t.subtrees(lambda node: node.label() == 'NP')]:
            tokens = {}

            for leaf in leaves:
                if leaf:
                    split = leaf.split('|')
                    idx = int(split[1])
                    word = split[0]

                    assert self.tokens[idx].word == word
                    tokens[idx] = self.tokens[idx]

            nps.append(Np(tokens))

        self.nps = nps

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    print(Sentence.to_string(sentences))
    return


if __name__ == '__main__':
    main()
