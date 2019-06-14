from database_creation.utils import BaseClass, Dependency
from database_creation.np import Np
from database_creation.token import Token

from nltk import tree


class Sentence(BaseClass):
    # region Class initialization

    to_print = ['tokens', 'text', 'nps', 'dependencies']
    print_attribute, print_lines, print_offsets = False, 1, 2

    def __init__(self, sentence_element):
        """
        Initializes an instance of Sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.tokens = None
        self.parse = None
        self.dependencies = None

        self.text = None
        self.nps = None

        self.compute_tokens(sentence_element)
        self.compute_parse(sentence_element)
        self.compute_dependencies(sentence_element)

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

    def compute_dependencies(self, sentence_element):
        """
        Compute the dependencies of the sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        dependencies = {}

        for dependency_element in sentence_element.findall('./dependencies'):
            dependency_list = []

            for dep in dependency_element:
                type_ = dep.attrib['type']

                gov_idx = int(dep.find('governor').attrib['idx'])
                dep_idx = int(dep.find('dependent').attrib['idx'])

                if gov_idx != 0:
                    gov_word = self.tokens[gov_idx].word
                    assert gov_word == dep.find('governor').text
                else:
                    gov_word = 'ROOT'

                dep_word = self.tokens[dep_idx].word
                assert dep_word == dep.find('dependent').text

                dependency_list.append(Dependency(type_=type_,
                                                  gov_word=gov_word, gov_idx=gov_idx,
                                                  dep_word=dep_word, dep_idx=dep_idx))

            dependencies[dependency_element.attrib['type']] = dependency_list

        self.dependencies = dependencies

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

    def compute_similarities(self, entities):
        """
        Compute the similarities of the NPs to the entities in the article.

        Args:
            entities: dict, full entities of the sentence.
        """

        for np in self.nps:
            np.compute_similarities(entities)

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = {'person': ['James Joyce', 'Richard Bernstein'], 'location': [], 'organization': [],
                'all': ['James Joyce', 'Richard Bernstein']}
    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    for sentence in sentences:
        sentence.compute_similarities(entities)

    print(Sentence.to_string(sentences))


if __name__ == '__main__':
    main()
