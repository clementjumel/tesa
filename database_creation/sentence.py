from database_creation.utils import BaseClass, Dependency
from database_creation.np import Np
from database_creation.token import Token

from nltk import tree


class Sentence(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], False, 1, 2

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

                governor_idx = int(dep.find('governor').attrib['idx'])
                dependent_idx = int(dep.find('dependent').attrib['idx'])

                if governor_idx != 0:
                    governor = self.tokens[governor_idx]
                    assert governor.word == dep.find('governor').text
                else:
                    governor = 'ROOT'

                dependent = self.tokens[dependent_idx]
                assert dependent.word == dep.find('dependent').text

                dependency_list.append(Dependency(type_=type_, governor=governor, dependent=dependent))

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
        idx = 1

        for position in t.treepositions('leaves'):
            t[position] += '|' + str(idx)
            idx += 1

        nps = []

        for leaves in [subtree.leaves() for subtree in t.subtrees(lambda node: node.label() == 'NP')]:
            tokens = {}

            for leaf in leaves:
                idx = int(leaf.split('|')[1])
                word = leaf.split('|')[0]
                assert self.tokens[idx].word == word

                tokens[idx] = self.tokens[idx]

            nps.append(Np(tokens))

        self.nps = nps

    def compute_similarities(self, entities):
        """
        Compute the similarities of the NPs to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        for np in self.nps:
            np.compute_similarities(entities)

    def compute_candidates(self, entities, context):
        """
        Compute the candidate NPs of the sentence.

        Args:
            entities: list, preprocessed entities of the sentence.
            context: collections.deque, queue containing the text of the sentences of the context.
        """

        for np in self.nps:
            np.compute_candidate(entities, context)

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = ['James Joyce', 'Richard Bernstein']
    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    # TODO: repair
    # for sentence in sentences:
    #     sentence.compute_similarities(entities)

    Sentence.set_parameters(to_print=[], print_attribute=True)
    print(Sentence.to_string(sentences))


if __name__ == '__main__':
    main()
