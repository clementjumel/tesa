from database_creation.utils import BaseClass
from database_creation.np import Np
from database_creation.word import Word

from nltk import tree


class Sentence(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = ['nps'], False, 1, 2

    def __init__(self, element):
        """
        Initializes an instance of Sentence.

        Args:
            element: ElementTree.Element, annotations of the sentence.
        """

        self.parse = None
        self.text = None
        self.nps = None

        self.compute_parse(element)
        self.compute_text()
        self.compute_nps()

    # endregion

    # region Methods compute_

    def compute_parse(self, element):
        """
        Compute the parsing of the sentence from its element.

        Args:
            element: ElementTree.Element, annotations of the sentence.
        """

        parses = element.findall('./parse')
        assert len(parses) == 1

        self.parse = parses[0].text

    def compute_text(self):
        """ Compute the text defined by self.parse. """

        t = tree.Tree.fromstring(self.parse)
        words = t.leaves()

        text = ''
        for word in words:
            text += '' if word[0] in Word.punctuation else ' '
            text += word

        self.text = text

    def compute_nps(self):
        """ Compute the NPs defined by self.parse. """

        t = tree.Tree.fromstring(self.parse)
        nps = [child.pos() for child in t.subtrees(lambda node: node.label() == 'NP')]

        self.nps = [Np(np) for np in nps]

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

    from database_creation.article import Article

    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.compute_sentences()

    for i in range(1, 4):
        article.sentences[i].compute_similarities(['The New York Times'])

        print(article.sentences[i])

    return


if __name__ == '__main__':
    main()
