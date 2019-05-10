from database_creation.utils import BaseClass
from database_creation.nps import Np

from nltk import tree


class Sentence(BaseClass):
    # region Class initialization

    to_print = ['nps', 'similarity']
    print_attribute = False
    print_lines = 1
    print_offsets = 2

    def __init__(self, element):
        """
        Initializes an instance of Sentence.

        Args:
            element: ElementTree.Element, annotations of the sentence.
        """

        parses = element.findall('./parse')
        assert len(parses) == 1
        self.parse = parses[0].text

        # TODO
        # tokens = element.findall('./tokens')
        # assert len(tokens) == 1
        # self.tokens = ElementTree.tostring(tokens[0]).decode()

        # dependencies = element.findall('./dependencies')
        # self.dependencies = [ElementTree.tostring(dependency).decode() for dependency in dependencies]

        # machine_readings = element.findall('./MachineReading')
        # self.machine_reading = [ElementTree.tostring(mr).decode() for mr in machine_readings]

        # [element.remove(parse) for parse in parses]
        # [element.remove(token) for token in tokens]
        # [element.remove(dependency) for dependency in dependencies]
        # [element.remove(machine_reading) for machine_reading in machine_readings]
        # assert not element

        self.nps = self.get_nps()

    # endregion

    # region Methods get

    def get_nps(self):
        """
        Compute the NPs defined by self.parse.

        Returns:
            list, NPs contained in the parse.
        """

        t = tree.Tree.fromstring(self.parse)
        nps = [child.pos() for child in t.subtrees(lambda node: node.label() == 'NP')]

        return [Np(np) for np in nps]

    def get_similarities(self, entities_locations, entities_persons, entities_organizations):
        """
        Compute the similarities of the NPs to the entities in the article.

        Args:
            entities_locations: list, location entities mentioned in the article.
            entities_persons: list, person entities mentioned in the article.
            entities_organizations: list, organization entities mentioned in the article.
        """

        for np in self.nps:
            np.get_similarities(entities_locations, entities_persons, entities_organizations)

    def get_candidates_idx(self):
        """
        Compute the indexes of the candidate NP of the sentence.

        Returns:
            list, indexes of the candidates in the sentence.
        """

        candidates_idxs = []

        for i in range(len(self.nps)):
            if self.nps[i].candidate_criterion():
                candidates_idxs.append(i)

        return candidates_idxs

    # TODO
    def get_candidate_info(self):
        pass

    # endregion


def main():
    from database_creation.articles import Article

    a = Article('0', '', '../databases/nyt_jingyun/content_annotated/2000content_annotated/1185897.txt.xml')
    a.get_sentences()

    print(a.sentences[1])
    return


if __name__ == '__main__':
    main()
