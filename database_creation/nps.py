from database_creation.utils import BaseClass
from database_creation.words import Word


class Np(BaseClass):
    # region Class initialization

    to_print = []
    print_attribute = False
    print_lines = 1
    print_offsets = 4

    def __init__(self, np):
        """
        Initializes an instance of Np.

        Args:
            np: list, words of the NP (each one being a tuple (text, pos)).
        """

        self.words = tuple([Word(text=word[0], pos=word[1]) for word in np])

        self.similarity = None
        self.candidate = None
        self.entities = None
        self.context = None

    # endregion

    # region Methods get

    def get_similarities(self, entities_locations, entities_persons, entities_organizations):
        """
        Compute the similarities of the NP and its words to the entities in the article.

        Args:
            entities_locations: list, location entities mentioned in the article.
            entities_persons: list, person entities mentioned in the article.
            entities_organizations: list, organization entities mentioned in the article.
        """

        for word in self.words:
            word.get_similarity(entities_locations, entities_persons, entities_organizations)

        # TODO
        # self.similarity = max([word.distance if word.distance else 0. for word in self.words])

    # endregion

    # region Other methods

    def candidate_criterion(self):
        """ Defines if a NP is can be a candidate. """

        # TODO
        self.candidate = False

    # endregion


def main():
    from database_creation.articles import Article

    a = Article('0', '', '../databases/nyt_jingyun/content_annotated/2000content_annotated/1185897.txt.xml')
    a.get_sentences()

    print(a.sentences[1])
    return


if __name__ == '__main__':
    main()
