from database_creation.utils import BaseClass
from database_creation.word import Word


class Np(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], False, 1, 4

    def __init__(self, np):
        """
        Initializes an instance of Np.

        Args:
            np: list, tuples (text, pos) that defines a word.
        """

        self.words = tuple([Word(text=word[0], pos=word[1]) for word in np])

        self.similarity = None
        self.score = None
        self.candidate = None
        self.entities = None
        self.context = None

    # endregion

    # region Methods compute_

    def compute_similarities(self, entities_locations, entities_persons, entities_organizations):
        """
        Compute the similarities of the NP and its words to the entities in the article.

        Args:
            entities_locations: list, location entities mentioned in the article.
            entities_persons: list, person entities mentioned in the article.
            entities_organizations: list, organization entities mentioned in the article.
        """

        for word in self.words:
            word.compute_similarity(entities_locations, entities_persons, entities_organizations)

        # TODO
        # self.similarity = max([word.distance if word.distance else 0. for word in self.words])
        # self.score =

    def compute_candidate(self, entities_locations, entities_persons, entities_organizations, context):
        """
        Compute the candidate NPs of the sentence.

        Args:
            entities_locations: list, location entities mentioned in the article.
            entities_persons: list, person entities mentioned in the article.
            entities_organizations: list, organization entities mentioned in the article.
            context: collections.deque, queue containing the text of the sentences of the context.
        """

        self.candidate = True
        self.entities = entities_locations + entities_persons + entities_organizations
        self.context = ' '.join([context.popleft() for _ in range(len(context))])

    # endregion

    # region Methods criterion_

    def criterion_candidate(self):
        # TODO
        return False

    # endregion


def main():
    from database_creation.article import Article

    article = Article('0', '', '../databases/nyt_jingyun/content_annotated/2000content_annotated/1185897.txt.xml')
    article.compute_sentences()

    print(article.sentences[0])
    return


if __name__ == '__main__':
    main()
