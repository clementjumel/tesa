from database_creation.utils import BaseClass

from copy import deepcopy


class Np(BaseClass):
    # region Class initialization

    to_print = ['tokens', 'token_similarity']
    print_attribute, print_lines, print_offsets = False, 1, 4

    def __init__(self, tokens):
        """
        Initializes an instance of Np.

        Args:
            tokens: dict, Tokens that define the Noun Phrase.
        """

        self.tokens = tokens
        self.token_similarity = None

    # endregion

    # region Methods compute_

    def compute_similarities(self, entities):
        """
        Compute the similarities of the NP to the entities.

        Args:
            entities: dict, full entities of the sentence.
        """

        self.compute_token_similarity(entities)

    def compute_token_similarity(self, entities):
        """
        Compute the token similarity of the NP to the entities in the article.

        Args:
            entities: dict, full entities of the sentence.
        """

        sim = None

        for idx in self.tokens:
            token = self.tokens[idx]

            token.compute_similarity(entities)

            if token.similarity is not None:
                if sim is None or token.similarity.score > sim.score:
                    sim = deepcopy(token.similarity)

                elif sim is not None and token.similarity.score == sim.score:
                    sim.items.extend(token.similarity.items)
                    sim.similar_items.extend(token.similarity.similar_items)

        self.token_similarity = sim

    # endregion


def main():
    from database_creation.sentence import Sentence
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = {'person': ['James Joyce', 'Richard Bernstein'], 'location': [], 'organization': [],
                'all': ['James Joyce', 'Richard Bernstein']}
    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    for sentence in sentences:
        sentence.compute_similarities(entities)

    for sentence in sentences:
        print(Np.to_string(sentence.nps))


if __name__ == '__main__':
    main()
