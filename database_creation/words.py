from database_creation.utils import BaseClass

from string import punctuation
from copy import copy


class Word(BaseClass):
    # region Class initialization

    to_print = []
    print_attribute = False
    print_lines = 0
    print_offsets = 0

    custom_punctuation = ("''", '``')

    # TODO
    word_embedding = None

    def __init__(self, text, pos):
        """
        Initializes an instance of Word.

        Args:
            text: str, the word itself.
            pos: str, Part Of Speech tag of the word.
        """

        assert text

        self.text = text
        self.pos = pos if pos not in punctuation and pos not in self.custom_punctuation else None

        self.similarity = None

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Word.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_print_parameters()[:4]
        attributes = copy(to_print) or list(self.__dict__.keys())

        string1 = self.prefix(print_lines=print_lines, print_offsets=print_offsets) + str(self.text)
        string2 = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'text' else ''
            string2 += ', ' if s and string2 else ''
            string2 += self.prefix(print_attribute=print_attribute, attribute=attribute) + s if s else ''

        return string1 + ' (' + string2 + ')' if string2 else string1

    # endregion

    # region Methods get

    def get_similarity(self, entities_locations, entities_persons, entities_organizations):
        """
        Compute the similarity of the word to the entities in the article.

        Args:
            entities_locations: list, location entities mentioned in the article.
            entities_persons: list, person entities mentioned in the article.
            entities_organizations: list, organization entities mentioned in the article.
        """
        pass
        # TODO
        # entity_words = [entity.split(' ') for entity in entities_locations] + \
        #                [entity.split(' ') for entity in entities_persons] + \
        #                [entity.split(' ') for entity in entities_organizations]
        #
        # entity_words = [item for sublist in entity_words for item in sublist]
        #
        # w_synset = wn.synsets(self.text)[0] if wn.synsets(self.text) else None
        # e_synsets = [wn.synsets(entity)[0] if wn.synsets(entity) else None for entity in entity_words]
        #
        # if w_synset and e_synsets:
        #     self.distance = max([w_synset.path_similarity(e_synset) if w_synset and e_synset and w_synset.path_similarity(e_synset) else 0. for e_synset in e_synsets])

    # endregion


def main():
    from database_creation.articles import Article

    a = Article('0', '', '../databases/nyt_jingyun/content_annotated/2000content_annotated/1185897.txt.xml')
    a.get_sentences()

    print(a.sentences[1])
    return


if __name__ == '__main__':
    main()
