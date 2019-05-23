from database_creation.utils import BaseClass


class Coreference(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], True, 1, 2

    def __init__(self, element, entities):
        """
        Initializes an instance of the Coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
            entities: list, entities of the articles.
        """

        self.representative = None
        self.mentions = None

        self.entity = None
        self.sentences = None

        self.compute_mentions(element)

        self.compute_entity(entities)
        self.compute_sentences()

    # endregion

    # region Methods compute_

    def compute_mentions(self, element):
        """
        Compute the mentions and the representative of the coreference, and the corresponding sentences.

        Args:
            element: ElementTree.Element, annotations of the coreference.
        """

        element_mentions = element.findall('./mention')

        mention = element_mentions[0]
        assert mention.attrib and mention.attrib['representative'] == 'true'
        representative = [mention.find('text').text, mention.find('sentence').text,
                          mention.find('start').text, mention.find('end').text]

        mentions = []
        for i in range(1, len(element_mentions)):

            mention = element_mentions[i]
            mentions.append([mention.find('text').text, mention.find('sentence').text,
                             mention.find('start').text, mention.find('end').text])

        self.representative = representative
        self.mentions = mentions

    def compute_entity(self, entities):
        """
        Compute the entity the coreference refer to, or None.

        Args:
            entities: list, entities of the articles.
        """

        for mention in [self.representative] + self.mentions:
            text = self.get_text(mention)

            for entity in entities:
                if self.match(text, entity):
                    self.entity = entity
                    return

    def compute_sentences(self):
        """ Compute the indexes of the sentences of the coreference chain. """

        sentences = set([self.get_sentence(mention) for mention in [self.representative] + self.mentions])

        self.sentences = sentences

    # endregion

    # region Methods get_

    @staticmethod
    def get_text(mention):
        """
        Returns the text of a mention.

        Args:
            mention: tuple, mention to analyse.

        Returns:
            str, text of the mention.
        """

        return mention[0]

    @staticmethod
    def get_sentence(mention):
        """
        Returns the sentence index of a mention.

        Args:
            mention: tuple, mention to analyse.

        Returns:
            int, index of the sentence of the mention.
        """

        return int(mention[1])

    @staticmethod
    def get_range(mention):
        """
        Returns the range of words of a mention.

        Args:
            mention: tuple, mention to analyse.

        Returns:
            tuple, start and end indexes of the mention.
        """

        return int(mention[2]), int(mention[3])

    # endregion


def main():

    from database_creation.article import Article

    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.compute_entities()
    article.compute_coreferences()

    print(BaseClass.to_string(article.entities))

    for i in range(3):
        print(article.coreferences[i])

    return


if __name__ == '__main__':
    main()
