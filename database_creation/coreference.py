from database_creation.utils import BaseClass


class Coreference(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], True, 1, 2

    def __init__(self, element):
        """
        Initializes an instance of the Coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
        """

        self.representative = None
        self.mentions = None

        self.compute_attributes(element)

    # endregion

    # region Methods compute_

    def compute_attributes(self, element):
        """
        Compute the mentions and the representative of the coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
        """

        representative, mentions = [], []

        for mention in element.findall('./mention'):

            if mention.attrib and mention.attrib['representative'] == 'true':
                representative.append(mention.find('text').text)

            else:
                mentions.append(mention.find('text').text)

            element.remove(mention)

        assert len(representative) == 1 and not element

        self.representative = representative[0]
        self.mentions = mentions

    # endregion


def main():
    from database_creation.article import Article

    article = Article('0', '', '../databases/nyt_jingyun/content_annotated/2000content_annotated/1185897.txt.xml')
    article.compute_coreferences()

    print(article.coreferences[0])
    return


if __name__ == '__main__':
    main()
