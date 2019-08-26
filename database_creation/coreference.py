from database_creation.utils import BaseClass, Mention


class Coreference(BaseClass):
    # region Class initialization

    to_print = ['representative', 'mentions', 'entity', 'sentences']
    print_attribute, print_lines, print_offsets = True, 1, 2

    def __init__(self, element, entities):
        """
        Initializes an instance of the Coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
            entities: set, Entities of the articles.
        """

        self.representative = None
        self.mentions = None
        self.entity = None
        self.sentences = None

        self.compute_annotations(element)
        self.compute_entity(entities)
        self.compute_sentences()

    # endregion

    # region Methods compute_

    def compute_annotations(self, element):
        """
        Compute the mentions and the representative of the coreference, and the corresponding sentences.

        Args:
            element: ElementTree.Element, annotations of the coreference.
        """

        element_mentions = element.findall('./mention')

        m = element_mentions[0]
        assert m.attrib and m.attrib['representative'] == 'true'
        representative = Mention(text=m.find('text').text, sentence=int(m.find('sentence').text),
                                 start=int(m.find('start').text), end=int(m.find('end').text))

        mentions = []
        for i in range(1, len(element_mentions)):
            m = element_mentions[i]
            mentions.append(Mention(text=m.find('text').text, sentence=int(m.find('sentence').text),
                                    start=int(m.find('start').text), end=int(m.find('end').text)))

        self.representative = representative
        self.mentions = mentions

    def compute_entity(self, entities):
        """
        Compute the entity the coreference refer to, or None.

        Args:
            entities: set, Entities of the articles.
        """

        for m in [self.representative] + self.mentions:
            for entity in entities:
                if entity.match(m.text, flexible=True):
                    self.entity = str(entity)
                    return

    def compute_sentences(self):
        """ Compute the indexes of the sentences of the coreference chain. """

        self.sentences = set([m.sentence for m in [self.representative] + self.mentions])

    # endregion


def main():
    from database_creation.utils import Entity
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml')
    root = tree.getroot()

    entities = {Entity(original_name='Babel, Isaac', type_='person'),
                Entity(original_name='Campbell, James', type_='person')}

    coreferences = [Coreference(coreference_element, entities) for coreference_element
                    in root.findall('./document/coreference/coreference')]

    for coreference in coreferences:
        if coreference.entity is not None:
            print(Coreference.to_string(coreference))
    return


if __name__ == '__main__':
    main()
