from database_creation.utils import BaseClass, Mention


class Coreference(BaseClass):
    # region Class initialization

    to_print = ['representative', 'mentions', 'entity']
    print_attribute, print_lines, print_offsets = True, 1, 2

    def __init__(self, element, entities):
        """
        Initializes an instance of the Coreference.

        Args:
            element: ElementTree.Element, annotations of the coreference.
            entities: dict, full entities of the articles.
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
            entities: dict, full entities of the articles.
        """

        for m in [self.representative] + self.mentions:
            for type_ in entities:
                if type_ != 'all':
                    for entity in entities[type_]:
                        if self.match(m.text, entity, type_=type_, flexible=True):
                            self.entity = entity
                            return

    def compute_sentences(self):
        """ Compute the indexes of the sentences of the coreference chain. """

        self.sentences = set([m.sentence for m in [self.representative] + self.mentions])

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = {'person': ['James Joyce', 'Richard Bernstein'], 'location': [], 'organization': [],
                'all': ['James Joyce', 'Richard Bernstein']}
    coreferences = [Coreference(coreference_element, entities) for coreference_element
                    in root.findall('./document/coreference/coreference')]

    print(Coreference.to_string(coreferences))


if __name__ == '__main__':
    main()
