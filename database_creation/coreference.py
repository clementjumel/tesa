from database_creation.utils import Mention


class Coreference:
    # region Class initialization

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

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Coreference.

        Returns:
            str, readable format of the instance.
        """

        entity = '[' + self.entity + '] ' if self.entity is not None else ''

        return entity + '; '.join([str(mention) for mention in [self.representative] + self.mentions])

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

            text = m.find('text').text
            if len(text.split()) <= 10:
                mentions.append(Mention(text=text, sentence=int(m.find('sentence').text),
                                        start=int(m.find('start').text), end=int(m.find('end').text)))

        self.representative = representative
        self.mentions = mentions

    def compute_entity(self, entities):
        """
        Compute the entity the coreference refer to, or None.

        Args:
            entities: set, Entities of the articles.
        """

        texts = set([self.representative.text] + [mention.text for mention in self.mentions])

        matches = [str(entity) for entity in entities
                   if entity.match_string(string=self.representative.text, flexible=False)]

        if len(matches) == 1:
            self.entity = matches[0]
            return
        elif len(matches) > 1:
            return

        matches = [str(entity) for entity in entities
                   if entity.match_string(string=self.representative.text, flexible=True)]

        if len(matches) == 1:
            self.entity = matches[0]
            return
        elif len(matches) > 1:
            return

        matches = [str(entity) for entity in entities for text in texts
                   if entity.match_string(string=text, flexible=False)]

        if len(matches) == 1:
            self.entity = matches[0]
        elif len(matches) > 1:
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
        print(str(coreference))

    return


if __name__ == '__main__':
    main()
