from database_creation.utils import BaseClass, Context
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from xml.etree import ElementTree
from collections import defaultdict


class Article(BaseClass):
    # region Class initialization

    to_print = ['entities', 'title', 'date', 'abstract', 'contexts']
    print_attribute, print_lines, print_offsets = True, 1, 0

    def __init__(self, original_path, annotated_path):
        """
        Initializes an instance of Article.

        Args:
            original_path: str, path of the article's original corpus' content.
            annotated_path: str, path of the article's annotated corpus' content.
        """

        self.original_path = original_path
        self.annotated_path = annotated_path

        self.title = None
        self.date = None
        self.abstract = None
        self.entities = None
        self.raw_entities = None

        self.sentences = None
        self.coreferences = None
        self.contexts = None

    # endregion

    # region Methods compute_

    def compute_metadata(self):
        """ Compute the metadata (title, date, abstract, entities, raw_entities) of the article. """

        root = ElementTree.parse(self.original_path).getroot()

        title = self.title or self.get_title(root)
        date = self.date or self.get_date(root)
        abstract = self.abstract or self.get_abstract(root)
        entities, raw_entities = (self.entities, self.raw_entities) if self.entities and self.raw_entities \
            else self.get_entities(root)

        self.title = title
        self.date = date
        self.abstract = abstract
        self.entities = entities
        self.raw_entities = raw_entities

    def compute_annotations(self):
        """ Compute the annotations (sentences, coreferences) of the article. """

        root = ElementTree.parse(self.annotated_path).getroot()

        sentences = self.sentences or self.get_sentences(root)
        coreferences = self.coreferences or self.get_coreferences(root)

        self.sentences = sentences
        self.coreferences = coreferences

    def compute_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for idx in self.sentences:
            self.sentences[idx].compute_similarities(self.entities)

    def compute_contexts(self, entities, type_):
        """
        Compute the contexts of the article for the entities, according to the specified context type.

        Args:
            entities: tuple, entities mentioned in the article.
            type_: str, type of the context, must be 'same_sent', 'neigh_sent', or 'same_role'.
        """

        self.contexts = self.contexts or dict()
        self.contexts[entities] = getattr(self, 'contexts_' + type_)(entities)

    # endregion

    # region Methods get_

    @staticmethod
    def get_title(root):
        """
        Returns the title of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            str, title of the article.
        """

        return root.find('./head/title').text

    @staticmethod
    def get_date(root):
        """
        Returns the date of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            str, date of the article.
        """

        d = root.find('./head/meta[@name="publication_day_of_month"]').get('content')
        m = root.find('./head/meta[@name="publication_month"]').get('content')
        y = root.find('./head/meta[@name="publication_year"]').get('content')

        d = '0' + d if len(d) == 1 else d
        m = '0' + m if len(m) == 1 else m

        return '/'.join([y, m, d])

    @staticmethod
    def get_abstract(root):
        """
        Returns the abstract of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            str, abstract of the article.
        """

        return root.find('./body/body.head/abstract/p').text

    def get_entities(self, root):
        """
        Returns the entities and the raw_entities of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            entities: dict, standardized entities of the article sorted into 'all' or each type.
            raw_entities: dict, mapping from the standardized entities to their original versions.
        """

        locations = [(self.standardize_location(entity.text), entity.text)
                     for entity in root.findall('./head/docdata/identified-content/location')]
        persons = [(self.standardize_person(entity.text), entity.text)
                   for entity in root.findall('./head/docdata/identified-content/person')]
        organizations = [(self.standardize_organization(entity.text), entity.text)
                         for entity in root.findall('./head/docdata/identified-content/org')]

        entities = {'location': [entity[0] for entity in locations],
                    'person': [entity[0] for entity in persons],
                    'organization': [entity[0] for entity in organizations],
                    'all': [entity[0] for entity in locations + persons + organizations]}
        raw_entities = dict(locations + persons + organizations)

        return entities, raw_entities

    @staticmethod
    def get_sentences(root):
        """
        Returns the Sentences of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            dict, Sentences of the article (mapped with their indexes).
        """

        elements = root.findall('./document/sentences/sentence')
        sentences = {int(element.attrib['id']): Sentence(element) for element in elements}

        return sentences

    def get_coreferences(self, root):
        """
        Returns the Coreferences of an article given the tree of its metadata.

        Args:
            root: ElementTree.root, root of the metadata of the article.

        Returns:
            list, Coreferences of the article.
        """

        elements = root.findall('./document/coreference/coreference')
        coreferences = [Coreference(element, self.entities) for element in elements]

        return coreferences

    def get_entity_sentences(self, entity):
        """
        Returns the indexes of the sentences where there is a mention of the specified entity.

        Args:
            entity: str, entity we want to find mentions of.

        Returns:
            list, sorted list of sentences' indexes.
        """

        entity_sentences = set()

        for coreference in [c for c in self.coreferences if c.entity and c.entity == entity]:
            entity_sentences.update(coreference.sentences)

        return sorted(entity_sentences)

    # endregion

    # region Methods criterion_

    def criterion_data(self):
        """
        Check if an article's data is complete, ie if its annotation file exists.

        Returns:
            bool, True iff the article's data is incomplete.
        """

        try:
            f = open(self.annotated_path, 'r')
            f.close()
            return False

        except FileNotFoundError:
            return True

    def criterion_entity(self):
        """
        Check if an article has at least 2 entities of the same type.

        Returns:
            bool, True iff the article hasn't 2 entities of the same type.
        """

        if max([len(self.entities['location']), len(self.entities['person']), len(self.entities['organization'])]) < 2:
            return True
        else:
            return False

    # endregion

    # region Methods contexts_

    def contexts_same_sent(self, entity_tuple):
        """
        Returns the same-sentence contexts for a single entity tuple, that is the sentences where all the entities are
        mentioned.

        Args:
            entity_tuple: tuple, entities to analyse.

        Returns:
            dict, same-sentence Contexts of the entities mapped with their sentence idx.
        """

        sentences = defaultdict(int)

        for entity in entity_tuple:
            for idx in self.get_entity_sentences(entity):
                sentences[idx] += 1

        sentences = [idx for idx in sentences if sentences[idx] == len(entity_tuple)]

        contexts = dict()

        for idx in sentences:
            before_texts, before_idxs = [], []

            try:
                before_texts.append(self.sentences[idx - 1].text)
                before_idxs.append(idx - 1)
            except KeyError:
                pass

            contexts[idx] = Context(sentence_texts=[self.sentences[idx].text], sentence_idxs=[idx],
                                    before_texts=before_texts, before_idxs=before_idxs,
                                    after_texts=None, after_idxs=None)

        return contexts

    def contexts_neigh_sent(self, entity_tuple):
        """
        Returns the neighboring-sentences contexts for a single entity tuple, that is the neighboring sentences where
        the entities are mentioned.

        Args:
            entity_tuple: tuple, entities to analyse.

        Returns:
            dict, neighbouring-sentences Contexts of the entities, mapped with their sentences span (indexes of the
            first and last sentences separated by |).
        """

        sentences = defaultdict(set)

        for i in range(len(entity_tuple)):
            for idx in self.get_entity_sentences(entity_tuple[i]):
                sentences[idx].add(i)

        contexts_sentences = set()

        for idx in sentences:
            unseens = list(range(len(entity_tuple)))
            seers = set()

            for i in range(len(entity_tuple)):
                if idx + i in sentences:
                    for j in sentences[idx + i]:
                        try:
                            unseens.remove(j)
                            seers.add(idx + i)
                        except ValueError:
                            pass

                    if not unseens:
                        seers = sorted(seers)
                        contexts_sentences.add(tuple(range(seers[0], seers[-1] + 1)))
                        break

        contexts_sentences = sorted(contexts_sentences)
        contexts = dict()

        for idxs in contexts_sentences:
            id_ = str(idxs[0]) + '|' + str(idxs[-1])
            contexts[id_] = Context(sentence_texts=[self.sentences[idx].text for idx in idxs],
                                    sentence_idxs=list(idxs))

        return contexts

    # endregion


def main():
    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.compute_metadata()
    article.compute_annotations()

    article.compute_contexts(('James Joyce', 'Richard Bernstein'), 'neigh_sent')

    print(article)


if __name__ == '__main__':
    main()
