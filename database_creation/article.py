from database_creation.utils import BaseClass
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from xml.etree import ElementTree
from collections import deque
from copy import copy


class Article(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = ['title', 'entities', 'sentences'], True, 1, 0

    context_range = 1

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

        self.entities = None
        self.entities_locations = None
        self.entities_persons = None
        self.entities_organizations = None

        self.sentences = None
        self.coreferences = None

        self.tuple_contexts = None

    # endregion

    # region Main methods

    def preprocess(self):
        """ Preprocess the article. """

        self.compute_title()
        self.compute_entities()
        self.compute_sentences()
        self.compute_coreferences()

    def process_candidates(self):
        """ Process the articles. """

        self.compute_similarities()
        self.compute_candidates()

    def process_tuples(self):
        """ Performs the processing of the frequent entity tuples of the article. """

        self.compute_tuple_contexts()

    def write_candidates(self, f):
        """
        Write the candidates of the articles in an opened file.

        Args:
            f: file, ready to be written in.
        """

        candidates = [np for sentence in self.sentences for np in sentence.nps if np.candidate]

        f.write(self.to_string(candidates))

    # endregion

    # region Methods compute_

    def compute_title(self):
        """ Compute the title of an article. """

        if self.title is None:
            tree = ElementTree.parse(self.original_path)
            root = tree.getroot()

            self.title = root.find('./head/title').text

    def compute_entities(self):
        """ Compute all the entities of an article. """

        if self.entities is None:
            tree = ElementTree.parse(self.original_path)
            root = tree.getroot()

            entities_locations = [entity.text for entity in root.findall('./head/docdata/identified-content/location')]
            entities_persons = [entity.text for entity in root.findall('./head/docdata/identified-content/person')]
            entities_organizations = [entity.text for entity in root.findall('./head/docdata/identified-content/org')]

            self.entities_locations = [self.standardize_location(entity) for entity in entities_locations]
            self.entities_persons = [self.standardize_person(entity) for entity in entities_persons]
            self.entities_organizations = [self.standardize_organization(entity) for entity in entities_organizations]

            self.entities = self.entities_locations + self.entities_persons + self.entities_organizations

    def compute_sentences(self):
        """ Compute the sentences of the article. """

        if self.sentences is None:
            root = ElementTree.parse(self.annotated_path)
            elements = root.findall('./document/sentences/sentence')

            self.sentences = {int(element.attrib['id']): Sentence(element) for element in elements}

    def compute_coreferences(self):
        """ Compute the coreferences of the article. """

        if self.coreferences is None:
            root = ElementTree.parse(self.annotated_path)
            elements = root.findall('./document/coreference/coreference')

            self.coreferences = [Coreference(element, self.entities) for element in elements]

    def compute_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for sentence in self.sentences:
            sentence.compute_similarities(self.entities)

    def compute_candidates(self):
        """ Computes and fills the candidate NPs of the article. """

        context = deque([self.sentences[i].text if 0 <= i < len(self.sentences) else ''
                         for i in range(-Article.context_range, Article.context_range + 1)])

        for i in range(len(self.sentences)):
            self.sentences[i].compute_candidates(entities=self.entities, context=copy(context))

            context.popleft()
            context.append(self.sentences[i + Article.context_range + 1].text
                           if i + Article.context_range + 1 < len(self.sentences) else '')

    def compute_tuple_contexts(self):
        """ Compute the tuple contexts of the article. """

        tuple_contexts = {}

        for entity_type in ['locations', 'persons', 'organizations']:
            entities = getattr(self, 'entities_' + entity_type)

            entity_tuples = self.subtuples(entities)

            for entity_tuple in entity_tuples:
                context = self.context(entity_tuple)
                if context is not None:
                    tuple_contexts[entity_tuple] = context

        self.tuple_contexts = tuple_contexts if tuple_contexts else None

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

        return max([len(self.entities_locations), len(self.entities_persons), len(self.entities_organizations)]) < 2

    def criterion_context(self):
        """
        Check if an article has a context.

        Returns:
            bool, True iff the article doesn't have a context.
        """

        return True if self.tuple_contexts is None else False

    # endregion

    # region Other methods

    def context(self, entity_tuple):
        """
        Returns the context for a single entity tuple, that is the sentences where the entities are referred to.

        Args:
            entity_tuple: tuple, entities to analyse.

        Returns:
            dict, context of the entity, ie the sentences with references to all the entities of the tuple, and the
            previous and following ones.
        """

        sentences = set()

        for entity in entity_tuple:
            coreferences = [c for c in self.coreferences if c.entity and c.entity == entity]

            if len(coreferences) == 0:
                return None

            else:
                coreference = coreferences[0]

                if not sentences:
                    sentences.update(coreference.sentences)
                else:
                    sentences.intersection_update(coreference.sentences)

                if not sentences:
                    return None

        sentences = sorted(sentences)
        context = {'sentences': sentences}

        for idx in sentences:
            sample = []

            for j in range(idx - self.context_range, idx + self.context_range + 1):
                try:
                    sample.append(self.sentences[j].text)
                except KeyError:
                    pass

            context['sample_' + str(idx)] = ' '.join(sample)

        return context

    # endregion


def main():
    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.preprocess()
    article.process_tuples()

    article.set_parameters(to_print=['entities', 'tuple_contexts'], print_attribute=True)
    print(article)


if __name__ == '__main__':
    main()
