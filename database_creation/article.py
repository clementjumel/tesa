from database_creation.utils import BaseClass, Context
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from xml.etree import ElementTree
from collections import deque
from copy import copy
from collections import defaultdict


class Article(BaseClass):
    # region Class initialization

    to_print = ['entities', 'title', 'contexts']
    print_attribute, print_lines, print_offsets = True, 1, 0

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

        self.sentences = None
        self.coreferences = None

        self.contexts = None

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

    def process_contexts(self):
        """ Performs the processing of the frequent entity tuples of the article. """

        self.compute_contexts(['same_sent', 'neigh_sent', 'same_role'])

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

            entities_location = [entity.text for entity in root.findall('./head/docdata/identified-content/location')]
            entities_person = [entity.text for entity in root.findall('./head/docdata/identified-content/person')]
            entities_organization = [entity.text for entity in root.findall('./head/docdata/identified-content/org')]

            entities_location = [self.standardize_location(entity) for entity in entities_location]
            entities_person = [self.standardize_person(entity) for entity in entities_person]
            entities_organization = [self.standardize_organization(entity) for entity in entities_organization]

            self.entities = {
                'all': entities_location + entities_person + entities_organization,
                'location': entities_location,
                'person': entities_person,
                'organization': entities_organization
            }

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
            entities = self.entities['all']

            self.coreferences = [Coreference(element, entities) for element in elements]

    def compute_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for sentence in self.sentences:
            sentence.compute_similarities(self.entities['all'])

    def compute_candidates(self):
        """ Computes and fills the candidate NPs of the article. """

        context = deque([self.sentences[i].text if 0 <= i < len(self.sentences) else ''
                         for i in range(-Article.context_range, Article.context_range + 1)])

        for i in range(len(self.sentences)):
            self.sentences[i].compute_candidates(entities=self.entities['all'], context=copy(context))

            context.popleft()
            context.append(self.sentences[i + Article.context_range + 1].text
                           if i + Article.context_range + 1 < len(self.sentences) else '')

    def compute_contexts(self, types):
        """
        Compute some contexts of the article, according to the specified contexts types.

        Args:
            types: list, types of the context, each one being 'same_sent', 'neigh_sent', or 'same_role'.
        """

        contexts = {}

        entities_tuples = set()
        for entity_type in ['location', 'person', 'organization']:
            entities = getattr(self, 'entities')[entity_type]

            entities_tuples.update(self.subtuples(entities))

        for type_ in types:
            type_contexts = {}

            for entity_tuple in entities_tuples:
                type_contexts[entity_tuple] = getattr(self, 'contexts_' + type_)(entity_tuple)

            contexts[type_] = type_contexts

        self.contexts = contexts

    # endregion

    # region Methods get_

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

    def criterion_context(self):
        """
        Check if an article has a context.

        Returns:
            bool, True iff the article doesn't have a context.
        """

        if self.contexts['same_sent'] is None and self.contexts[' neigh_sent'] is None \
                and self.contexts['same_role'] is None:
            return True
        else:
            return False

    # endregion

    # region Methods context_

    def contexts_same_sent(self, entity_tuple):
        """
        Returns the same-sentence contexts for a single entity tuple, that is the sentences where all the entities are
        mentioned.

        Args:
            entity_tuple: tuple, entities to analyse.

        Returns:
            list, same-sentence Contexts of the entities.
        """

        sentences = defaultdict(int)

        for entity in entity_tuple:
            for idx in self.get_entity_sentences(entity):
                sentences[idx] += 1

        sentences = [idx for idx in sentences if sentences[idx] == len(entity_tuple)]

        contexts = []

        for idx in sentences:
            before_texts, before_idxs = [], []

            for i in range(self.context_range):
                try:
                    before_texts.append(self.sentences[idx - self.context_range + i].text)
                    before_idxs.append(idx - self.context_range + i)
                except KeyError:
                    pass

            contexts.append(Context(sentence_texts=[self.sentences[idx].text], sentence_idxs=[idx],
                                    before_texts=before_texts, before_idxs=before_idxs,
                                    after_texts=None, after_idxs=None))

        return contexts

    def contexts_neigh_sent(self, entity_tuple):
        """
        Returns the neighboring-sentences contexts for a single entity tuple, that is the neighboring sentences where
        the entities are mentioned.

        Args:
            entity_tuple: tuple, entities to analyse.

        Returns:
            list, neighbouring-sentences Contexts of the entities.
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
                        contexts_sentences.add(tuple(sorted(seers)))
                        break

        contexts_sentences = sorted(contexts_sentences)
        contexts = []

        for idxs in contexts_sentences:
            contexts.append(Context(sentence_texts=[self.sentences[idx].text if idx in idxs else ''
                                                    for idx in range(idxs[0], idxs[-1] + 1)],
                                    sentence_idxs=list(range(idxs[0], idxs[-1] + 1))))

        return contexts

    def contexts_same_role(self, entity_tuple):
        # TODO: create function
        return []

    # endregion


def main():
    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.preprocess()
    article.process_contexts()

    # article.set_parameters(to_print=[], print_attribute=True)
    print(article)


if __name__ == '__main__':
    main()
