from database_creation.utils import BaseClass
from database_creation.token import Token

from copy import deepcopy
from gensim.models import KeyedVectors


class Np(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], True, 1, 4

    embedding_entity = None
    token_threshold, entity_threshold = 0.3, 0.3

    def __init__(self, tokens):
        """
        Initializes an instance of Np.

        Args:
            tokens: dict, Tokens that define the Noun Phrase.
        """

        self.tokens = tokens

        self.token_similarity = None
        self.entity_similarity = None

        self.candidate = None
        self.entities = None
        self.context = None

    # endregion

    # region Methods compute_

    def compute_similarities(self, entities):
        """
        Compute the similarities of the NP to the entities.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        self.compute_token_similarity(entities)
        # TODO: repair
        # self.compute_entity_similarity(entities)

    def compute_token_similarity(self, entities):
        """
        Compute the token similarity of the NP to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
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

    # TODO: repair
    def compute_entity_similarity(self, entities):
        """
        Compute the entity similarity of the NP to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        if self.embedding_entity is None:
            self.load_embedding()

        vocab_np = self.find_vocab_np()
        if not vocab_np:
            return

        vocab_entities = self.find_vocab_entities(entities)
        if not vocab_entities:
            return

        similarities, similar_entities = [], []

        for np in vocab_np:
            for entity in vocab_entities:
                similarity = self.embedding_entity.similarity(np, entity)

                similarities.append(similarity) if similarity else None
                similar_entities.append((np, entity)) if similarity else None

        if similarities:
            self.entity_similarity = max(similarities)
            # self.entity_similar_entities = [similar_entities[i] for i in [idx for idx, val in enumerate(similarities)
            #                                                               if val == self.entity_similarity]]

    def compute_candidate(self, entities, context):
        """
        Compute the candidate attributes of an NP.

        Args:
            entities: list, preprocessed entities of the sentence.
            context: collections.deque, queue containing the text of the sentences of the context.
        """

        if self.criterion_candidate():
            self.candidate = True
            self.entities = entities
            self.context = ' '.join([context.popleft() for _ in range(len(context))])

    # endregion

    # region Methods criterion_

    def criterion_candidate(self):
        """
        Check if an article is a suitable candidate.

        Returns:
            bool, True iff the article is a candidate.
        """

        score_token = self.token_similarity.score if self.token_similarity is not None else None
        score_entity = self.entity_similarity.score if self.entity_similarity is not None else None

        return True if (score_token and score_token >= self.token_threshold) \
                       or (score_entity and score_entity >= self.entity_threshold) else False

    # endregion

    # region Methods load_

    @classmethod
    @BaseClass.Verbose("Loading entity embeddings...")
    def load_embedding(cls):
        """ Load the entity embedding. """

        cls.embedding_entity = KeyedVectors.load_word2vec_format(
            fname='../pre_trained_models/freebase-vectors-skipgram1000-en.bin',
            binary=True
        )

    # endregion

    # region Other methods

    # TODO: repair
    def find_vocab_np(self):
        """
        Finds in the NP some entities from the vocabulary. Otherwise, returns None.

        Returns:
            list, entities which belong to the vocabulary of the entity embedding, or None.
        """

        tokens_groups = []
        separators = [-1]

        for idx in range(len(self.tokens)):
            if self.tokens[idx].text in [',', ';', 'and', 'or']:
                separators.append(idx)

        separators.append(len(self.tokens))

        for i in range(len(separators) - 1):
            start = separators[i] + 1
            end = separators[i + 1]

            tokens_groups.append(self.tokens[start: end])

        tokens_groups.append(self.tokens) if len(separators) > 2 else None

        entities = []

        for tokens in tokens_groups:

            plausible_entities = []

            for criterion_name in ['punctuation', 'possessive', 'stopwords', 'determiner', 'number', 'adjective']:
                criterion = getattr(Token, 'criterion_' + criterion_name)

                tokens = [token for token in tokens if not criterion(token)]
                plausible_entity = '_'.join([token.text.lower().replace('-', '_') for token in tokens])

                if plausible_entity and plausible_entity not in plausible_entities:
                    plausible_entities.append(plausible_entity)

            entity = None
            for e in plausible_entities:
                if '/en/' + e in self.embedding_entity.vocab:
                    entity = '/en/' + e
                    break

            entities.append(entity) if entity else None

        return entities if entities else None

    # TODO: repair
    def find_vocab_entities(self, entities):
        """
        Finds for each entity an entity from the vocabulary. Otherwise, returns None.

        Returns:
            list, entities which belongs to the vocabulary of the entity embedding, or None.
        """

        vocab_entities = []

        for entity in entities:

            entity = entity.lower().replace(' ', '_').replace('-', '_').replace('.', '').replace(',', '')
            plausible_entities = [entity] + entity.split('_')

            for e in plausible_entities:
                if '/en/' + e in self.embedding_entity.vocab:
                    vocab_entities.append('/en/' + e)
                    break

        return vocab_entities if vocab_entities else None

    # endregion


def main():
    from database_creation.sentence import Sentence
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    entities = ['James Joyce', 'Richard Bernstein']
    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    for sentence in sentences:
        sentence.compute_similarities(entities)

    Np.set_parameters(to_print=[], print_attribute=True)
    for sentence in sentences:
        print(Np.to_string(sentence.nps))


if __name__ == '__main__':
    main()
