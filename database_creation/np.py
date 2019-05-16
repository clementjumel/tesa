from database_creation.utils import BaseClass
from database_creation.word import Word

from gensim.models import KeyedVectors


class Np(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = [], True, 1, 4

    embedding_entity = None
    word_threshold, entity_threshold = 0.3, 0.3

    def __init__(self, np):
        """
        Initializes an instance of Np.

        Args:
            np: list, tuples (text, pos) that defines a word.
        """

        self.words = tuple([Word(text=word[0], pos=word[1]) for word in np])

        self.word_similarity = None
        self.entity_similarity = None

        self.word_similar_entities = None
        self.entity_similar_entities = None

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

        self.compute_word_similarity(entities)
        self.compute_entity_similarity(entities)

    def compute_word_similarity(self, entities):
        """
        Compute the word similarity of the NP to the entities in the article.

        Args:
            entities: list, preprocessed entities of the sentence.
        """

        similarities, similar_entities = [], []

        for word in self.words:
            word.compute_similarity(entities)

            similarities.append(word.similarity) if word.similarity is not None else None
            similar_entities.append((word.text, word.similar_entities)) if word.similarity is not None else None

        if similarities:
            self.word_similarity = max(similarities)
            self.word_similar_entities = \
                [similar_entities[i] for i in [idx for idx, val in enumerate(similarities)
                                               if val == self.word_similarity]]

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
            self.entity_similar_entities = [similar_entities[i] for i in [idx for idx, val in enumerate(similarities)
                                                                          if val == self.entity_similarity]]

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

        if (self.word_similarity and self.word_similarity >= self.word_threshold) \
                or (self.entity_similarity and self.entity_similarity >= self.entity_threshold):

            return True

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

    def find_vocab_np(self):
        """
        Finds in the NP some entities from the vocabulary. Otherwise, returns None.

        Returns:
            list, entities which belong to the vocabulary of the entity embedding, or None.
        """

        words_groups = []
        separators = [-1]

        for idx in range(len(self.words)):
            if self.words[idx].text in [',', ';', 'and', 'or']:
                separators.append(idx)

        separators.append(len(self.words))

        for i in range(len(separators) - 1):
            start = separators[i] + 1
            end = separators[i + 1]

            words_groups.append(self.words[start: end])

        words_groups.append(self.words) if len(separators) > 2 else None

        entities = []

        for words in words_groups:

            plausible_entities = []

            for criterion_name in ['punctuation', 'possessive', 'stopwords', 'determiner', 'number', 'adjective']:
                criterion = getattr(Word, 'criterion_' + criterion_name)

                words = [word for word in words if not criterion(word)]
                plausible_entity = '_'.join([word.text.lower().replace('-', '_') for word in words])

                if plausible_entity and plausible_entity not in plausible_entities:
                    plausible_entities.append(plausible_entity)

            entity = None
            for e in plausible_entities:
                if '/en/' + e in self.embedding_entity.vocab:
                    entity = '/en/' + e
                    break

            entities.append(entity) if entity else None

        return entities if entities else None

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

    nps = [
        Np([('A', 'DT'), ('nice', 'JJ'), ('city', 'NN')]),
        Np([('New', 'NNP'), ('York', 'NNP')]),
        Np([('The', 'DT'), ('New', 'NNP'), ('York', 'NNP'), ('Times', 'NNP')]),
        Np([('New', 'NNP'), ('York', 'NNP'), ('and', 'CC'), ('Chicago', 'NNP')]),
    ]

    entities = ['San Francisco', 'Philadelphia', 'town']

    for np in nps:
        np.compute_similarities(entities)

    print(Np.to_string(nps))

    return


if __name__ == '__main__':
    main()
