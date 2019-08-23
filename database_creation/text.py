from database_creation.utils import BaseClass, Context
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from collections import defaultdict
from copy import deepcopy


class Text(BaseClass):
    # region Class initialization

    def __init__(self, root, entities):
        """
        Initializes an instance of Text (content or summary).

        Args:
            root: ElementTree.root, root of the annotations of the Text.
            entities: set, Entities of the article.
        """

        self.sentences = self.get_sentences(root)
        self.coreferences = self.get_coreferences(root, entities)

    # endregion

    # region Methods get_

    @staticmethod
    def get_sentences(root):
        """
        Returns the Sentences of a Text given its tree.

        Args:
            root: ElementTree.root, root of the annotations of the Text.

        Returns:
            dict, Sentences of the article (mapped with their indexes).
        """

        elements = root.findall('./document/sentences/sentence')
        sentences = {int(element.attrib['id']): Sentence(element) for element in elements}

        return sentences

    @staticmethod
    def get_coreferences(root, entities):
        """
        Returns the Coreferences of a Text given its tree.

        Args:
            root: ElementTree.root, root of the annotations of the Text.
            entities: set, Entities of the article.

        Returns:
            list, Coreferences of the article.
        """

        elements = root.findall('./document/coreference/coreference')
        coreferences = [Coreference(element, entities) for element in elements]

        return coreferences

    def get_entity_sentences(self, entity):
        """
        Returns the indexes of the sentences where there is a mention of the specified entity.

        Args:
            entity: Entity, entity we want to find mentions of.

        Returns:
            list, sorted list of sentences' indexes.
        """

        entity_sentences = set()

        for coreference in self.coreferences:
            if coreference.entity and entity.match(coreference.entity):
                entity_sentences.update(coreference.sentences)

        return sorted(entity_sentences)

    # endregion

    # region Methods contexts_

    # TODO: change
    def contexts_neigh_sent(self, tuple_, type_):
        """
        Returns the neighboring-sentences contexts for a Tuple (neighboring sentences where the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: str, type of the Context.

        Returns:
            dict, neighbouring-sentences Contexts of the entities, mapped with their sentences span (indexes of the
            first and last sentences separated by '_').
        """

        sentences_entities = defaultdict(set)

        for i in range(len(tuple_.entities)):
            for idx in self.get_entity_sentences(tuple_.entities[i]):
                sentences_entities[idx].add(i)

        contexts_sentences = set()

        for idx in sentences_entities:
            unseens = list(range(len(tuple_.entities)))
            seers = set()

            for i in range(len(tuple_.entities)):
                if idx + i in sentences_entities:
                    for j in sentences_entities[idx + i]:
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
            entity_coreferences = {}
            for idx in idxs:
                correspondences = []

                for entity in tuple_.entities:
                    correspondence = [coreference for coreference in self.coreferences if idx in coreference.sentences
                                      and coreference.entity and coreference.entity == entity.name]

                    correspondences.append(tuple([entity, correspondence]))

                entity_coreferences[idx] = correspondences

            id_ = str(idxs[0]) + '_' + str(idxs[-1])
            contexts[id_] = Context(sentences={idx: deepcopy(self.sentences[idx]) for idx in idxs},
                                    entity_coreferences=entity_coreferences,
                                    type_=type_)

        return contexts

    # TODO: change
    def contexts_all_sent(self, tuple_, type_):
        """
        Returns the all-sentences contexts for a Tuple (neighboring sentences where the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: str, type of the Context.

        Returns:
            dict, all-sentences Contexts of the entities, mapped with '0'.
        """

        for entity in tuple_.entities:
            if not self.get_entity_sentences(entity):
                return dict()

        contexts_sentences = {tuple(range(list(self.sentences.keys())[0], list(self.sentences.keys())[-1] + 1))}
        contexts = dict()

        for idxs in contexts_sentences:
            entity_coreferences = {}
            for idx in idxs:
                correspondences = []

                for entity in tuple_.entities:
                    correspondence = [coreference for coreference in self.coreferences if
                                      idx in coreference.sentences
                                      and coreference.entity and coreference.entity == entity.name]

                    correspondences.append(tuple([entity, correspondence]))

                entity_coreferences[idx] = correspondences

            id_ = '0'
            contexts[id_] = Context(sentences={idx: deepcopy(self.sentences[idx]) for idx in idxs},
                                    entity_coreferences=entity_coreferences,
                                    type_=type_)

        return contexts

    # endregion
