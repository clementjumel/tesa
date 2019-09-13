from database_creation.utils import Context
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from copy import deepcopy
from collections import defaultdict


class Text:
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

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Text.

        Returns:
            str, readable format of the instance.
        """

        return ' '.join([str(sentence) for _, sentence in self.sentences.items()])

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

    def contexts_neigh_sent(self, tuple_, type_):
        """
        Returns the neighboring-sentences Contexts for a Tuple (neighboring sentences where the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: str, type of the Text used for the Context.

        Returns:
            dict, Contexts mapped with the indexes of the first and last sentences separated by '_'.
        """

        contexts, context_length = {}, len(tuple_.entities)

        # Mapping between the name of the entities and the indexes of the mentions' sentences
        entities_sentences = dict([(str(entity), self.get_entity_sentences(entity)) for entity in tuple_.entities])
        all_entities_sentences = set([idx for _, sentences in entities_sentences.items() for idx in sentences])

        for start_idx in all_entities_sentences:
            seen = set()

            for idx in range(start_idx, start_idx + context_length):
                seen.update([name for name, sentences in entities_sentences.items() if idx in sentences])

                if len(seen) == context_length:
                    sentences = {i: deepcopy(self.sentences[i]) for i in range(start_idx, idx + 1)}
                    self.highlight(sentences=sentences, tuple_=tuple_)

                    id_ = str(start_idx) + '_' + str(idx)
                    contexts[id_] = Context(sentences=sentences, type_=type_)
                    break

        return contexts

    def contexts_all_sent(self, tuple_, type_):
        """
        Returns the all-sentences Contexts for a Tuple (all the sentences if all the entities are mentioned).

        Args:
            tuple_: Tuple, entities to analyse.
            type_: type of the Text used for the Context.

        Returns:
            dict, Contexts mapped with the index '0'.
        """

        contexts, context_length = {}, len(tuple_.entities)

        for entity in tuple_.entities:
            if not self.get_entity_sentences(entity):
                return contexts

        sentences = deepcopy(self.sentences)
        self.highlight(sentences=sentences, tuple_=tuple_)

        contexts['0'] = Context(sentences=sentences, type_=type_)

        return contexts

    # endregion

    # region Other methods

    def highlight(self, sentences, tuple_):
        """
        Highlight (put in bold) the mentions of the entities of the Tuple in the sentences.

        Args:
            sentences: dict, sentences to highlight.
            tuple_: Tuple, entities to analyse.
        """

        entities_boundaries = defaultdict(set)

        idx = 0
        entity_to_color = dict()
        for i in range(len(tuple_.entities)):
            entity = tuple_.entities[i]
            entity_to_color[entity.name] = 'color' + str(i)

        for coreference in self.coreferences:
            if coreference.entity and coreference.entity in tuple_.get_name():
                entity = [entity for entity in tuple_.entities if entity.name == coreference.entity][0]
                for mention in [coreference.representative] + coreference.mentions:
                    if mention.sentence in sentences:
                        entities_boundaries[mention.sentence].add((entity.name, mention.start, mention.end))

        for sentence_id, sentence in sentences.items():
            boundaries = entities_boundaries[sentence_id]
            to_remove = set()

            for boundary1 in boundaries:
                for boundary2 in boundaries:
                    if boundary1[0] == boundary2[0] and \
                            ((boundary1[1] > boundary2[1] and boundary1[2] <= boundary2[2])
                             or (boundary1[1] >= boundary2[1] and boundary1[2] < boundary2[2])):
                        to_remove.add(boundary2)

            for boundary in to_remove:
                boundaries.remove(boundary)

            for name, start_idx, end_idx in boundaries:
                color = entity_to_color[name]
                start_tag = '<div class="popup" onclick="pop(' + str(idx) + ')"><' + color + '>'
                end_tag = '</' + color + '><span class="popuptext" id="' + str(idx) + '">' + name + '</span></div>'

                if sentence.tokens[start_idx].start_tag is None:
                    sentence.tokens[start_idx].start_tag = start_tag
                else:
                    sentence.tokens[start_idx].start_tag += start_tag

                if sentence.tokens[end_idx - 1].end_tag is None:
                    sentence.tokens[end_idx - 1].end_tag = end_tag
                else:
                    sentence.tokens[end_idx - 1].end_tag += end_tag

                idx += 1

            sentence.compute_text()

    # endregion


def main():
    from database_creation.article import Article

    article = Article('../databases/nyt_jingyun/data/2006/01/01/1728670.xml',
                      '../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml',
                      '../databases/nyt_jingyun/summary_annotated/2006summary_annotated/1728670.txt.xml')

    article.compute_entities()
    article.compute_annotations()

    print(article.content)
    print(article.summary)

    return


if __name__ == '__main__':
    main()
