from database_creation.utils import BaseClass
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from xml.etree import ElementTree
from collections import deque, OrderedDict
from copy import copy
import re


class Article(BaseClass):
    # region Class initialization

    to_print, print_attribute, print_lines, print_offsets = ['title', 'preprocessed_entities', 'sentences'], True, 1, 0

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

        self.entities_locations = None
        self.entities_persons = None
        self.entities_organizations = None

        self.preprocessed_entities = None

        self.sentences = None
        self.coreferences = None

    # endregion

    # region Main methods

    def preprocess(self):
        """ Preprocess the article. """

        self.compute_title()
        self.compute_entities()
        self.compute_preprocessed_entities()
        self.compute_sentences()
        self.compute_coreferences()

    def process(self):
        """ Process the articles. """

        self.compute_similarities()
        self.compute_candidates()

    def write(self, f):
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

        tree = ElementTree.parse(self.original_path)
        root = tree.getroot()

        self.title = root.find('./head/title').text

    def compute_entities(self):
        """ Compute all the entities of an article. """

        tree = ElementTree.parse(self.original_path)
        root = tree.getroot()

        self.entities_locations = [entity.text for entity in root.findall('./head/docdata/identified-content/location')]
        self.entities_persons = [entity.text for entity in root.findall('./head/docdata/identified-content/person')]
        self.entities_organizations = [entity.text for entity in root.findall('./head/docdata/identified-content/org')]

    def compute_preprocessed_entities(self):
        """ Compute the preprocessed entities (consider the text in parenthesis as a separate entity). """

        preprocessed_entities = []

        for entity in self.entities_locations + self.entities_organizations:

            before = re.findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis
            in_parenthesis = re.findall(r'\((.*?)\)', entity)  # find the text in parenthesis (possibly several cases)

            if len(before) == 0:
                before = entity
            else:
                before = before[0]

            preprocessed_entities.append(before)
            preprocessed_entities.extend(in_parenthesis) if in_parenthesis else None

        for entity in self.entities_persons:

            before = re.findall(r'(.*?)\s*\(', entity)  # find the text before the parenthesis
            in_parenthesis = re.findall(r'\((.*?)\)', entity)  # find the text in parenthesis (possibly several cases)

            if len(before) == 0:
                before = entity
            else:
                before = before[0]

            if len(before.split(', ')) == 2:
                before = ' '.join([before.split(', ')[1], before.split(', ')[0]])

            for p in in_parenthesis:
                if p.replace('-', '').isdigit():
                    in_parenthesis.remove(p)

            preprocessed_entities.append(before)
            preprocessed_entities.extend(in_parenthesis) if in_parenthesis else None

        preprocessed_entities = list(OrderedDict.fromkeys(preprocessed_entities))  # remove duplicates (keep the order)

        self.preprocessed_entities = preprocessed_entities

    def compute_sentences(self):
        """ Compute the sentences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/sentences/sentence')

        self.sentences = [Sentence(element) for element in elements]

    def compute_coreferences(self):
        """ Compute the coreferences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/coreference/coreference')

        self.coreferences = [Coreference(element) for element in elements]

    def compute_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for sentence in self.sentences:
            sentence.compute_similarities(self.preprocessed_entities)

    def compute_candidates(self):
        """ Computes and fills the candidate NPs of the article. """

        context = deque([self.sentences[i].text if 0 <= i < len(self.sentences) else ''
                         for i in range(-Article.context_range, Article.context_range + 1)])

        for i in range(len(self.sentences)):

            self.sentences[i].compute_candidates(entities=self.preprocessed_entities, context=copy(context))

            context.popleft()
            context.append(self.sentences[i + Article.context_range + 1].text
                           if i + Article.context_range + 1 < len(self.sentences) else '')

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

    # endregion


def main():

    article = Article('../databases/nyt_jingyun/data/2000/01/01/1165027.xml',
                      '../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')

    article.preprocess()
    article.process()

    print(article)

    return


if __name__ == '__main__':
    main()
