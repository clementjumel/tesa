from database_creation.utils import BaseClass
from database_creation.sentence import Sentence
from database_creation.coreference import Coreference

from xml.etree import ElementTree
from collections import deque


class Article(BaseClass):
    # region Class initialization

    to_print = ['title', 'entities_locations', 'entities_persons', 'entities_organizations', 'sentences',
                'coreferences']
    print_attribute, print_lines, print_offsets = True, 1, 0

    context_range = 0

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

        self.sentences = None
        self.coreferences = None

    # endregion

    # region Main methods

    def preprocess(self):
        """ Preprocess the article. """

        self.compute_title()
        self.compute_entities()
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

        candidates = [np if np.candidate else None for sentence in self.sentences for np in sentence.nps]
        candidates = filter(None, candidates)

        f.write(self.to_string(candidates)) if candidates else None

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

    def compute_sentences(self):
        """ Compute the sentences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/sentences/sentence')

        self.sentences = tuple([Sentence(element) for element in elements])

    def compute_coreferences(self):
        """ Compute the coreferences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/coreference/coreference')

        self.coreferences = tuple([Coreference(element) for element in elements])

    def compute_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for sentence in self.sentences:
            sentence.compute_similarities(self.entities_locations, self.entities_persons, self.entities_organizations)

    def compute_candidates(self):
        """ Computes and fills the candidate NPs of the article. """

        context = deque([self.sentences[i].text if 0 <= i < len(self.sentences) else None
                         for i in range(-Article.context_range, Article.context_range + 1)])

        for i in range(len(self.sentences)):
            self.sentences[i].compute_candidates(entities_locations=self.entities_locations,
                                                 entities_persons=self.entities_persons,
                                                 entities_organizations=self.entities_organizations,
                                                 context=context)

            context.popleft()
            context.append(self.sentences[i + Article.context_range + 1]
                           if i + Article.context_range + 1 < len(self.sentences) else None)

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
        Check if an article has some entity.

        Returns:
            bool, True iff the article has no entity.
        """

        return max([len(self.entities_locations), len(self.entities_persons), len(self.entities_organizations)]) == 0

    def criterion_similarity(self):
        # TODO
        pass

    # endregion


def main():
    from database_creation.database import Database

    db = Database(max_size=100)

    db.preprocess()
    db.process()

    return


if __name__ == '__main__':
    main()
