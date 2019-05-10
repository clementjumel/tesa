from database_creation.utils import BaseClass
from database_creation.sentences import Sentence
from database_creation.coreferences import Coreference

from xml.etree import ElementTree


class Article(BaseClass):
    # region Class initialization

    to_print = ['title', 'entities_locations', 'entities_persons', 'entities_organizations', 'sentences',
                'coreferences']
    print_attribute = True
    print_lines = 1
    print_offsets = 0

    def __init__(self, article_id, original_path, annotated_path):
        """
        Initializes an instance of Article.

        Args:
            article_id: str, ID of the article.
            original_path: str, path of the article's original corpus' content.
            annotated_path: str, path of the article's annotated corpus' content.
        """

        self.article_id = str(article_id)
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

        self.get_title()
        self.get_entities()
        self.get_sentences()
        self.get_coreferences()

    def process(self):
        """ Process the articles. """

        self.get_similarities()
        self.get_candidates()

    def write_candidates(self, f):
        """
        Write the candidates of the articles in an opened file.

        Args:
            f: file, ready to be written in.
        """

        candidates = [np if np.candidate else None for sentence in self.sentences for np in sentence.nps]
        candidates = filter(None, candidates)

        f.write(self.to_string(candidates)) if candidates else None

    # endregion

    # region Methods get

    def get_title(self):
        """ Compute the title of an article. """

        tree = ElementTree.parse(self.original_path)
        root = tree.getroot()

        self.title = root.find('./head/title').text

    def get_entities(self):
        """ Compute all the entities of an article. """

        tree = ElementTree.parse(self.original_path)
        root = tree.getroot()

        self.entities_locations = [entity.text for entity in root.findall('./head/docdata/identified-content/location')]
        self.entities_persons = [entity.text for entity in root.findall('./head/docdata/identified-content/person')]
        self.entities_organizations = [entity.text for entity in root.findall('./head/docdata/identified-content/org')]

    def get_sentences(self):
        """ Compute the sentences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/sentences/sentence')

        self.sentences = [Sentence(element) for element in elements]

    def get_coreferences(self):
        """ Compute the coreferences of the article. """

        root = ElementTree.parse(self.annotated_path)
        elements = root.findall('./document/coreference/coreference')

        self.coreferences = [Coreference(element) for element in elements]

    def get_similarities(self):
        """ Compute the similarity of the NPs to the entities in the article. """

        for sentence in self.sentences:
            sentence.get_similarities(self.entities_locations, self.entities_persons, self.entities_organizations)

    def get_candidates(self):
        """ Compute and fills the candidate NPs of the article. """

        for i in range(len(self.sentences)):
            idxs = self.sentences[i].get_candidates_idx()

            if idxs:
                self.sentences[i].get_candidate_info(idxs,
                                                     self.entities_locations,
                                                     self.entities_persons,
                                                     self.entities_organizations,
                                                     self.get_context(i))

    # TODO
    def get_context(self, i):
        pass

    # endregion

    # region Methods criterion

    def to_clean(self):
        """ Check if an article has to be removed during the preprocessing (incomplete data). """

        return self.criterion_no_annotation()

    def criterion_no_annotation(self):
        """
        Criterion to check if an article has an annotation file.

        Returns:
            bool, True iff the article has no annotation file.
        """

        try:
            f = open(self.annotated_path, 'r')
            f.close()
            return False

        except FileNotFoundError:
            return True

    def criterion_no_entity(self):
        """
        Criterion to check if an article has some entity.

        Returns:
            bool, True iff the article has no entity.
        """

        return max([len(self.entities_locations), len(self.entities_persons), len(self.entities_organizations)]) == 0

    def criterion_few_entities(self):
        """
        Criterion to check if an article has several entities of the same type.

        Returns:
            bool, True iff the article has at least 2 entities of the same type.
        """

        return max([len(self.entities_locations), len(self.entities_persons), len(self.entities_organizations)]) <= 1

    # endregion


def main():
    from database_creation.databases import Database

    db = Database(year='2000', limit_articles=100)
    db.process()
    db.write_candidates('out')
    return


if __name__ == '__main__':
    main()
