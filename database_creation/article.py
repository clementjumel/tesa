from database_creation.utils import Entity
from database_creation.text import Text

from xml.etree import ElementTree


class Article:
    # region Class initialization

    def __init__(self, data_path, content_path, summary_path):
        """
        Initializes an instance of Article.

        Args:
            data_path: str, path of the article's data.
            content_path: str, path of the article's annotated content.
            summary_path: str, path of the article's annotated summary.
        """

        self.data_path = data_path
        self.content_path = content_path
        self.summary_path = summary_path

        self.title = None
        self.date = None
        self.entities = None
        self.content = None
        self.summary = None
        self.contexts = None

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Article.

        Returns:
            str, readable format of the instance.
        """

        return '\n'.join([self.title, str(self.summary), str(self.content)])

    # endregion

    # region Methods compute_

    def compute_metadata(self):
        """ Compute the metadata (title, date) of the article. """

        root = ElementTree.parse(self.data_path).getroot()

        title = self.get_title(root)
        date = self.get_date(root)

        self.title = title
        self.date = date

    def compute_annotations(self):
        """ Compute the annotated texts (content or summary) of the article. """

        content_root = ElementTree.parse(self.content_path).getroot()
        summary_root = ElementTree.parse(self.summary_path).getroot()

        self.content = Text(root=content_root, entities=self.entities)
        self.summary = Text(root=summary_root, entities=self.entities)

    def compute_contexts(self, tuple_):
        """
        Compute the contexts of the article for the Tuple of entities, according to the specified context types.

        Args:
            tuple_: Tuple, tuple of Entities mentioned in the article.
        """

        name = tuple_.get_name()
        self.contexts = self.contexts or dict()

        contexts = dict()

        contexts.update(self.content.contexts_neigh_sent(tuple_=tuple_, type_='content'))
        contexts.update(self.summary.contexts_all_sent(tuple_=tuple_, type_='summary'))

        self.contexts[name] = contexts

    def compute_entities(self):
        """ Compute the entities of the article. """

        entities = dict()

        for entity in self.get_entities():
            if entity.name in entities:
                entities[entity.name].update_info(entity)
            else:
                entities[entity.name] = entity

        self.entities = {entities[name] for name in {entity.name for entity in self.get_entities()}}

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

        element = root.find('./head/title')

        return element.text if element is not None else 'No title.'

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

    def get_entities(self):
        """
        Returns the Entities of the article given the tree of its metadata. However it can return several times the
        same entities (which is not an issue in the whole pipeline).

        Returns:
            list, Entities of the article.
        """

        root = ElementTree.parse(self.data_path).getroot()

        locations = [Entity(entity.text, 'location')
                     for entity in root.findall('./head/docdata/identified-content/location')]
        persons = [Entity(entity.text, 'person') for entity in root.findall('./head/docdata/identified-content/person')]
        orgs = [Entity(entity.text, 'org') for entity in root.findall('./head/docdata/identified-content/org')]

        entities = locations + persons + orgs

        return entities

    # endregion

    # region Methods criterion_

    def criterion_content(self):
        """
        Check if an article's content file exists.

        Returns:
            bool, True iff the file doesn't exist and must be deleted.
        """

        try:
            f = open(self.content_path, 'r')
            f.close()
            return False

        except FileNotFoundError:
            return True

    def criterion_summary(self):
        """
        Check if an article's summary file exists.

        Returns:
            bool, True iff the file doesn't exist and must be deleted.
        """

        try:
            f = open(self.summary_path, 'r')
            f.close()
            return False

        except FileNotFoundError:
            return True

    # endregion

    # region Methods debug_

    @staticmethod
    def debug_articles(id_):
        """
        Returns a string showing the debugging of an article.

        Args:
            id_: str, id_ of the article.

        Returns:
            str, debugging of the article.
        """

        return id_ + ':'

    def debug_metadata(self, id_):
        """
        Returns a string showing the debugging of an article.

        Args:
            id_: str, id_ of the article.

        Returns:
            str, debugging of the article.
        """

        return id_ + ': ' + self.date + '; ' + self.title

    def debug_entities(self, id_):
        """
        Returns a string showing the debugging of an article.

        Args:
            id_: str, id_ of the article.

        Returns:
            str, debugging of the article.
        """

        s = id_ + ': '
        s += ', '.join([str(entity) for entity in self.get_entities()])
        s += ' -> ' + ', '.join([str(entity) for entity in self.entities])

        return s

    def debug_annotations(self, id_):
        """
        Returns a string showing the debugging of an article.

        Args:
            id_: str, id_ of the article.

        Returns:
            str, debugging of the article.
        """

        s = id_ + ': '
        s += '; '.join([str(coreference) for coreference in self.content.coreferences + self.summary.coreferences])

        return s

    def debug_contexts(self, id_):
        """
        Returns a string showing the debugging of an article.

        Args:
            id_: str, id_ of the article.

        Returns:
            str, debugging of the article.
        """

        s = id_ + ': '
        s += '; '.join([name + ': ' + str(context) for name, context in self.contexts.items()])

        return s

    # endregion


def main():
    from database_creation.utils import Tuple

    article = Article('../databases/nyt_jingyun/data/2006/01/01/1728670.xml',
                      '../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml',
                      '../databases/nyt_jingyun/summary_annotated/2006summary_annotated/1728670.txt.xml')

    article.compute_entities()
    article.compute_metadata()
    article.compute_annotations()
    article.compute_contexts(tuple_=Tuple(id_='0', entities=tuple(article.entities)))

    print(article)

    return


if __name__ == '__main__':
    main()
