from database_creation.utils import BaseClass


class Np(BaseClass):
    # region Class initialization

    to_print = ['tokens']
    print_attribute, print_lines, print_offsets = False, 1, 4

    def __init__(self, tokens):
        """
        Initializes an instance of Np.

        Args:
            tokens: dict, Tokens that define the Noun Phrase.
        """

        self.tokens = tokens

    # endregion


def main():
    from database_creation.sentence import Sentence
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2000content_annotated/1165027.txt.xml')
    root = tree.getroot()

    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    for sentence in sentences:
        print(Np.to_string(sentence.nps))
    return


if __name__ == '__main__':
    main()
