from database_creation.token import Token


class Sentence:
    # region Class initialization

    def __init__(self, sentence_element):
        """
        Initializes an instance of Sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        self.tokens = None

        self.compute_tokens(sentence_element)

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Sentence.

        Returns:
            str, readable format of the instance.
        """

        s, start = '', True

        for _, token in self.tokens.items():
            if start:  # Beginning of the sentence
                if token.criterion_punctuation():
                    s += str(token)
                else:
                    s += str(token)[0].capitalize() + str(token)[1:]
                    start = False

            else:
                if not token.criterion_punctuation():
                    s += ' '

                s += str(token)

        return s

    # endregion

    # region Methods compute_

    def compute_tokens(self, sentence_element):
        """
        Compute the tokens of the sentence.

        Args:
            sentence_element: ElementTree.Element, annotations of the sentence.
        """

        tokens = {}

        for token_element in sentence_element.findall('./tokens/token'):
            tokens[int(token_element.attrib['id'])] = Token(token_element)

        self.tokens = tokens

    # endregion


def main():
    from xml.etree import ElementTree

    tree = ElementTree.parse('../databases/nyt_jingyun/content_annotated/2006content_annotated/1728670.txt.xml')
    root = tree.getroot()

    sentences = [Sentence(sentence_element) for sentence_element in root.findall('./document/sentences/sentence')[:3]]

    for sentence in sentences:
        print(str(sentence))

    return


if __name__ == '__main__':
    main()
