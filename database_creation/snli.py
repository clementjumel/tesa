import jsonlines
import random as rd


def get_keys(file_name):
    """
    Return the list of keys in the jsonl file
    :param file_name: str, name of the file
    :return: list, keys
    """
    with jsonlines.open(file_name) as reader:
        for sentence in reader:
            return list(sentence.keys())


def get_entailment_degree(sentence):
    """
    Return the entailment degree (number of entailment judgements) of a sentence
    :param sentence: dict, sentence to analyze
    :return: int, entailment degree
    """
    return len([1 for x in sentence['annotator_labels'] if x == 'entailment'])


def get_entailed_sentences(file_name, min_entailment_degree, max_entailment_degree):
    """
    Return a list of entailed sentences from file_nme; entailed min number of entailed judgements above
    min_entailment_degree.
    :param file_name: str, origin file name
    :param min_entailment_degree: int, min degree of entailment
    :param max_entailment_degree: int, max degree of entailment
    :return: list, entailed sentences
    """
    res = []
    cmpt_entailed, cmpt_all = 0, 0

    with jsonlines.open(file_name) as reader:
        for sentence in reader:
            cmpt_all += 1
            entailment_degree = get_entailment_degree(sentence)

            if min_entailment_degree <= entailment_degree <= max_entailment_degree:
                cmpt_entailed += 1
                res.extend([sentence])

    print("Number of remaining sentences (entailment): {}/{} ({}%)".format(cmpt_entailed, cmpt_all,
                                                                           round(100 * cmpt_entailed / cmpt_all)))
    return res


def remove_same_first_phrase(sentences):
    """
    Remove the sentences where the first phrases are the same
    :param sentences: list, sentences to sort
    :return: list, sorted sentences
    """
    res = []
    for sentence in sentences:
        if not is_same_first_phrase(sentence):
            res.extend([sentence])

    len_1, len_2 = len(res), len(sentences)
    print(
        "Number of remaining sentences (same first phrase): {}/{} ({}%)".format(len_1, len_2,
                                                                                round(100 * len_1 / len_2)))
    return res


def remove_to_exclude(sentences, to_exclude):
    """
    Remove from the sentences the examples we want to avoid
    :param sentences: list, sentences to sort
    :param to_exclude: list, string that we want to avoid
    :return: list, sorted sentences
    """
    res = []
    for sentence in sentences:
        if not is_to_exclude(sentence, to_exclude):
            res.extend([sentence])

    len_1, len_2 = len(res), len(sentences)
    print("Number of remaining sentences (exclusion): {}/{} ({}%)".format(len_1, len_2, round(100 * len_1 / len_2)))
    return res


def display_sentence(sentence, display_original_sentence=True, display_entailment_degree=False, display_parse=False,
                     display_first_phrase=False):
    """
    Display the elements of a sentence
    :param sentence: dict, object to display
    :param display_original_sentence: bool, option of displaying the original sentence
    :param display_entailment_degree: bool, option of displaying entailment degree
    :param display_parse: bool, option of displaying the parsing of the sentences
    :param display_first_phrase: bool, option of displaying the first phrase
    :return: /
    """
    if display_entailment_degree:
        print("Entailment degree: {}/{}".format(get_entailment_degree(sentence), 5))
    if display_original_sentence:
        print(sentence['sentence1'])
    if display_parse:
        print(sentence['sentence1_parse'])
    if display_first_phrase:
        print(get_phrase_from_branch(get_first_branch(sentence['sentence1_parse'])))
    if display_original_sentence:
        print(sentence['sentence2'])
    if display_parse:
        print(sentence['sentence2_parse'])
    if display_first_phrase:
        print(get_phrase_from_branch(get_first_branch(sentence['sentence2_parse'])))
    print('')


def display_examples(entailed_sentences, N_examples, seed=None,
                     display_original_sentence=True, display_entailment_degree=False, display_parse=False,
                     display_first_phrase=False):
    """
    Display some examples of sentences, selected randomly.
    :param entailed_sentences: list, sentences to sample
    :param N_examples: int, number of examples to display
    :param seed: int, optional, seed of the random generator
    :param display_original_sentence: bool, optional, option of displaying the original sentences
    :param display_entailment_degree: bool, optional, option of displaying the entailment degree
    :param display_parse: bool, optional, option of displaying the parsing of the sentence
    :parma display_first_prhase: bool, optional, option of displaying the first phrase
    :return: /
    """
    rd.seed(a=seed)
    examples = rd.sample(entailed_sentences, N_examples)
    for sentence in examples:
        display_sentence(sentence, display_original_sentence, display_entailment_degree, display_parse,
                         display_first_phrase)


def get_first_branch(sentence_parse):
    """
    Return the first branch of a parse tree
    :param sentence_parse: str, parsing of a sentence
    :return: str, piece of the parsing corresponding to the fist branch
    """
    res = ''
    cmpt = 0
    bool = False

    for character in sentence_parse[7:-1]:
        res += character
        if character == '(':
            cmpt += 1
            bool = True
        elif character == ')':
            cmpt -= 1
            if cmpt == 0 & bool:
                break
    return res


def get_phrase_from_branch(branch):
    """
    Return the phrase extracted from a parse branch
    :param branch: str, branch to analyze
    :return: str, resulting phrase
    """
    words, items = [], branch.split(' ')
    for item in items:
        if item[-1] == ')':
            words.extend([item.replace(')', '')])
    return ' '.join(words)


def get_first_phrase(sentence_parse):
    """
    Return the first phrase of a sentence from its parsing
    :param sentence_parse: str, parsing of the sentence
    :return: str, first phrase
    """
    return get_phrase_from_branch(get_first_branch(sentence_parse))


def is_same_first_phrase(sentence):
    """
    Return a boolean indicating whether the first phrases of the sentences are the same
    :param sentence: dict, sentence to analyze
    :return: bool, True if the first phrases are the same
    """
    return get_first_phrase(sentence['sentence1_parse']).lower() \
           == get_first_phrase(sentence['sentence2_parse']).lower()


def is_to_exclude(sentence, to_exclude):
    """
    Return a boolean indicating if the sentence must be filtered out
    :param sentence: dictionnary, sentence to analyze
    :param to_exclude: list, phrases we want to avoid
    :return:
    """
    if (get_first_phrase(sentence['sentence1_parse']).lower() in to_exclude) \
            or (get_first_phrase(sentence['sentence2_parse']).lower() in to_exclude):
        return True
    else:
        return False


if __name__ == '__main__':
    # Parameters
    jsonl_file = 'original_database/snli_1.0/snli_1.0_test.jsonl'
    min_entailment_degree = 1
    max_entailment_degree = 5
    to_exclude = ['there']
    N_examples = 20
    # seed = 3
    seed = 6

    # Display options
    display_original_sentence = True
    display_entailment_degree = False
    display_parse = False
    display_first_phrase = False

    # Check the keys of the dictionnaries
    # print("Keys: {} \n".format(get_keys(jsonl_file)))

    # Sort sentences
    sentences = get_entailed_sentences(jsonl_file, min_entailment_degree, max_entailment_degree)
    sentences = remove_same_first_phrase(sentences)
    sentences = remove_to_exclude(sentences, to_exclude)
    print('')

    # Display examples
    display_examples(sentences, N_examples, seed,
                     display_original_sentence, display_entailment_degree, display_parse, display_first_phrase)
