from database_creation.utils import to_string

import numpy as np
import glob
import inflect
from collections import OrderedDict
import xml.etree.ElementTree as ET
import nltk.data
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import spacy

# Initialization of paths, xpaths and variables
db_root = '../databases/nyt_jingyun'
results_path = 'results/'
paths = {
    'content_raw': db_root + '/content/2000content/',
    'content_annotated': db_root + '/content_annotated/2000content_annotated/',
    'content_original': db_root + '/data/2000/',
    'summary_raw': db_root + '/summary/2000summary/',
    'summary_annotated': db_root + '/summary_annotated/2000summary_annotated/',
}
xpaths = {
    'persons': './head/docdata/identified-content/person',
    'locations': './head/docdata/identified-content/location',
    'organizations': './head/docdata/identified-content/org',
    'parse': './document/sentences/sentence/parse',
    'coreferences': './document/coreference',
    'title': './head/title',
}
entities_label = ['persons', 'locations', 'organizations']
stopwords = set(stopwords.words('english'))
stopwords.update(['us', 'more'])
to_exclude = ['united states']
not_plural = ['Texas']
numeric_articles = ['two', 'both', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'dozen', 'dozens']
time_words = ['day', 'month', 'year']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
brown_ic = wordnet_ic.ic('ic-brown.dat')
inflect = inflect.engine()
nlp = spacy.load('en_core_web_lg')


def get_file_paths(limit=None):
    """
    Compute the file paths of the original content of the articles.

    Args:
        limit: int, maximum number of files to compute

    Returns:
        list, file paths of the articles
    """

    file_paths = glob.glob(paths['content_original'] + '*/*/*.xml')
    file_paths.sort()

    if limit:
        print("Computing paths of {} files\n".format(limit))
        return file_paths[:limit]

    else:
        print("Computing paths of {} files\n".format(len(file_paths)))
        return file_paths


def get_article(text_type, file_id):
    """
    Compute some data of an article (content or summary, raw or annotated).

    Args:
        text_type: str, text type to compute ((content or summary)_((raw or annotated))
        file_id: int, id corresponding to the article

    Returns:
        str, content of the data (or None if not found)
    """

    if text_type.split('_')[1] == 'raw':
        path = paths[text_type] + str(file_id) + '.txt'

        try:
            return open(path, "r").read()

        except FileNotFoundError:
            return None

    elif text_type.split('_')[1] == 'annotated':
        path = paths[text_type] + str(file_id) + '.txt.xml'

        try:
            return open(path, "r").read()

        except FileNotFoundError:
            return None

    else:
        raise ValueError("Wrong text type.")


def is_to_exclude(phrase):
    """
    Check if a phrase is to be excluded (that is, when lowered and stopwords are removed, is in to_exclude).

    Args:
        phrase: list, phrase to check

    Returns:
        bool, True iff is to be excluded
    """

    phrase_text = ' '.join([word for word in phrase[0].lower().split(' ') if word not in stopwords])

    return phrase_text in to_exclude


def is_numeric_text(phrase):
    """
    Check if a phrase contains a numeric article (which is not a stopword), based on the text of the phrase.

    Args:
        phrase: list, phrase to check

    Returns:
        bool, True iff phrase contains a numeric article which is not a stopword
    """

    for word in phrase[0].lower().split(' '):
        if (word in numeric_articles) and (word not in stopwords):
            return True

    return False


def is_numeric_parse(phrase):
    """
    Check if a phrase contains a numeric article, based on the parsing of the phrase.

    Args:
        phrase: list, phrase to check

    Returns:
        bool, True iff phrase contains a numeric article
    """

    for word in phrase[1].split(' '):
        if word == 'CD':
            return True

    return False


def is_plural_text(phrase):
    """
    Check if a phrase contains a plural (not a stopword nor a time word) based on the text of the phrase. If there is
    an ', deal only with the first part.

    Args:
        phrase: list, phrase to check

    Returns:
        bool, True iff phrase contains a plural which is not a stop/time word
    """

    for word in phrase[0].lower().split(' '):
        word = word.split("'")[0]
        if word and (word not in stopwords) and (word not in not_plural):
            singular = inflect.singular_noun(word)
            if singular and (singular not in time_words):
                return True

    return False


def is_entity(phrase, entities):
    """
    Check if a phrase belongs to the entity (compare the strings lowered without the stopwords).

    Args:
        phrase: list, phrase to analyze
        entities: list, entities to compare the phrase to

    Returns:
        bool, True iff phrase (minus the stopwords) is in the entities
    """

    phrase_text = ' '.join([word for word in phrase[0].lower().split(' ') if word not in stopwords])

    for entity in entities:
        if phrase_text == ' '.join([word for word in entity.lower().split(' ') if word not in stopwords]):
            return True

    return False


def initialize_articles(file_paths, display_count=10000):
    """
    Compute a dictionary of articles, indexed by their ids, with the content_original_path of each article.

    Args:
        file_paths: list, paths of the original content of the articles

    Returns:
        dict, mapping between the ids (int) and the contents (containing content_original_path)
    """

    print("Initializing the articles...")
    count_files = 0
    articles = {}

    for file_path in file_paths:
        count_files += 1
        if count_files % display_count == 0:
            print("File {}/{}...".format(count_files, len(file_paths)))

        file_id = int(file_path.split('/')[-1].split('.')[0])
        articles[file_id] = {'content_original_path': file_path}

    print("Articles initialized ({} articles).\n".format(len(articles)))

    return articles


def filter_articles(articles, filter_entity_types, n_entities, display_count=10000):
    """
    Filter out articles that have less that n_entities in all of the specified entity types

    Args:
        articles: dict, articles to filter
        filter_entity_types: list, entity types to check
        n_entities: int, min number of entities

    Returns:
        dict, updated articles
    """

    print("Filtering articles ({})...".format(' '.join([str(n_entities), ', '.join(filter_entity_types)])))

    entity_xpath = [xpaths[entity] for entity in filter_entity_types]
    count_files, to_pop = 0, []

    for file_id in articles:

        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        tree = ET.parse(articles[file_id]['content_original_path'])
        root = tree.getroot()

        if max([len(root.findall(xpath)) for xpath in entity_xpath]) < n_entities:
            to_pop.append(file_id)

    for file_id in to_pop:
        articles.pop(file_id)

    if to_pop:
        print("{} articles filtered out.".format(len(to_pop)))
    print("Articles filtered ({} articles).\n".format(len(articles)))

    return articles


def compute_entities(articles, compute_entity_types, display_count=10000):
    """
    Compute the entities of entity_types in an articles. Add 'locations' or 'organizations' if relevant.

    Args:
        articles: dict, articles to fill
        compute_entity_types: list of str, types of entities to compute (in 'locations, 'persons' and 'organizations')

    Returns:
        dict, updated articles
    """

    print("Computing entities({})...".format(' '.join(compute_entity_types)))

    entity_xpaths = [xpaths[entity_type] for entity_type in compute_entity_types]
    count_files, to_pop = 0, []

    for file_id in articles:

        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        tree = ET.parse(articles[file_id]['content_original_path'])
        root = tree.getroot()

        entities = [entity.text for xpath in entity_xpaths for entity in root.findall(xpath)]

        if 'persons' in compute_entity_types:
            entities.append('persons')
        if 'organizations' in compute_entity_types:
            entities.append('organizations')

        if entities:
            articles[file_id]['entities'] = entities
        else:
            to_pop.append(file_id)

    for file_id in to_pop:
        articles.pop(file_id)

    if to_pop:
        print("{} entities don't exist.".format(len(to_pop)))
    print("Entities computed ({} articles).\n".format(len(articles)))

    return articles


def compute_titles(articles, display_count=10000):
    """
    Compute the titles of the articles.

    Args:
        articles: dict, articles to fill

    Returns:
        dict, updated articles
    """

    print("Computing titles...")

    xpath = xpaths['title']
    count_files, to_pop = 0, []

    for file_id in articles:

        count_files += 1
        if count_files % display_count == 0:
            print("File {}/{}...".format(count_files, len(articles)))

        tree = ET.parse(articles[file_id]['content_original_path'])
        root = tree.getroot()
        title = root.find(xpath).text

        if title:
            articles[file_id]['title'] = title
        else:
            to_pop.append(file_id)

    for file_id in to_pop:
        articles.pop(file_id)

    if to_pop:
        print("{} titles don't exist.".format(len(to_pop)))
    print("Titles computed ({} articles).\n".format(len(articles)))

    return articles


def compute_text_type(articles, text_type, display_count=10000):
    """
    Compute the data (according to text_type) of the articles.

    Args:
        articles: dict, articles to fill
        text_type: str, text type to compute ((content or summary)_((raw or annotated))

    Returns:
        dict, updated articles
    """

    print("Computing articles' data ({})...".format(' '.join(text_type.split('_'))))

    count_files, to_pop = 0, []

    for file_id in articles:

        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        data = get_article(text_type, file_id)

        if data:
            articles[file_id][text_type] = data
        else:
            to_pop.append(file_id)

    for file_id in to_pop:
        articles.pop(file_id)

    if to_pop:
        print("{} requested data don't exist.".format(len(to_pop)))
    print("Articles' data computed ({} articles).\n".format(len(articles)))

    return articles


# Old
def compute_coreferences(articles, display_count=10000):
    """
    Compute the coreferences of each article from content_annotated.

    Args:
        articles: dict, articles to fill

    Returns:
        dict, updated articles
    """

    print("Computing coreferences...")

    xpath = xpaths['coreferences']
    count_files = 0

    for file_id in articles:
        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        try:
            root = ET.fromstring(articles[file_id]['content_annotated'])
            coreferences = []

            for coreference_root in root.findall(xpath):
                for coreference_chain in coreference_root:
                    if not coreference_chain[0].get('representative'):
                        raise ValueError("Representative not in first position")

                    chain = []
                    for mention in coreference_chain:
                        chain.append(mention.find('text').text)

                    chain = list(dict.fromkeys([coreference for coreference in chain if
                                                coreference.lower() not in stopwords]))
                    if chain and len(chain) > 1:
                        coreferences.append(chain)

            if coreferences:
                articles[file_id]['coreferences'] = coreferences

        except ET.ParseError:
            pass

    print("Coreferences computed.\n")

    return articles


def compute_nps(articles, text_type, display_count=10000):
    """
    Compute the nps of the articles, from the parses of text_type.

    Args:
        articles: dict, articles to fill
        text_type: str, content_annotated or summary_annotated

    Returns:
        dict, updated articles (with a field 'nps' which contains a list [text, parse])
    """

    print("Computing NPs (from {})...".format(' '.join(text_type.split('_'))))

    xpath = xpaths['parse']
    count_files = 0

    for file_id in articles:
        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        try:
            root = ET.fromstring(articles[file_id][text_type])
            parses = [parse.text for parse in root.findall(xpath)]

            # Remove duplicates
            nps_string = list(OrderedDict.fromkeys([nps for parse in parses for nps in get_nps(parse)]))
            nps_list = [np_string.split('|') for np_string in nps_string]

            if 'nps' in articles[file_id]:
                articles[file_id]['nps'].extend(nps_list)
            else:
                articles[file_id]['nps'] = nps_list

        except ET.ParseError:
            pass

    print("NPs computed ({} articles).\n".format(len(articles)))

    return articles


def get_nps(parse):
    """
    Compute the text and the parsing of the nps mentioned in parse.

    Args:
        parse: str, parsing of a whole sentence

    Returns:
        list, for each NP, text and parsing separated by the character '|'
    """

    res, temp = [], []
    phrases = parse.split('(NP ')[1:]

    for phrase in phrases:

        if phrase == ' ':
            pass

        else:
            count = 1
            for i in range(len(phrase)):
                if phrase[i] == '(':
                    count += 1
                elif phrase[i] == ')':
                    count -= 1

                if count == 0:
                    temp.append(phrase[:i])
                    break

    for phrase in temp:

        phrase_text, phrase_parse = [], []
        words = phrase.replace(')', '(').split('(')

        for word in words:

            if word == '' or word == ' ':
                pass

            else:
                word_split = word.split(' ')
                if len(word_split) == 2:
                    phrase_parse.append(word_split[0])
                    phrase_text.append(word_split[1])
        if len(phrase_text) != len(phrase_parse):
            raise ValueError("Text and parse don't have the same length.")

        res.append(' '.join(phrase_text) + '|' + ' '.join(phrase_parse))

    return res


def compute_filtered_nps(articles, filters, in_name, out_name, similarity_threshold, n_synsets=None, display_count=10000):
    """
    Apply to the nps of articles in in_name the filters specified, and put them in out_name.

    Args:
        articles: dict, articles to fill
        filters: list, filters to apply (in the order)
        in_name: str, name of the field of the input
        out_name: str, name of the field of the output

    Returns:
        dict, updated articles
    """

    print("Computing filtered NPs (filters: {}; from '{}', to '{}')...".format(', '.join(filters), in_name, out_name))

    count_files, count_treated, count_filtered_in, count_filtered_out = 0, 0, 0, 0

    for file_id in articles:

        count_files += 1
        if count_files % display_count == 0:
            print("Article {}/{}...".format(count_files, len(articles)))

        try:
            filtered_nps = articles[file_id][in_name]
            entities = articles[file_id]['entities']

            count_treated += 1

            for f in filters:
                if filtered_nps:
                    filtered_nps = filter_nps(filtered_nps, f, entities, similarity_threshold, n_synsets)

            if filtered_nps:
                articles[file_id][out_name] = filtered_nps
                count_filtered_in += 1
            else:
                if in_name == out_name:
                    del articles[file_id][in_name]
                count_filtered_out += 1

        except KeyError:
            pass

    print("Filtered NPs computed ({} articles treated: {} filtered in, {} filtered out).\n".format(
        count_treated,
        count_filtered_in,
        count_filtered_out)
    )

    return articles


def filter_nps(nps, f, entities, similarity_threshold, n_synsets):
    """
    Apply the filter f to the nps.

    Args:
        nps: list, nps to filter
        f: str, filter to apply
        entities: list, entities to be similar with

    Returns:
        list, filtered nps
    """

    res = []

    if f == 'to_exclude':
        for phrase in nps:
            if not is_to_exclude(phrase):
                res.append(phrase)

    elif f == 'is_entity':
        for phrase in nps:
            if not is_entity(phrase, entities):
                res.append(phrase)

    elif f == 'is_plural_text':
        for phrase in nps:
            if is_plural_text(phrase):
                res.append(phrase)

    elif f == 'is_numeric_text':
        for phrase in nps:
            if is_numeric_text(phrase):
                res.append(phrase)

    elif f == 'is_numeric_parse':
        for phrase in nps:
            if is_numeric_parse(phrase):
                res.append(phrase)

    elif f.split('_')[0] == 'similar':
        similarity_type = f.split('_')[1:]
        for phrase in nps:
            score, similar_word, similar_entity = compute_similarity(phrase, entities, similarity_type, n_synsets)
            if score >= similarity_threshold:
                res.append(phrase + [score, similar_word, similar_entity])

    else:
        raise ValueError("Wrong filter.")

    return res


def compute_similarity(phrase, entities, similarity_type, n_synsets):
    """
    Compute the similarity between a phrase and the entities, according to the specified similarity_type.

    Args:
        phrase: list, phrase to check
        entities: list, entities to check
        similarity_type: list, 1st term specified the global method (wn), second the particular similarity method
            (path), 3rd term the words to analyze (text of nouns or pluralnouns)

    Returns:
        list, score, and corresponding word and entity
    """

    res = [0., '', '']

    if similarity_type[2] == 'all':
        words = [word for word in phrase[0].lower().split(' ') if word not in stopwords]
    elif similarity_type[2] == 'nouns':
        words = [phrase[0].lower().split(' ')[i] for i, x in enumerate(phrase[1].split(' '))
                 if x in ['NN', 'NNS', 'NNP', 'NNPS']]
    elif similarity_type[2] == 'pluralnouns':
        words = [phrase[0].lower().split(' ')[i] for i, x in enumerate(phrase[1].split(' '))
                 if x in ['NNS']]
    else:
        raise ValueError("Wrong similarity type (3).")

    if similarity_type[0] == 'wn':
        if similarity_type[2] in ['nouns', 'pluralnouns']:
            # entities_synsets = [
            #     wn.synsets(entity_word, pos=wn.NOUN)[:min([len(wn.synsets(entity, pos=wn.NOUN)), n_synsets])]
            #     for entity in entities for entity_word in entity.split(' ')
            # ]
            entities_synsets = [
                [wn.synsets(entity_word, pos=wn.NOUN)[:min([len(wn.synsets(entity_word, pos=wn.NOUN)), n_synsets])]
                 for entity_word in entity.split(' ')] for entity in entities
            ]
            entities_synsets = [[s for word_s in entity_s for s in word_s] for entity_s in entities_synsets]
        else:
            entities_synsets = [
                [wn.synsets(entity_word)[:min([len(wn.synsets(entity_word)), n_synsets])]
                 for entity_word in entity.split(' ')] for entity in entities
            ]
            entities_synsets = [[s for word_s in entity_s for s in word_s] for entity_s in entities_synsets]

        for word in words:
            if similarity_type[2] in ['nouns', 'pluralnouns']:
                word_synsets = wn.synsets(word, pos=wn.NOUN)[:min([len(wn.synsets(word, pos=wn.NOUN)), n_synsets])]
            else:
                word_synsets = wn.synsets(word)[:min([len(wn.synsets(word)), n_synsets])]

            if word_synsets:
                for i_entity in range(len(entities)):
                    if entities_synsets[i_entity]:
                        if similarity_type[1] == 'path':
                            score = max(
                                [max([s1.path_similarity(s2) if s1.path_similarity(s2)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        elif similarity_type[1] == 'lch':
                            score = max(
                                [max([s1.lch_similarity(s2) if s1.lch_similarity(s2)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        elif similarity_type[1] == 'wup':
                            score = max(
                                [max([s1.wup_similarity(s2) if s1.wup_similarity(s2)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        elif similarity_type[1] == 'res':
                            score = max(
                                [max([s1.res_similarity(s2, brown_ic) if s1.res_similarity(s2, brown_ic)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        elif similarity_type[1] == 'jcn':
                            score = max(
                                [max([s1.jcn_similarity(s2, brown_ic) if s1.jcn_similarity(s2, brown_ic)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        elif similarity_type[1] == 'lin':
                            score = max(
                                [max([s1.lin_similarity(s2, brown_ic) if s1.lin_similarity(s2, brown_ic)
                                      else -1 for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                            )
                        else:
                            raise ValueError("Wrong similarity type (2).")

                        if score > res[0]:
                            res = [score, word, entities[i_entity]]

    elif similarity_type[0] == 'we':
        if similarity_type[1] == 'spacy':
            words_str = ' '.join(words)
            entities_str = ' '.join(entities)

            for w in nlp(words_str):
                for e in nlp(entities_str):
                    score = w.similarity(e)
                    if score >= res[0]:
                        res = [score, w.text, e.text]

    else:
        raise ValueError("Wrong similarity type (1).")

    return res


def compute_score(articles, in_name, n_synsets):
    for file_id in articles:

        try:
            nps = articles[file_id][in_name]
            entities = articles[file_id]['entities']

            for phrase in nps:
                phrase.append(score(phrase, entities, n_synsets))

        except KeyError:
            pass

    return articles


def score(phrase, entities, n_synsets):

    words = [phrase[0].lower().split(' ')[i] for i, x in enumerate(phrase[1].split(' '))
             if x not in ['CD', 'DT', 'CC'] and i not in stopwords]

    entities_synsets = [
        [wn.synsets(entity_word)[:min([len(wn.synsets(entity_word)), n_synsets])]
         for entity_word in entity.split(' ')] for entity in entities
    ]
    entities_synsets = [[s for word_s in entity_s for s in word_s] for entity_s in entities_synsets]

    score = 0.
    for word in words:

        word_synsets = wn.synsets(word)[:min([len(wn.synsets(word)), n_synsets])]

        if word_synsets:

            for i_entity in range(len(entities)):
                if entities_synsets[i_entity]:
                    score += max(
                        [max([s1.path_similarity(s2) if s1.path_similarity(s2)
                              else 0. for s1 in word_synsets]) for s2 in entities_synsets[i_entity]]
                    )/len(entities)

    return score

def compute_most_similar_nps(articles, idx_score, n_best, in_name, out_name, display_count=10000):
    """
    Compute the n_best most similar NPs from the similarities of in_name, in out_name

    Args:
        articles: dict of dict, articles to fill
        n_best: int, number of desired examples
        in_name: str, name of the field of the input
        out_name: str, name of the field of the output

    Returns:
        list of dict, nps of articles updated with the filtered NPs
    """

    print("Computing {} most similar NPs (from {}, in {})...".format(n_best, in_name, out_name))

    count_files = 0
    count_articles, count_articles_in, count_articles_out = 0, 0, 0
    count_nps, count_nps_in, count_nps_out = 0, 0, 0,
    temp = []

    for file_id in articles:
        count_files += 1
        if count_files % display_count == 0:
            print("Step 1: article {}/{}...".format(count_files, len(articles)))

        try:
            for phrase in articles[file_id][in_name]:
                temp.append(phrase[idx_score])

        except KeyError:
            pass

    temp.sort(reverse=True)
    if len(temp) >= n_best:
        threshold = temp[n_best]
    else:
        print("Already less than {} examples ({}).".format(n_best, len(temp)))
        threshold = 0.

    for file_id in articles:
        count_files += 1
        if count_files % display_count == 0:
            print("Step 2: article {}/{}...".format(count_files, len(articles)))

        try:
            phrases = articles[file_id][in_name]
            count_articles += 1
            temp = []

            for phrase in phrases:
                count_nps += 1

                if phrase[idx_score] >= threshold:
                    temp.append(phrase)
                    count_nps_in += 1

                else:
                    count_nps_out += 1

            if temp:
                articles[file_id][out_name] = temp
                count_articles_in += 1
            else:
                if in_name == out_name:
                    del articles[file_id][out_name]
                count_articles_out += 1

        except KeyError:
            pass

    print("Most similar NPs computed (threshold: {}):".format(round(threshold, 2)))
    print("{} articles treated: {} filtered in, {} filtered out).".format(count_articles, count_articles_in,
                                                                          count_articles_out))
    print("{} NPs treated: {} filtered in, {} filtered out).\n".format(count_nps, count_nps_in, count_nps_out))

    return articles


def write_articles(articles, file_name, keys=None, display_count=10000):
    """
    Write in results/{file_name}.txt the fields keys of the articles in articles. Overwrite the file if it already
    exists. Write only if all the keys exist in the article.

    Args:
        articles: dict, articles to write
        file_name: str, name of the file
        keys: list, keys of the articles to write; if None, write all the keys
        display_count: int, number of articles seen between display
    """

    print("Writing the articles (keys: {}; in results/{}.txt)...".format(', '.join(keys), file_name))

    count_articles, count_written = 0, 0

    with open('../results/' + file_name + ".txt", "w+") as f:

        for file_id in articles:

            count_articles += 1
            if count_articles % display_count == 0:
                print("File {}/{}...".format(count_articles, len(articles)))

            if keys is None:
                keys = articles[file_id].keys()

            to_write = 'id: ' + str(file_id) + '\n'

            try:
                to_write += to_string({key: articles[file_id][key] for key in keys}) + '\n'
                f.write(to_write)
                count_written += 1

            except KeyError:
                pass

    print("Articles written ({} written/{} articles).\n".format(count_written, count_articles))


def display_articles(articles, keys=None, limit=10, random=False):
    """
    Display the fields keys of the articles. Display only if all the keys exist in the article.

    Args:
        articles: dict, articles to write
        keys: list, keys of the articles to display; if None, display all the keys
        limit: int, maximum number of articles to display
        random: bool, whether or not to select the articles randomly
    """

    if keys is not None:
        print("Displaying the articles (keys: {}; limit: {}; random: {}):\n\n".format(', '.join(keys), limit, random))
    else:
        print("Displaying the full articles (limit: {}; random: {}):\n\n".format(limit, random))

    count_printed, count_possible = 0, 0

    file_ids = list(articles.keys())

    limit = min(limit, len(file_ids))

    if random:
        np.random.shuffle(file_ids)

    for file_id in file_ids:

        if keys is None:
            keys_temp = articles[file_id].keys()
        else:
            keys_temp = keys

        to_print = 'id: ' + str(file_id)

        try:
            to_print += to_string({key: articles[file_id][key] for key in keys_temp}) + '\n'

            if count_printed < limit:
                print(to_print)
                count_printed += 1

            count_possible += 1

        except KeyError:
            pass

    print("\nArticles written ({} printed/{} possible).\n".format(count_printed, count_possible))


def main():
    return


if __name__ == '__main__':
    main()
