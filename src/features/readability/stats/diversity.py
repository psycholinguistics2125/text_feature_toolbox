"""
The diversity module contains functions allowing to calculate notions related to text diversity. 

Currently it's only the text token ratio and its noun-only version.
These ratios can have variants which can be selected by changing the mode parameter. (root TTR, corrected TTR)
For instance, mode == "root" will square the denominator of the ratio, which is supposed to be more robust for longer texts.

For future development, things that could be added are : Yule's k, lexical density measures, and n-gram lexical features.
"""
import math
import string
import pandas as pd

from ..utils import utils
from collections import Counter


def type_token_ratio(token_list, mode=None):
    """
    Outputs two ratios : ttr and root ttr : number of lexical items / number of words

    :param str text: Content of a text, converted to string if it's already a list of tokens
    :param str mode: Which version of the ttr to return
    :return: text token ratio, mode can be "root", "corrected", and defaults to standard (TTR)
    :rtype: float
    """

    # Convert to string if list/list of lists + handle punctuation.
    clean_token_list = [
        elt for elt in token_list if elt not in [c for c in string.punctuation]
    ]

    nb_unique = len(set(clean_token_list))
    nb_tokens = len(clean_token_list)

    if nb_tokens == 0:
        print(
            "WARNING: Current text's content is empty, returned type_token_ratio value has been set to 0"
        )
        return 0

    if mode == "corrected":
        return nb_unique / math.sqrt(2 * nb_tokens)
    elif mode == "root":
        return nb_unique / math.sqrt(nb_tokens)
    else:
        return nb_unique / nb_tokens


def noun_token_ratio(pos_list, mode=None):
    """
    Outputs variant of the type token ratio, the TotalNoun / Noun ratio.

    :param str text: Content of a text, converted to string if it's already a list of tokens
    :param nlp: What natural language processor to use, currently only spacy is supported.
    :type nlp: spacy.lang
    :param str mode: Which version of the ttr to return
    :return: noun token ratio, mode can be "root", "corrected", and defaults to standard (TTR)
    :rtype: float
    """

    nouns = [elt for elt in pos_list if elt[1] == "NOUN"]
    nb_unique = len(set(nouns))
    nb_tokens = len(nouns)

    if nb_tokens == 0:
        print(
            "WARNING: Current text's content is empty or no nouns have been recognized, returned noun_token_ratio value has been set to 0"
        )
        return 0

    if mode == "corrected":
        return nb_unique / math.sqrt(2 * nb_tokens)
    elif mode == "root":
        return nb_unique / math.sqrt(nb_tokens)
    else:
        return nb_unique / nb_tokens


def proper_noun_token_ratio(pos_list, mode=None):
    """
    Outputs variant of the type token ratio, the TotalNoun / Noun ratio.

    :param str text: Content of a text, converted to string if it's already a list of tokens
    :param nlp: What natural language processor to use, currently only spacy is supported.
    :type nlp: spacy.lang
    :param str mode: Which version of the ttr to return
    :return: noun token ratio, mode can be "root", "corrected", and defaults to standard (TTR)
    :rtype: float
    """

    nouns = [elt for elt in pos_list if elt[1] == "PROPN"]
    nb_unique = len(set(nouns))
    nb_tokens = len(nouns)

    if nb_tokens == 0:
        print(
            "WARNING: Current text's content is empty or no nouns have been recognized, returned noun_token_ratio value has been set to 0"
        )
        return 0

    if mode == "corrected":
        return nb_unique / math.sqrt(2 * nb_tokens)
    elif mode == "root":
        return nb_unique / math.sqrt(nb_tokens)
    else:
        return nb_unique / nb_tokens


def adverb_token_ratio(pos_list, mode=None):
    """
    Outputs variant of the type token ratio, the TotalNoun / Noun ratio.

    :param str text: Content of a text, converted to string if it's already a list of tokens
    :param nlp: What natural language processor to use, currently only spacy is supported.
    :type nlp: spacy.lang
    :param str mode: Which version of the ttr to return
    :return: noun token ratio, mode can be "root", "corrected", and defaults to standard (TTR)
    :rtype: float
    """

    nouns = [elt for elt in pos_list if elt[1] == "ADV"]
    nb_unique = len(set(nouns))
    nb_tokens = len(nouns)

    if nb_tokens == 0:
        print(
            "WARNING: Current text's content is empty or no nouns have been recognized, returned noun_token_ratio value has been set to 0"
        )
        return 0

    if mode == "corrected":
        return nb_unique / math.sqrt(2 * nb_tokens)
    elif mode == "root":
        return nb_unique / math.sqrt(nb_tokens)
    else:
        return nb_unique / nb_tokens
