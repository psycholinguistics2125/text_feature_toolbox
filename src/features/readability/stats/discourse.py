"""
The discourse module contains functions allowing to calculate notions related to text cohesion.

Text cohesion means how parts of a text are related with explicit formal grammatical ties.
The following notions are used: co-reference or anaphoric chains, entity density, POS-tag based cohesion measures.
Currently, most of these features have been based upon the ones referenced in this paper: 
https://hal.archives-ouvertes.fr/hal-01430554 [Are Cohesive Features Relevant for Text Readability Evaluation?]
However, please note that some implementations could be improved, as this is a somewhat recent notion.
"""
import os
import coreferee
import pandas as pd
import spacy

from ..utils import utils


DATA_ENTRY_POINT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "data")
)


def entity_density(doc, unique=False):
    """
    Entity density is the number of all or unique entities in document, divided by text length.

    :param bool unique: Whether to return proportion of all entities in document, or only unique entities.
    """
    chains = doc._.coref_chains
    length = len(doc.text)

    if unique:
        # Number of chains ~~ Number of unique entities. Not exactly true since that won't include entities that only appear once.

        return len(chains) / length
    else:
        counter = 0
        # Number of entities == Number of every mention in chains.
        for chain in chains:
            counter += len(chain)
        return counter / length


def referring_entity_ratio(doc):
    """Returns amount of anaphoric mentions in text : Mentions of an entity except for the entity itself."""

    chains = doc._.coref_chains
    length = len(doc.text)
    counter = 0
    for chain in chains:
        counter += (
            len(chain) - 1
        )  # Remove initial appearence of entity when counting number of referring entities
    counter = counter / len(chains)  # Average over number of chains
    return counter / length


# NOTE: coreferee only gives us one token per entity, so things like New York will be shortened to New
# Unfortunately, coreferee ignores the additional spacy pipeline component 'merge_entities'.
# A temporary solution is to use spacy.ents to get the full name of any recognized named entity.
# However this won't work for not-named entities that are composite, like "cette femme".
def average_entity_word_length(doc: spacy.tokens.doc.Doc):
    """Returns average word length of each entity."""

    text = doc.text
    counter = 0
    entity_dict = dict()
    nb_entities = 0
    # Store composite entity's length for later
    for ent in doc.ents:
        entity_dict[ent.start] = len(ent.text.split())

    for chain in doc._.coref_chains:
        for mention in chain:
            nb_entities = nb_entities + 1
            for index in mention.token_indexes:
                # At this point we have an index, check if that index is part of a composite entity to get its length.
                if index in list(entity_dict.keys()):
                    counter += entity_dict[index]
                else:
                    counter += 1
    # Average over number of entities
    counter = counter / nb_entities
    return counter / len(text)


# Co-reference chain properties.
def average_length_reference_chain(doc):
    """Counts number of mentions appearing in coreference chains, and returns the average."""

    chains = doc._.coref_chains
    length = len(doc.text)
    counter = 0
    for chain in chains:
        counter += len(chain)  # Get length of chain
    counter = counter / len(chains)  # Average over number of chains
    return counter


# Utility function for co-reference chains
def spacy_filter_coreference_count(
    doc, mention_index, mention_type, noun_groups_info=None
):
    """Utility function for handling co-reference chains, not meant to be called directly from the processor."""
    if mention_type == "indefinite_NP":
        for possible_group in noun_groups_info:
            if possible_group[0] < mention_index < possible_group[1]:
                if doc[possible_group[0]].morph.__contains__("Definite=Ind"):
                    return 1
        return 0
    elif mention_type == "definite_NP":
        for possible_group in noun_groups_info:
            if possible_group[0] < mention_index < possible_group[1]:
                if doc[possible_group[0]].morph.__contains__("Definite=Def"):
                    return 1
        return 0
    elif mention_type == "NP_without_determiner":
        for possible_group in noun_groups_info:
            if possible_group[0] < mention_index < possible_group[1]:
                if not doc[possible_group[0]].dep_ == "det":
                    return 1
        return 0
    elif mention_type == "possessive_determiner":
        if doc[mention_index].dep_ == "det" and doc[mention_index].morph.__contains__(
            "Poss=Yes"
        ):
            return 1
        else:
            return 0
    elif mention_type == "demonstrative_determiner":
        if doc[mention_index].dep_ == "det" and doc[mention_index].morph.__contains__(
            "PronType=Dem"
        ):
            return 1
        else:
            return 0
    elif mention_type == "proper_name":
        if doc[mention_index].pos_ == "PROPN":
            return 1
        else:
            return 0
    elif mention_type == "personal_pronoun":
        # FIXME: this is not accurate enough.
        if doc[mention_index].pos_ == "PRON" and (
            doc[mention_index].morph.__contains__("Gender=Masc")
            or doc[mention_index].morph.__contains__("Gender=Fem")
        ):
            return 1
        else:
            return 0
    elif mention_type == "reflexive_pronoun":
        if doc[mention_index].pos_ == "PRON" and doc[mention_index].morph.__contains__(
            "Reflex=Yes"
        ):
            return 1
        else:
            return 0
    elif mention_type == "relative_pronoun":
        if doc[mention_index].pos_ == "PRON" and doc[mention_index].morph.__contains__(
            "PronType=Rel"
        ):
            return 1
        else:
            return 0
    elif mention_type == "indefinite_pronoun":
        # Can't figure out how to get it with only spacy's information
        return 0
    elif mention_type == "demonstrative_pronoun":
        if doc[mention_index].pos_ == "PRON" and doc[mention_index].morph.__contains__(
            "PronType=Dem"
        ):
            return 1
        else:
            return 0
    else:
        print("i don't recognize that type of mention")
        return -1


def count_type_mention(doc: spacy.tokens.doc.Doc, mention_type=None, nlp=None):
    """
    Returns the ratio of a specific type of mention in a text per coreference chain.

    As of now, the recognized mention types are :
        * indefinite_NP
        * definite_NP
        * NP_without_determiner
        * possessive_determiner
        * demonstrative_determiner
        * proper_name
        * reflexive_pronoun
        * relative_pronoun
        * demonstrative_pronoun

    :param str mention_type: Denotes the type of mention to be recognized.
    :return: Amount of times a specific type of mention appears in a text, divided by the number of coreference chains.
    :rtype: float
    """

    counter = 0

    # For noun phrases, coreferee only returns a single token but noun phrases are several token longs (they're spans)
    # So each noun phrase's starting and ending indexes are stored for later.
    noun_phrases_info = []
    for np in doc.noun_chunks:
        noun_phrases_info.append((np.start, np.end, np))

    # For the first mention of each entity, get the index via mention.token_indexes. It's a complex mention if nb token_indexes > 1
    for chain in doc._.coref_chains:
        for mention in chain:
            for index in mention.token_indexes:
                # Increment counter if mention type to check is the same as mention's type thanks to spacy filter
                counter += spacy_filter_coreference_count(
                    doc, index, mention_type, noun_phrases_info
                )
    counter = counter / len(doc._.coref_chains)  # Average over number of chains
    return counter


def count_type_opening(doc: spacy.tokens.doc.Doc, mention_type=None):
    """
    Returns the ratio of a specific type of mention in a text for the first mention, which usually introduces the entity, per coreference chain.

    As of now, the recognized mention types are :
        * indefinite_NP
        * definite_NP
        * NP_without_determiner
        * possessive_determiner
        * demonstrative_determiner
        * proper_name
        * reflexive_pronoun
        * relative_pronoun
        * demonstrative_pronoun

    :param str mention_type: Denotes the type of mention to be recognized.
    :return: Amount of times a specific type of mention appears in a text, divided by the number of coreference chains.
    :rtype: float
    """

    counter = 0

    # For noun phrases, coreferee only returns a single token but noun phrases are several token longs (they're spans)
    # So each noun phrase's starting and ending indexes are stored for later.
    noun_phrases_info = []
    for np in doc.noun_chunks:
        noun_phrases_info.append((np.start, np.end, np))

    # For the first mention of each entity, get the index via mention.token_indexes. It's a complex mention if nb token_indexes > 1
    for chain in doc._.coref_chains:
        for index in chain[0].token_indexes:
            # Increment counter if mention type to check is the same as mention's type thanks to spacy filter
            counter += spacy_filter_coreference_count(
                doc, index, mention_type, noun_phrases_info
            )
    counter = counter / len(doc._.coref_chains)  # Average over number of chains
    return counter


def stub_lexical_tightness(text, nlp=None):
    """Not implemented yet, research is needed to understand what lexical tightness is, and how to evaluate it."""
    # TODO : Research what is lexical tightness, and how to evaluate it.
    return 0


# The following features were significant in Todirascu's paper but I don't quite know what they are.
# This is syntactic transition type?
def distance_object_to_none(text, nlp=None):
    """
    Object to None : distance between 2 consecutive mentions of same chain is larger than 1 sentence.
    First, get coreference chains on entire text:
    Then divide text into sentences
    Associate each mention into its sentence thanks to doc._.coref_chains[][].token_indexes
    Get the most "relevant" mention per sentence thanks to sentence._.most_relevant_thing (i forgot the actual name, do a .__dict__ to see)
    Then check the type between each mention of specific entities. (or if it doesn't appear in adjacent sentences => X to None)
    Remember to NOT recreate a doc by doing sentence = nlp(sentence) because the lack of context will remove certain entities.
    Instead manually recreate the ChainHolder items by subsetting.
    """
    return 0


# TODO: figure out how to recognize deictic words.
def first_chain_is_deictic(text, nlp=None):
    """Not implemented yet, need to figure out how to recognize deictic words."""
    return 0
