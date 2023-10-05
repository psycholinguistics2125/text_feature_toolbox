""" This analysis are an adaptation of this repesotory :
https://github.com/nicolashernandez/READI-LREC22
Some part of yhe code are direct copy
Author :  Quillivic Robin
created_at : 2022, 19 nov

Description:  compute readability features
"""

import numpy as np
import pandas as pd
import re
import spacy
from spacy.matcher import Matcher


from src.features.readability.stats import common_scores as CS
from src.features.readability.stats import diversity
from src.features.readability.stats import perplexity
from src.features.readability.utils.utils import load_dependency
from src.features.readability.stats import word_list_based as WLB
from src.features.readability.stats import discourse


def load_common_scores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the common score of readability
    # 3min 54 pour le corpus 13 Novembre
    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data["sentences"] = data["token"].apply(
        lambda x: [
            sent.split(" ")
            for sent in re.split("\.|\?|\!", " ".join(x))
            if len(sent) > 0
        ]
    )  # list(list(str))
    data["sentences_number"] = data["sentences"].apply(len)
    data["words_number"] = data["token"].apply(len)
    data["longWords_number"] = data["token"].apply(
        lambda x: len([elt for elt in x if len(elt) > 6])
    )

    data["GFI_score"] = data["sentences"].apply(CS.GFI_score)
    data["ARI_score"] = data["sentences"].apply(CS.ARI_score)
    data["FRE_score"] = data["sentences"].apply(CS.FRE_score)
    data["FKGL_score"] = data["sentences"].apply(CS.FKGL_score)
    data["SMOG_score"] = data["sentences"].apply(CS.SMOG_score)
    data["REL_score"] = data["sentences"].apply(CS.REL_score)

    features_names = [
        "sentences_number",
        "words_number",
        "longWords_number",
        "GFI_score",
        "ARI_score",
        "FRE_score",
        "FKGL_score",
        "SMOG_score",
        "REL_score",
    ]

    return data, features_names


def load_diversity_score(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the common score of diversity
    # 20s pour le corpus 13 Novembre
    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    data["token_ratio_score"] = data["pos"].apply(diversity.type_token_ratio)
    data["noun_ratio_score"] = data["pos"].apply(diversity.noun_token_ratio)
    data["proper_noun_ratio_score"] = data["pos"].apply(
        diversity.proper_noun_token_ratio
    )
    data["adverb_ratio_score"] = data["pos"].apply(diversity.adverb_token_ratio)

    features_names = [
        "noun_ratio_score",
        "token_ratio_score",
        "adverb_ratio_score",
        "proper_noun_ratio_score",
    ]
    return data, features_names


def load_perplexity_score(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the perplexity score
    # NEED 3 min 42 on all 13 November Corpus
    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    GPT2_LM = load_dependency("GPT2_LM")
    data["perplexity_score"] = data["text"].apply(
        lambda x: perplexity.PPPL_score(GPT2_LM, x)
    )

    features_names = ["perplexity_score"]
    return data, features_names


def load_external_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the score from the external ressources duboix and lexique.org

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    df_duboix = load_dependency("dubois_dataframe")
    df_lexique = load_dependency("lexique_dataframe")

    features_names = ["lexique_old20_score", "duboix_score"]

    data["lexique_old20_score"] = data["token"].apply(
        lambda x: WLB.average_levenshtein_distance(
            df_lexique["dataframe"], token_list=x
        )
    )
    data["duboix_score"] = data["lemma"].apply(
        lambda x: WLB.dubois_buyse_ratio(df_duboix["dataframe"], lemma_list=x)
    )

    return data, features_names


def load_discourses_scores(data: pd.DataFrame, nlp: spacy.Language) -> pd.DataFrame:
    """Compute discourse features,


    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data["text"] = data["text"].apply(
        lambda x: x.replace("  ", " ")
    )  # replace double space
    doc_list = []
    for doc in nlp.pipe(data["text"], batch_size=50, as_tuples=False, n_process=-1):
        doc_list.append(doc)
    data["spacy_doc"] = doc_list

    # building_features
    # To do Parralelize spacy
    data["discourse_entity_density"] = data["spacy_doc"].apply(
        lambda x: discourse.entity_density(x, unique=False)
    )
    data["discourse_unique_entity_density"] = data["spacy_doc"].apply(
        lambda x: discourse.entity_density(x, unique=True)
    )
    data["discourse_referring_ratio"] = data["spacy_doc"].apply(
        discourse.referring_entity_ratio
    )
    data["discourse_average_length"] = data["spacy_doc"].apply(
        discourse.average_length_reference_chain
    )

    for kind in [
        "indefinite_NP",
        "definite_NP",
        "NP_without_determiner",
        "possessive_determiner",
        "demonstrative_determiner",
        "proper_name",
        "reflexive_pronoun",
        "relative_pronoun",
        "demonstrative_pronoun",
    ]:
        data[f"discourse_mention_{kind}"] = data["spacy_doc"].apply(
            lambda x: discourse.count_type_mention(x, mention_type=kind)
        )
        data[f"discourse_opening_{kind}"] = data["spacy_doc"].apply(
            lambda x: discourse.count_type_opening(x, mention_type=kind)
        )

    features_names = [
        "discourse_entity_density",
        "discourse_unique_entity_density",
        "discourse_referring_ratio",
        "discourse_average_length",
        "discourse_mention_indefinite_NP",
        "discourse_opening_indefinite_NP",
        "discourse_mention_definite_NP",
        "discourse_opening_definite_NP",
        "discourse_mention_NP_without_determiner",
        "discourse_opening_NP_without_determiner",
        "discourse_mention_possessive_determiner",
        "discourse_opening_possessive_determiner",
        "discourse_mention_demonstrative_determiner",
        "discourse_opening_demonstrative_determiner",
        "discourse_mention_proper_name",
        "discourse_opening_proper_name",
        "discourse_mention_reflexive_pronoun",
        "discourse_opening_reflexive_pronoun",
        "discourse_mention_relative_pronoun",
        "discourse_opening_relative_pronoun",
        "discourse_mention_demonstrative_pronoun",
        "discourse_opening_demonstrative_pronoun",
    ]

    return data, features_names


def load_PQP_score(data: pd.DataFrame, nlp: spacy.Language) -> pd.DataFrame:
    """Compute the prop of plus que parfait in text

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data["text"] = data["text"].apply(lambda x: x.replace("  ", " "))

    PQP_rule = [
        {"POS": "AUX", "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}, "OP": "+"},
        {"TAG": "ADV", "OP": "*"},
        {
            "POS": "VERB",
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part"]},
        },
    ]

    matcher = Matcher(nlp.vocab)
    matcher.add("PQP_rule", [PQP_rule], greedy="LONGEST")

    if "spacy_doc" not in data.columns:
        doc_list = []
        for doc in nlp.pipe(data["text"], batch_size=50, as_tuples=False, n_process=-1):
            doc_list.append(doc)
        data["spacy_doc"] = doc_list

    data["PQP_list"] = data["spacy_doc"].apply(
        lambda x: [x[s:e] for id_, s, e in matcher(x)]
    )
    data["PQP_score"] = data["PQP_list"].apply(len) / data["token"].apply(len) * 1000

    features_names = ["PQP_score"]
    return data, features_names


def load_direct_discours_score(data: pd.DataFrame) -> pd.DataFrame:
    """compute the proportion of direct discourse drom colomn 'text'
    This score is based on regulmar expression

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    pattern = r""":\s?\"(.*?)\""""
    data["direct_discourse_list"] = data["text"].apply(
        lambda x: list(re.finditer(pattern, x))
    )
    data["direct_discourse_text"] = data["direct_discourse_list"].apply(
        lambda x: "\n".join(
            [elt.group() for elt in x if elt.span()[1] - elt.span()[0] < 500]
        )
    )
    data["direct_discourse_score"] = data["direct_discourse_text"].apply(len) / data[
        "text"
    ].apply(len)

    features_names = ["direct_discourse_score", "direct_discourse_text"]
    return data, features_names


def load_readability_score(data: pd.DataFrame, nlp_readability, _logger):
    features = pd.DataFrame(index=data.index)
    for col in ["code"]:
        features[col] = data[col]

    _logger.info(
        "Readability features were selected, computing all the readability scores in the text data..."
    )

    features_names = []

    _logger.info("Computing common scores for readability...")
    features_CS, readability_features_names_CS = load_common_scores(data)
    features = features.merge(
        features_CS[readability_features_names_CS], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_CS)

    _logger.info("Computing diversity score for readability")
    features_Div, readability_features_names_Div = load_diversity_score(data)
    features = features.merge(
        features_Div[readability_features_names_Div], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_Div)

    _logger.info("Computing perplexity scores for readability")
    features_P, readability_features_names_P = load_perplexity_score(data)
    features = features.merge(
        features_P[readability_features_names_P], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_P)

    _logger.info("Computing externals scores for readability")
    features_E, readability_features_names_Ext = load_external_scores(data)
    features = features.merge(
        features_E[readability_features_names_Ext], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_Ext)

    _logger.info("Computing Discourse scores for readability")
    features_D, readability_features_names_D = load_discourses_scores(
        data, nlp_readability
    )
    features = features.merge(
        features_D[readability_features_names_D], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_D)

    _logger.info("Computing Pqp score")
    features_pqp, readability_features_names_pqp = load_PQP_score(data, nlp_readability)
    features = features.merge(
        features_pqp[readability_features_names_pqp], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_pqp)

    _logger.info("Computing discour direct scores for readability")
    features_dd, readability_features_names_dd = load_direct_discours_score(data)
    features = features.merge(
        features_dd[readability_features_names_dd], left_index=True, right_index=True
    )
    features_names.append(readability_features_names_dd)

    return features, features_names
