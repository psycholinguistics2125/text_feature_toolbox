""" 
File that includes tools to plot verbatim from the textual features
"""

import pandas as pd
import os, sys
import logging

from src import utils
from src.features import analysis_sentiment
import re


def load_data_sample(config, k=10):
    logger = logging.getLogger()

    nlp_data = utils.load_nlp_data(config)
    filter_data = utils.filter_data(nlp_data, config, logger=logger)

    # testing_purposes
    data = filter_data[["code", "text", "token", "lemma", "morph", "pos"]].reset_index()

    data_copy = data.copy()

    return data_copy.sample(k)


def get_rule(elt, tag):
    if tag == "indicatif_future":
        rule = ("Mood=Ind" in elt[2]) and ("Tense=Fut" in elt[2]) and (elt[1] == "VERB")
    elif tag == "participe_passe":
        rule = (
            ("Tense=Past" in elt[2])
            and ("VerbForm=Part" in elt[2])
            and (elt[1] == "VERB")
        )
    elif tag == "indicatif_imparfait":
        rule = ("Mood=Ind" in elt[2]) and ("Tense=Imp" in elt[2]) and (elt[1] == "VERB")
    elif tag == "indicatif_present":
        rule = (
            ("Mood=Ind" in elt[2]) and ("Tense=Pres" in elt[2]) and (elt[1] == "VERB")
        )
    elif tag == "conditionnel":
        rule = ("Mood=Cnd" in elt[2]) and (elt[1] == "VERB")
    else:
        rule = False
        return "tag is not correct ! "

    return rule


def interpret_morph(line, tag):
    select = []
    ref = []
    context = []
    x = line["morph"]

    for i in range(len(x)):
        elt = x[i]
        if elt:
            rule = get_rule(elt, tag)
            if rule:
                select.append(elt[0])
                context.append(" ".join(line["token"][i - 10 : i + 10]))
            if elt[1] == "VERB":
                ref.append(elt[0])

    df_interpret = pd.DataFrame()
    df_interpret["target"] = select
    df_interpret["context"] = context
    df_interpret["validation"] = None

    return df_interpret


def save_interpretation_morph(config, k=100):
    sample = load_data_sample(config, k=k)
    saving_folder = config["data"]["interpretation_folder"]
    for tag in [
        "indicatif_future",
        "participe_passe",
        "indicatif_imparfait",
        "indicatif_present",
        "conditionnel",
    ]:
        df_ = pd.concat(
            sample.apply(lambda x: interpret_morph(x, tag), axis=1).tolist()
        ).reset_index()
        saving_path = os.path.join(saving_folder, f"{tag}_interpretation.csv")
        df_.to_csv(saving_path, index=None)

    return None


def interpret_lexicon(line, lexicon):
    affect_list = []
    target_list = []
    context_list = []
    affect_dict = dict()
    lexicon_keys = lexicon.keys()
    for i in range(len(line.token)):
        word = line.lemma[i]
        if word in lexicon_keys:
            target_list.append(word)
            affect_list.append(lexicon[word])
            affect_dict.update({word: lexicon[word]})
            context = " ".join(line.token[i - 10 : i + 10])
            context_list.append(context)

    df_ = pd.DataFrame()
    df_["label"] = affect_list
    df_["word"] = target_list
    df_["context"] = context_list

    return df_


def get_lexicon_from_name(lexicon_name, sentiment_pipeline):
    if lexicon_name == "feel":
        lexicon = sentiment_pipeline.feel_lexicon
    elif lexicon_name == "polarimot":
        lexicon = sentiment_pipeline.polarimot_lexicon
    elif lexicon_name == "empath":
        lexicon = sentiment_pipeline.augustin_lexicon
    elif lexicon_name == "liwc":
        return sentiment_pipeline.liwc_parser
    else:
        lexicon = False

    return lexicon


def interpret_liwc(line, liwc_parser):
    affect_list = []
    target_list = []
    context_list = []
    for i in range(len(line.token)):
        word = line.token[i]
        target = [category for category in liwc_parser(word)]
        if len(target) > 1:
            target_list.append(word)
            affect_list.append(target)
            context = " ".join(
                line.token[max(0, i - 10) : min(i + 10, len(line.token))]
            )
            # print(context)
            context_list.append(context)

    df_ = pd.DataFrame()
    df_["label"] = affect_list
    df_["word"] = target_list
    df_["context"] = context_list

    return df_


def save_interpretation_lexicon(config, k=100):
    sample = load_data_sample(config, k=k)
    sentiment_pipeline = analysis_sentiment.SentimentAnalysis(
        config["features"]["sentiments"]["resources_path"]
    )
    saving_folder = config["data"]["interpretation_folder"]
    for lexicon_name in ["feel", "polarimot", "empath", "liwc"]:
        lexicon = get_lexicon_from_name(lexicon_name, sentiment_pipeline)
        if lexicon_name == "liwc":
            df_ = pd.concat(
                sample.apply(lambda x: interpret_liwc(x, lexicon), axis=1).tolist()
            ).reset_index()
        else:
            df_ = pd.concat(
                sample.apply(lambda x: interpret_lexicon(x, lexicon), axis=1).tolist()
            ).reset_index()
        saving_path = os.path.join(saving_folder, f"{lexicon_name}_interpretation.csv")
        df_.to_csv(saving_path, index=None)

    return None


def interpret_truncation(line, pattern):
    text = line.text
    context_list = []
    match_list = []
    for elt in list(re.finditer(pattern, text.strip().lower())):
        start = elt.span()[0]
        end = elt.span()[1]
        context = text[start - 30 : end + 20]
        context_list.append(context)
        match_list.append(text[start:end])

    df = pd.DataFrame()
    df["target"] = match_list
    df["context"] = context_list
    df["is_truncation"] = None

    return df


def save_interpretation_truncation(config, k=100):
    sample = load_data_sample(config, k=k)
    saving_folder = config["data"]["interpretation_folder"]
    for pattern in ["\.\.\."]:
        df_ = pd.concat(
            sample.apply(lambda x: interpret_truncation(x, pattern), axis=1).tolist()
        ).reset_index()
        if pattern == "\.\.\.":
            pattern_name = "truncation"
        else:
            pattern_name = pattern
        saving_path = os.path.join(saving_folder, f"{pattern_name}_interpretation.csv")
        df_.to_csv(saving_path, index=None)

    return None
