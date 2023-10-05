"""
Author: Quillivic Robin
Created_at: 2022,22 November
Description: This files contains some idea to detect dysfluences from french interviex retranscription

Detect disfluencies is a difficult task and often use mutlti modal data (audio and text).
In French, we did not find any pretrained model, furtermore, we are going to use some lecical based pproches
and rule based approach

Here are some useful ressources :

Camille dutrey: PhD : https://www.theses.fr/2014PA112415
Avanzi paper: https://www.researchgate.net/publication/281557958_Automatic_Detection_and_Annotation_of_Disfluencies_in_Spoken_French_Corpora
etude des particules: https://archivesic.ccsd.cnrs.fr/sic_00001231/document
corpus des particules: https://hal.inria.fr/hal-01585540/document
TO DO : ask for data and model : https://hal.inria.fr/hal-01585540/document



Annotated corpus for french, (not used now) : https://rhapsodie.modyco.fr/

Ressources in english :
https://github.com/pariajm/awesome-disfluency-detection
https://github.com/pariajm/deep-disfluency-detector#acnn-model

"""

import sys, os

import pandas as pd
import numpy as np

import pyphen
import string

dic = pyphen.Pyphen(lang="fr")

from src import utils
import re

# ressources

dyfluences_dict = {
    "generical_connector": [
        "mais",
        "donc",
        "aussi",
        "parce que",
        "et",
        "de plus",
        "puis",
        "en outre",
        "non seulement",
        "mais encore",
        "de surcroît",
        "ainsi que",
        "également",
    ],
    "temporal_connector": [
        "quand",
        "lorsque",
        "avant que",
        "après que",
        "alors que",
        "dès lors que",
        "depuis que",
        "tandis que",
        "en même temps que",
        "pendant que",
        "au moment où",
    ],
    "parenthetique": ["à savoir", "c'est-à-dire", "soit", "je veux dire", "je précise"],
    "paticules": [
        "bon",
        "bah",
        "voila",
        "ben",
        "quoi",
        "donc",
        "hein",
        "euh",
        "eh",
        "eh quoi",
        "ah",
    ],
    "troncations": [r"\.\.\."],
    "onomatopes": [
        "ah",
        "aïe",
        "atchoum",
        "badaboum",
        "bang",
        "pop",
        "bang",
        "pan" "tak",
        "bang",
        "blam",
        "boom",
        "broum",
        "bzzz",
        "chut",
        "clac",
        "crac" "grrr",
        "hé",
        "eh",
        "hi",
        "oh",
        "ouch",
        "ouf",
        "oups",
        "paf",
        "psitt",
        "psst",
        "snif",
        "toc",
        "tic",
    ],
}


def compute_syllabe_list(words: list) -> list:
    # remove punctuation from text and lower
    clean_words = [
        elt.lower().strip().translate(str.maketrans("", "", string.punctuation))
        for elt in words
    ]
    # compute and flatten syllabe list
    syl = [dic.inserted(word).split("-") for word in clean_words if len(word) > 0]
    flatten_syl = utils.flatten(syl)

    return flatten_syl


def compute_repetition(syl_list: list, windows_size=3) -> list:
    """_summary_

    Args:
        words (list): _description_
        windows_size (int, optional): _description_. Defaults to 3.

    Returns:
        list: _description_
    """
    # init repetition list
    rep = []
    for i in range(len(syl_list) - windows_size):
        s = syl_list[i]
        for k in range(1, windows_size + 1):
            s_bis = syl_list[i + k]
            if s_bis == s:
                rep.append((s, s_bis, k))

    return rep


def find_matches(text, word_list: list) -> list:
    matches = []
    for word in word_list:
        pattern = (
            r"\s+" + word + r"(\s+|[().,;:!?]+)"
        )  # the word with a space before and a space or punctuation mark after
        matches.append(list(re.finditer(pattern, text.strip().lower())))
    return utils.flatten(matches)


def load_dysfluences_scores(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
        list (_type_): _description_
    """

    data["text"] = data["text"].apply(lambda x: x.replace("  ", " "))
    data["syllabe_list"] = data["token"].apply(compute_syllabe_list)

    features_names = []
    # repetion_score
    for i in range(1, 7):
        data[f"repetiton_type_{i}_list"] = data["syllabe_list"].apply(
            lambda x: compute_repetition(x, windows_size=i)
        )
        data[f"repetitions_type_{i}_score"] = data[f"repetiton_type_{i}_list"].apply(
            len
        ) / data["syllabe_list"].apply(len)
        features_names.append(f"repetitions_type_{i}_score")

    for key, value in dyfluences_dict.items():
        if key == "troncations":
            data[f"{key}_matches"] = data["text"].apply(
                lambda x: utils.flatten(
                    [list(re.finditer(v, x.strip().lower())) for v in value]
                )
            )
        else:
            data[f"{key}_matches"] = data["text"].apply(
                lambda x: find_matches(x, value)
            )
        data[f"score_{key}_matches"] = (
            data[f"{key}_matches"].apply(len) / data["token"].apply(len) * 100
        )
        features_names.append(f"score_{key}_matches")

    return data, features_names
