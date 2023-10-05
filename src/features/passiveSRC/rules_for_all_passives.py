"""
Author : Quillivic Robin
created_at : 2022, 15th Nov
Inspired from this file : https://github.com/mitramir55/PassivePy/blob/main/PassivePyCode/PassivePySrc/rules_for_all_passives.py
Description: adaptation du module PassivePyde l'anglais au Français
"""

import spacy
from spacy.matcher import Matcher
import os
import torch
from thinc.api import set_gpu_allocator, require_gpu

try:
    intransitif_verbs_list = [
        elt.replace("\n", "").lower().strip()
        for elt in open(
            "/home/robin/Code_repo/text_features_toolbox/src/features/passiveSRC/data/intransif_verbs.txt",
            encoding="utf8",
        ).readlines()[1:]
    ]
    exception_adj = [
        elt.replace("\n", "").lower().strip()
        for elt in open(
            "/home/robin/Code_repo/text_features_toolbox/src/features/passiveSRC/data/exception_adj_ible.txt",
            encoding="utf8",
        ).readlines()
    ]
    passive_verb_etre = [
        elt.replace("\n", "").lower().strip()
        for elt in open(
            "/home/robin/Code_repo/text_features_toolbox/src/features/passiveSRC/data/transitif_etre_verbs.txt",
            encoding="utf8",
        ).readlines()
    ]
except:
    intransitif_verbs_list = [
        elt.replace("\n", "").lower().strip()
        for elt in open("../data/intransif_verbs.txt", encoding="utf8").readlines()[1:]
    ]
    exception_adj = [
        elt.replace("\n", "").lower().strip()
        for elt in open("../data/exception_adj_ible.txt").readlines()
    ]
    passive_verb_etre = [
        elt.replace("\n", "").lower().strip()
        for elt in open("../data/transitif_etre_verbs.txt").readlines()
    ]


def create_matcher(spacy_model="fr_core_news_lg", nlp: spacy.language.Language = None):
    """creates a matcher on the following vocabulary"""
    if not nlp:
        if "trf" in spacy_model:
            set_gpu_allocator("pytorch")
            require_gpu()
            nlp = spacy.load(spacy_model)
            print(f"trf model loaded, using gpu : {torch.cuda.current_device()}")
        else:
            nlp = spacy.load(spacy_model, disable=["ner"])
    matcher = Matcher(nlp.vocab)

    # list of verbs that their adjective form
    # is sometimes mistaken as a verb
    passiv_verbs = ["user", "subir", "coincer", "hospitaliser", "blesser"]
    # passive_verb_etre = []

    # --------------------------rules--------------------#

    # Passif canonique
    # exemple : J'ai été attaqué par des loups !
    passive_rule_0 = [
        {"POS": "AUX", "DEP": "aux", "OP": "*"},
        {
            "DEP": {"IN": ["iobj", "expl:comp"]},
            "TAG": "PRON",
            "OP": "!",
        },  # absence de pronom réfléxif
        {"POS": {"IN": ["AUX", "VERB"]}, "DEP": "aux:pass", "OP": "+"},
        {
            "DEP": "neg",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"DEP": "HYPH", "OP": "*"},
        {
            "DEP": "advmod",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"POS": {"IN": ["ADP", "DET"]}, "OP": "*"},  # un
        {"TAG": {"IN": ["ADJ", "ADV"]}, "OP": "*"},  # petit
        {"LEMMA": {"IN": [",", "..."]}, "OP": "*"},
        {"TAG": "ADV", "OP": "*"},  # peu
        {
            "POS": "VERB",
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part", "Voice=Pass"]},
            "LEMMA": {"NOT_IN": intransitif_verbs_list},
        },
        {"LOWER": "par"},
    ]
    # Passif tronqué
    # exemple : J'ai été attaqué !
    passive_rule_1 = [
        {"POS": "AUX", "DEP": "aux", "OP": "*"},
        {
            "DEP": {"IN": ["iobj", "expl:comp"]},
            "TAG": "PRON",
            "MORPH": {"IS_SUPERSET": ["Reflex=Yes"]},
            "OP": "!",
        },  # absence de pronom réfléxif
        {"POS": {"IN": ["AUX", "VERB"]}, "DEP": "aux:pass", "OP": "+"},
        {"POS": {"IN": ["ADP", "DET"]}, "OP": "*"},
        {
            "DEP": "neg",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"DEP": "HYPH", "OP": "*"},
        {
            "DEP": "advmod",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"POS": {"IN": ["ADP", "DET"]}, "OP": "*"},  # un
        {"TAG": {"IN": ["ADJ", "ADV"]}, "OP": "*"},  # petit
        {"LEMMA": {"IN": [",", "..."]}, "OP": "*"},
        {"TAG": "ADV", "OP": "*"},  # peu
        {
            "POS": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part"]},
            "LEMMA": {"NOT_IN": intransitif_verbs_list},
        },
    ]

    # Passif canonnique en séquence
    # Exemple: Elle a été entrainée et formaté par les CPGE
    passive_rule_3 = [
        {"POS": "AUX", "DEP": "aux", "OP": "*"},
        {"POS": "AUX", "DEP": "aux:pass", "OP": "+"},
        {
            "DEP": "neg",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"DEP": "HYPH", "OP": "*"},
        {
            "DEP": "advmod",
            "TAG": "ADV",
            "MORPH": {"IS_SUPERSET": ["Degree=Pos"]},
            "OP": "*",
        },
        {"POS": "VERB", "DEP": "ROOT", "LEMMA": {"NOT_IN": intransitif_verbs_list}},
        {"DEP": "cc"},
        {"TAG": "ADV", "OP": "*"},
        {
            "DEP": "advmod",
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part", "Voice=Pass"]},
            "OP": "*",
            "LEMMA": {"NOT_IN": ["pre"]},
        },
        {"DEP": "conj", "LEMMA": {"NOT_IN": intransitif_verbs_list}},
        {"DEP": "pobj", "OP": "!"},
    ]

    # Passif impersonel
    # Exemple: Accusé par la police, il fut comdané à 5 ans de prison
    passive_rule_4 = [
        {
            "DEP": {"IN": ["advcl", "ROOT"]},
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part", "Voice=Pass"]},
            "LEMMA": {"NOT_IN": intransitif_verbs_list},
        },
        {"DEP": "case", "TAG": "ADP"},
        {"DEP": "obl:agent", "OP": "*"},
    ]

    # Passif par la nature des verbe
    # Exemple: J'ai subi des pressions de la part de la Mafia
    passive_rule_6 = [{"LEMMA": {"IN": passiv_verbs}}, {"LOWER": "par"}]

    passive_rule_6_1 = [
        {"LEMMA": {"IN": ["être"]}, "OP": "+"},
        {"TAG": {"IN": ["ADP", "DET"]}, "OP": "*"},
        {"TAG": "ADV", "OP": "*"},
        {
            "LEMMA": {"IN": passiv_verbs + passive_verb_etre},
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part"]},
        },
    ]

    passive_rule_6_2 = [
        {"LEMMA": {"IN": passiv_verbs}},
    ]

    # Passif factif avec se faire
    # Exemple : Il s'est fait cambrioler sa voiture.
    """Formes passives factitives
    Dans ce type d’emplois représenté par 45 occurrences, on distinguera essentiellement les
    emplois dits « tolératifs » construits avec le semi-auxiliaire se laisser et les emplois
    proprement « factitifs » construits avec faire et parfois avec (se) voir. Cet emploi rappelle les
    « passifs canoniques » dans la possibilité qu’il a de pouvoir occulter l’agent AR2 ou de
    retarder son apparition :
    on a voulu passer le permis moto donc euh hm oui non mais on s'est fait
    rouspéter hein par lui
    """
    passive_rule_7 = [
        {
            "TAG": "PRON",
            "MORPH": {"IS_SUPERSET": ["Reflex=Yes"]},
            "OP": "+",
        },  # un pronom reflexif, facultatif
        {
            "TAG": "AUX",
            "DEP": {"IN": ["aux:tense", "aux:pass", "cop"]},
            "OP": "*",
        },  # un auxiliaire de temps
        {"TAG": "ADV", "OP": "*"},
        {
            "TAG": {"IN": ["VERB", "AUX"]},
            "LEMMA": {
                "IN": ["faire", "voir", "retrouver", "sentir", "laisser", "laisse"]
            },
        },  # verb/aux faire ou voir (laisser possible)
        {"TAG": {"IN": ["ADJ", "ADV"]}, "OP": "*"},  # petit
        {"LEMMA": {"IN": [",", "..."]}, "OP": "*"},
        {"TAG": "ADV", "OP": "*"},  # peu
        {
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": {"NOT_IN": ["exploser", "péter"] + intransitif_verbs_list},
        },  # un verbe à l'infinitif
        {"LOWER": "par", "OP": "*"},  # la proposition "par" qui est facultative
    ]

    passive_rule_7_1 = [
        {
            "TAG": "PRON",
            "MORPH": {"IS_SUPERSET": ["Reflex=Yes"]},
            "OP": "+",
        },  # un pronom reflexif,
        {
            "TAG": "AUX",
            "DEP": {"IN": ["aux:tense", "aux:pass", "cop"]},
            "OP": "*",
        },  # un auxiliaire de temps
        {
            "TAG": {"IN": ["VERB", "AUX"]},
            "LEMMA": {
                "IN": ["faire", "voir", "retrouver", "sentir", "laisser", "rester"]
            },
        },  # verb/aux faire ou voir (laisser possible)
        {"TAG": {"IN": ["ADJ", "ADV"]}, "OP": "*"},  # petit
        {"LEMMA": {"IN": [",", "..."]}, "OP": "*"},
        {"TAG": "ADV", "OP": "*"},  # peu
        {
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part"]},
        },  # un verbe au participe passé (Gramaticalement faux mais très pésent dans note corpus)
        {"LOWER": "par", "OP": "*"},  # la proposition "par" qui est facultative
    ]

    # Passif fAvec des adjectif en -ible -able -uble
    # Exemple : Il s'est fait cambrioler sa voiture.
    passive_rule_8 = [
        {
            "TAG": "ADJ",
            "TEXT": {"REGEX": r"\b(\w*(ible(s?)|able(s?)|uble(s?)))\b"},
            "LEMMA": {"NOT_IN": exception_adj},
        },  # adjectif se finnissant par ible ou able
    ]

    # Passif avec  simplement un Participe passé et la préposition par
    # Exemple : Il s'est fait cambrioler sa voiture.
    passive_rule_9 = [
        {
            "TAG": "VERB",
            "MORPH": {"IS_SUPERSET": ["Tense=Past", "VerbForm=Part"]},
            "LEMMA": {"NOT_IN": intransitif_verbs_list},
        },
        {"LOWER": "par", "OP": "+"},
    ]

    # ------------------adding rules to the matcher----------#
    matcher.add("passif_canonique", [passive_rule_0], greedy="LONGEST")
    matcher.add("passif_tronqué", [passive_rule_1], greedy="LONGEST")
    matcher.add("passif_sequencé", [passive_rule_3], greedy="LONGEST")
    matcher.add("passif_impersonel", [passive_rule_4], greedy="FIRST")
    matcher.add(
        "passif_verbale",
        [passive_rule_6, passive_rule_6_1, passive_rule_6_2],
        greedy="LONGEST",
    )
    matcher.add("passif_factif", [passive_rule_7, passive_rule_7_1], greedy="LONGEST")
    matcher.add("passif_adjectif", [passive_rule_8], greedy="LONGEST")
    matcher.add("passif_pp", [passive_rule_9], greedy="LONGEST")
    # print('Matcher is built.')

    return nlp, matcher
