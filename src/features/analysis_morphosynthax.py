import pandas as pd
from collections import Counter
from src import utils


def count(tags, init):
    tag_dict = Counter(tags)
    total = sum(tag_dict.values(), 0.0)
    for key in tag_dict.keys():
        tag_dict[key] /= total
        tag_dict[key] = tag_dict[key]
    return {**init, **tag_dict}


def transform_morph_dict(morph_dict, tag):
    converter = {}
    for key, value in morph_dict.items():
        # Hyperbase rewrite rules
        hvalue = value
        if hvalue == "Yes":  # Poss, Reflex, Foreign, Abbr, Typo
            hvalue = key
        if hvalue == "Number":
            key = ""
        if hvalue == "Imp":
            if key == "Mood":
                hvalue = "Impe"  # Impératif
            elif key == "Tense":
                hvalue = "Impf"  # Imparfait
            elif key == "Aspect":
                hvalue = "Impa"  # imperfect aspect
            try:
                hvalue = str(int(hvalue)) + "pers"  # Personnes
            except:
                pass

        converter[tag + key + value] = tag + ":" + hvalue
    return converter


def convert_morph(x):
    final_list = []
    for elt in x:
        if elt != False:
            morph = elt[2].split("|")
            tag = elt[1]
            if morph != [""]:
                morph_dict = {}
                for elt in morph:
                    key, value = elt.split("=")
                    morph_dict[key] = value

                t = transform_morph_dict(morph_dict, tag)
                morph_list = t.values()
                final_list.append(morph_list)
        else:
            final_list.append([])
    return utils.flatten(final_list)


def convert_tag(tags):
    tag_list = []
    for tag in tags:
        tag_list.append(tag[1])
    return tag_list


def load_tag(data):
    df = pd.DataFrame(columns=list(init_pos.keys()))
    data["tag"] = data["pos"].apply(convert_tag)
    for i in range(len(data)):
        result = count(data.iloc[i]["tag"], init_pos)
        df.loc[i] = pd.Series(result)

    result = data.merge(df, left_index=True, right_index=True)

    return result, list(init_pos.keys())


def count_first_personal_pronoun_sing(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Person=1" in elt[2])
                and ("PronType=Prs" in elt[2])
                and ("Number=Sing" in elt[2])
            ):
                select.append(elt[0])
            if elt[1] == "PRON":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_first_personal_pronoun_plur(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Person=1" in elt[2])
                and ("PronType=Prs" in elt[2])
                and ("Number=Plur" in elt[2])
            ):
                select.append(elt[0])
            if elt[1] == "PRON":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_second_personal_pronoun(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Person=2" in elt[2])
                and ("PronType=Prs" in elt[2])
                and (elt[1] == "PRON")
            ):
                select.append(elt[0])
            if elt[1] == "PRON":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_second_personal_pronoun(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Person=2" in elt[2])
                and ("PronType=Prs" in elt[2])
                and (elt[1] == "PRON")
            ):
                select.append(elt[0])
            if elt[1] == "PRON":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_third_personal_pronoun(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Person=3" in elt[2])
                and ("PronType=Prs" in elt[2])
                and ("Gender" in elt[2])
                and (elt[1] == "PRON")
            ):
                select.append(elt[0])
            if elt[1] == "PRON":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_indicatif_present(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Mood=Ind" in elt[2])
                and ("Tense=Pres" in elt[2])
                and (elt[1] == "VERB")
            ):
                select.append(elt[0])
            if elt[1] == "VERB":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_indicatif_future(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Mood=Ind" in elt[2])
                and ("Tense=Fut" in elt[2])
                and (elt[1] == "VERB")
            ):
                select.append(elt[0])
            if elt[1] == "VERB":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_indicatif_imparfait(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Mood=Ind" in elt[2])
                and ("Tense=Imp" in elt[2])
                and (elt[1] == "VERB")
            ):
                select.append(elt[0])
            if elt[1] == "VERB":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_paticitipe_passe(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if (
                ("Tense=Past" in elt[2])
                and ("VerbForm=Part" in elt[2])
                and (elt[1] == "VERB")
            ):
                select.append(elt[0])
            if elt[1] == "VERB":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def count_conditionel(x):
    select = []
    ref = []
    for elt in x:
        if elt:
            if ("Mood=Cnd" in elt[2]) and (elt[1] == "VERB"):
                select.append(elt[0])
            if elt[1] == "VERB":
                ref.append(elt[0])
    # print(select)
    return len(select) / len(ref)


def load_morph(data):
    data["first_personal_pronoun_sing"] = data["morph"].apply(
        count_first_personal_pronoun_sing
    )
    data["first_personal_pronoun_plur"] = data["morph"].apply(
        count_first_personal_pronoun_plur
    )
    data["second_personal_pronoun"] = data["morph"].apply(count_second_personal_pronoun)
    data["third_personal_pronoun"] = data["morph"].apply(count_third_personal_pronoun)
    data["verb_indicatif_present"] = data["morph"].apply(count_indicatif_present)
    data["verb_indicatif_future"] = data["morph"].apply(count_indicatif_future)
    data["verb_participe_passe"] = data["morph"].apply(count_paticitipe_passe)
    data["verb_conditionel"] = data["morph"].apply(count_conditionel)
    data["verb_indicatif_imparfait"] = data["morph"].apply(count_indicatif_imparfait)

    morph_cols = [
        "first_personal_pronoun_sing",
        "first_personal_pronoun_plur",
        "second_personal_pronoun",
        "third_personal_pronoun",
        "verb_indicatif_present",
        "verb_indicatif_future",
        "verb_participe_passe",
        "verb_conditionel",
        "verb_indicatif_imparfait",
    ]

    return data, morph_cols


"""def load_morph(data):
    df = pd.DataFrame(columns= list(init_morph.keys()))
    data['transform_morph'] = data['morph'].apply(convert_morph)
    for i in  range(len(data)) :

        result = count(data.iloc[i]['transform_morph'],init_morph)
    
        df.loc[i] = pd.Series(result)
    
    result = data.merge(df,left_index=True, right_index=True)
    #normalisation
    col_list = list(init_morph.keys())
    for col in col_list : 
        result[col] = result[col]/result['token'].apply(len)*1000
    return result, col_list"""

init_morph = {
    "ADJ:Fem": 0,
    "ADJ:Masc": 0,
    "ADJ:Ord": 0,
    "ADJ:Plur": 0,
    "ADJ:Sing": 0,
    "ADP:Art": 0,
    "ADP:Def": 0,
    "ADP:Masc": 0,
    "ADP:Plur": 0,
    "ADP:Sing": 0,
    "ADV:Int": 0,
    "ADV:Neg": 0,
    "AUX:1": 0,
    "AUX:2": 0,
    "AUX:3": 0,
    "AUX:Cnd": 0,
    "AUX:Fin": 0,
    "AUX:Fut": 0,
    "AUX:Impf": 0,
    "AUX:Ind": 0,
    "AUX:Inf": 0,
    "AUX:Part": 0,
    "AUX:Past": 0,
    "AUX:Plur": 0,
    "AUX:Pres": 0,
    "AUX:Sing": 0,
    "AUX:Sub": 0,
    "DET:Art": 0,
    "DET:Def": 0,
    "DET:Dem": 0,
    "DET:Fem": 0,
    "DET:Ind": 0,
    "DET:Int": 0,
    "DET:Masc": 0,
    "DET:Plur": 0,
    "DET:Poss": 0,
    "DET:Sing": 0,
    "NOUN:Card": 0,
    "NOUN:Fem": 0,
    "NOUN:Masc": 0,
    "NOUN:Plur": 0,
    "NOUN:Sing": 0,
    "NUM:Card": 0,
    "NUM:Masc": 0,
    "PRON:1": 0,
    "PRON:2": 0,
    "PRON:3": 0,
    "PRON:Card": 0,
    "PRON:Dem": 0,
    "PRON:Fem": 0,
    "PRON:Int": 0,
    "PRON:Masc": 0,
    "PRON:Plur": 0,
    "PRON:Prs": 0,
    "PRON:Reflex": 0,
    "PRON:Rel": 0,
    "PRON:Sing": 0,
    "PROPN:Fem": 0,
    "PROPN:Masc": 0,
    "PROPN:Plur": 0,
    "PROPN:Sing": 0,
    "VERB:1": 0,
    "VERB:2": 0,
    "VERB:3": 0,
    "VERB:Cnd": 0,
    "VERB:Fem": 0,
    "VERB:Fin": 0,
    "VERB:Fut": 0,
    "VERB:Impe": 0,
    "VERB:Impf": 0,
    "VERB:Ind": 0,
    "VERB:Inf": 0,
    "VERB:Masc": 0,
    "VERB:Part": 0,
    "VERB:Pass": 0,
    "VERB:Past": 0,
    "VERB:Plur": 0,
    "VERB:Pres": 0,
    "VERB:Sing": 0,
    "VERB:Sub": 0,
}

init_pos = {
    "ADP": 0,
    "NOUN": 0,
    "SPACE": 0,
    "ADV": 0,
    "PUNCT": 0,
    "DET": 0,
    "NUM": 0,
    "PRON": 0,
    "VERB": 0,
    "PROPN": 0,
    "SCONJ": 0,
    "AUX": 0,
    "X": 0,
    "CCONJ": 0,
    "ADJ": 0,
    "CONJ": 0,
    "INTJ": 0,
    "SYM": 0,
    "PART": 0,
}
