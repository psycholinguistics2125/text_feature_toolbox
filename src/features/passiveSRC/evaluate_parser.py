"""
Author: Quillivic Robin
created_at: 2022 08,2022
Description: script permettant d'évaluer les perfomances du modèles.
"""

import pandas as pd


# Notre texte n'est plus labélisé après la borne fin
def cut_text(line):
    labels = line["clean_label"]
    text = line["clean_text"]
    i_cut = -1
    for label in labels:
        if label[-1] == "Fin":
            i_cut = label[1]
            break
    return text[:i_cut]


# Nous nous somme apercçu que des doubles espaces était présents dans les données labélisé
def clean_data(annotation: pd.DataFrame) -> pd.DataFrame:
    """Remove all double spacing and update the annotation start and end

    Args:
        annotation (pd.DataFrame): data entry

    Returns:
        pd.DataFrame: data with two more columns (clean_text, clean_label)
    """
    annotation["clean_text"] = annotation["text"].apply(lambda x: x.replace("  ", " "))
    clean_annotation = []
    for i in range(len(annotation)):
        ex_label = annotation["label"].iloc[i]
        ex_text = annotation["text"].iloc[i]
        clean_ex_label = []
        for ann in ex_label:
            s = ann[0]
            e = ann[1]
            label = ann[2]
            diff_e = len(ex_text[:e]) - len(
                ex_text[:e].replace("  ", " ")
            )  # count how many caraters was deleted
            diff_s = len(ex_text[:s]) - len(ex_text[:s].replace("  ", " "))
            clean_ex_label.append([s - diff_s, e - diff_e, label])
        clean_annotation.append(clean_ex_label)
    annotation["clean_label"] = clean_annotation

    return annotation


# Notre système de détection de la voix passive authorise les overlappin g antre les réglès, pour l'évaluation, nous supprimons les doublons
def find_ecart_minimum(x: int, all_start: list) -> int:
    """find the next closer element of x in the list all_start

    Args:
        x (int): _description_
        all_start (list): _description_

    Returns:
        int: minimum differences
    """
    min = 200
    for elt in all_start:
        if elt != x:
            if abs(elt - x) < min:
                min = abs(elt - x)
    return min


#
def check_if_found(s, e, test_label, seuil=30):
    for t_label in test_label:
        t_s = t_label[0]
        t_e = t_label[1]
        if abs(t_s - s) < seuil:
            return 1
        elif abs(t_e - e) < seuil:
            return 1
        else:
            continue

    return 0


def eval_one_doc(x, analyzer, seuil=40):
    """Evaluate on document by returning, the number of True Positive

    Args:
        x (_type_): _description_
        analyzer (_type_): _description_
        seuil (int, optional): _description_. Defaults to 35.

    Returns:
        _type_: _description_
    """
    example = x["cut_text"]
    true_label = x["clean_label"]
    test_label = []
    # compute the passive with automated method
    matches, doc = analyzer.prepare_visualisation(example)
    for span in doc.spans["passive"]:
        span_text = doc[span.start - 2 : span.end + 2].text
        test_label.append([span.start_char, span.end_char, span.label_, span_text])

    # removing the doublon caused by the diffferent rules
    df_test = pd.DataFrame(test_label, columns=["start", "end", "label", "text"])
    all_start = df_test["start"].tolist()
    df_test["min_ecart"] = df_test["start"].apply(
        lambda x: find_ecart_minimum(x, all_start)
    )
    # computing the number of passife we found in all document
    select = df_test[df_test["min_ecart"] > 13]
    n_test = len(select)

    # find the False positive
    select["FP_found"] = select.apply(
        lambda x: check_if_found(x.start, x.end, true_label, seuil + 20), axis=1
    )

    FP_errors = select[select["FP_found"] == 0]
    VP_test = len(select[select["FP_found"] == 1])

    # Counting the matching passive found (labelisation is not precise)
    df_eval = pd.DataFrame(true_label, columns=["start", "end", "label"])
    df_eval["ex"] = df_eval.apply(lambda x: example[x.start - 2 : x.end], axis=1)
    df_eval["is_found"] = df_eval.apply(
        lambda x: check_if_found(x.start, x.end, test_label, seuil), axis=1
    )

    n_VP = len(df_eval[df_eval["is_found"] == 1])
    FN_errors = df_eval[df_eval["is_found"] == 0]
    n_true = len(df_eval["is_found"])

    return n_VP, n_true, n_test, VP_test, FN_errors, FP_errors
