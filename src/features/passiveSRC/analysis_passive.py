""" 
Author: Quillivic Robin
created_at : 2022, 15th Nov
Description: Quelques fonctions suppoort pour l'extraction de la voix passive
"""

import numpy as np
import pandas as pd
import math

from src.features.passiveSRC.PassivePy import PassivePyAnalyzer
from thinc.api import set_gpu_allocator, require_gpu


def load_model_from_name(model_name: str):
    """ """
    if "spacy" in model_name:
        import spacy

        name = "_".join(model_name.split("_")[1:])
        if "trf" in name:
            import torch

            torch.multiprocessing.set_start_method("spawn")
            set_gpu_allocator("pytorch")
            require_gpu()
            print("using gpu...")

        nlp = spacy.load(name)

    elif "stanza" in model_name:
        import spacy_stanza

        package = model_name.split("_")[1]
        nlp = spacy_stanza.load_pipeline("fr", package=package, disable=["mwt"])
    else:
        nlp = None
        return "The model name you enter in not valid ! "
    return nlp


def load_passive_features(
    data: pd.DataFrame, analyzer: PassivePyAnalyzer, n_process=-1
):
    """_summary_

    Args:
        data (pd.DataFrame): Données d'entrer avec une colonne texte
        analyzer (PassivePyAnalyzer): passive extrateur

    Returns:
        df (pd.DataFrame):dataframes containing the passive information
        cols_passive (list): liste des colonnes
    """
    # verify there are not double spacing that makes spacy going mad
    data["text"] = data["text"].apply(lambda x: x.replace("  ", " "))
    if n_process == 1:
        batch_size = 1
    else:
        batch_size = 50
    # compute all passive form in the corpus (might take a while)
    df = analyzer.match_corpus_level(
        data, "text", n_process=n_process, batch_size=batch_size, add_other_columns=True
    )
    df["text"] = df["document"]
    df["word_number"] = df["token"].apply(len)
    df["passive_count_norm"] = df["passive_count"] / df["word_number"].apply(math.log)
    cols_passive = [
        "count_sents",
        "all_passives",
        "passive_count",
        "passive_count_norm",
        "passive_sents_count",
        "passive_percentages",
        "binary",
    ]

    return df, cols_passive


def load_from_text(text: str, analyzer: PassivePyAnalyzer) -> pd.DataFrame:
    """Returns the passive feature from the text

    Args:
        text (str): _description_
        analyzer (PassivePyAnalyzer): _description_

    Returns:
        result (pd.DataFrame): résultats d'analyses
    """

    result = analyzer.match_text(text)

    return result.T


ex_passive = [
    "elle a été déterminée et formée ! ",
    "accusé par la police, il fut condamné a 5 ans de prison ferme",
    "ma mère a été élevée par une euh une une vieille dame euh à euh dans l'Allier",
    "puis on a été soutenu par un monsieur aussi c'est monsieur NPERS…",
    "la la République du Centre a été rachetée euh par un journal euh je voudrais pas dire de bêtises je crois que c'est Clermont-Ferrand Clermont enfin…",
    "alors il y a des écoles qui sont faites exclusivement par des religieuses ou certains par des prêtres mais alors maintenant ils ont beaucoup d'institutrices euh civiles qui sont dedans",
    "donc il est il est payé quoi bah c'est super",
    "à à mon avis ça a déjà été fait mais au niveau euh professionnel au niveau du travail je crois que ça se développe ouais",
    "quand tu habites dans le centre d'Orléans euh c'est quand même réservé à un public qui a … un minimum d'argent",
    "parce que même avant on a été coincé euh pendant pas mal d'années euh sans voiture et on travaillait pas",
    "tu es à peine payé et tout quoi",
    "tout à l'heure j'avais été inter- la dernière fois que j'avais été interviewée c'était pour la la construction de la nouvelle fac",
    "donc vous êtes  envoyée ponctuellement sur des sur des sur des sites sites",
    "d'accord je devais pas rester à Orléans je devais juste être formée pendant un an et partir sur Lille",
    "je trouve que pour le moment euh c'est pas encore trop bien aménagé euh pour les vélos hein vélos",
    "Les adultes quand tu habites dans le centre d'Orléans euh c'est quand même réservé à un public qui a hm ouais ouais bien sûr ouais un minimum d'argent"
    "des  des échanges qui qui se font pas par téléphone",
    "on a voulu passer le permis moto donc euh hm oui non mais on s'est fait rouspéter hein par lui",
    "Deux matins de suite qu'on se fait réveiller par la perceuse hein",
    "non mais il faut réussir à se faire respecter c'est ça le plus dur",
    "oui oui y aura toujours des professeurs qui se feront chahuter par leurs élèves",
    "genre euh genre si tu te fais je te le souhaite pas ah si tu te fais larguer par ton copain … tu vas dire ah c'est chips"
    "c'est que euh tu peux pas te balader euh tranquille quoi tu te fais emmerder par euh par un … par un tas de gens"
    "il a été fait toute une série de réformes euh successives hâtives euh non raisonnées non efficaces et finalement l'ensemble du pays était mécontent",
    "il a été décidé de qu'on soit transféré dans l'autre école av- avec l'idée de faire une école maternelle primaire commune",
    "il a été décidé au niveau national de faire une enquête des de toutes ces industries pour euh savoir d'où venait la source",
    "cet espace est resté vacant… pendant longtemps sauf les fêtes de la musique… il a été demandé de pouvoir l’utiliser",
]
