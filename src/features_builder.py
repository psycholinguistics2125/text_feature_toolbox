import pandas as pd
import os, sys
import logging

from src.features import (
    analysis_graph,
    analysis_morphosynthax,
    analysis_sentiment,
    analysis_readability_stats,
    analysis_dysfluences,
    analysis_coherence,
)

from src.utils import flatten


def extract_features(
    data: pd.DataFrame,
    config: dict,
    features_choices: list,
    resources_path: str,
    _logger=logging.getLogger(),
):
    """
    Extract the features from nlp data according to features choices

    Args:
        data (pd.DataFrame): _description_
        features_choices (list): _description_
        _logger (_type_, optional): _description_. Defaults to logging.getLogger().

    Returns:
        _type_: _description_
    """

    features_names = []
    data["text"] = data["text"].apply(
        lambda x: x.replace("  ", " ")
    )  # del double spacing
    features = pd.DataFrame(index=data.index)
    for col in ["code"]:
        features[col] = data[col]

    if "passive" in features_choices:
        from src.features.passiveSRC.PassivePy import PassivePyAnalyzer
        from src.features.passiveSRC.analysis_passive import load_passive_features

        _logger.info(
            "Passive way were selected, computing all the passive way in the text data..."
        )

        analyzer = PassivePyAnalyzer(
            spacy_model=config["features"]["passive"]["passive_model"]
        )
        n_process = config["features"]["passive"]["n_process"]
        passive_features, passive_features_names = load_passive_features(
            data, analyzer, n_process=n_process
        )
        features = features.merge(
            passive_features[passive_features_names], left_index=True, right_index=True
        )
        features_names.append(passive_features_names)

    if "readability" in features_choices:
        import spacy
        import coreferee

        nlp_readability = spacy.load("fr_core_news_lg")
        nlp_readability.add_pipe("coreferee")

        _logger.info(
            "Readability features were selected, computing all the readability scores in the text data..."
        )

        _logger.info("Computing common scores for readability...")
        (
            features_CS,
            readability_features_names_CS,
        ) = analysis_readability_stats.load_common_scores(data)
        features = features.merge(
            features_CS[readability_features_names_CS],
            left_index=True,
            right_index=True,
        )
        features_names.append(readability_features_names_CS)

        _logger.info("Computing diversity score for readability")
        (
            features_Div,
            readability_features_names_Div,
        ) = analysis_readability_stats.load_diversity_score(data)
        features = features.merge(
            features_Div[readability_features_names_Div],
            left_index=True,
            right_index=True,
        )
        features_names.append(readability_features_names_Div)

        _logger.info("Computing perplexity scores for readability")
        (
            features_P,
            readability_features_names_P,
        ) = analysis_readability_stats.load_perplexity_score(data)
        features = features.merge(
            features_P[readability_features_names_P], left_index=True, right_index=True
        )
        features_names.append(readability_features_names_P)

        _logger.info("Computing externals scores for readability")
        (
            features_E,
            readability_features_names_Ext,
        ) = analysis_readability_stats.load_external_scores(data)
        features = features.merge(
            features_E[readability_features_names_Ext],
            left_index=True,
            right_index=True,
        )
        features_names.append(readability_features_names_Ext)

        _logger.info("Computing Discourse scores for readability")
        (
            features_D,
            readability_features_names_D,
        ) = analysis_readability_stats.load_discourses_scores(data, nlp_readability)
        features = features.merge(
            features_D[readability_features_names_D], left_index=True, right_index=True
        )
        features_names.append(readability_features_names_D)

        _logger.info("Computing Pqp score")
        (
            features_pqp,
            readability_features_names_pqp,
        ) = analysis_readability_stats.load_PQP_score(data, nlp_readability)
        features = features.merge(
            features_pqp[readability_features_names_pqp],
            left_index=True,
            right_index=True,
        )
        features_names.append(readability_features_names_pqp)

        _logger.info("Computing discours direct scores for readability")
        (
            features_dd,
            readability_features_names_dd,
        ) = analysis_readability_stats.load_direct_discours_score(data)
        features = features.merge(
            features_dd[readability_features_names_dd],
            left_index=True,
            right_index=True,
        )
        features_names.append(readability_features_names_dd)

    if "graph" in features_choices:
        _logger.info("Graph is in the selected features, building graph structured...")
        graph_features, graph_feature_name = analysis_graph.load_graph(data, "text")
        _logger.info("Graph features were computed ! ")
        features = features.merge(
            graph_features[graph_feature_name], left_index=True, right_index=True
        )
        features_names.append(graph_feature_name)

    if "sentiment" in features_choices:
        _logger.info(
            "Sentiment features were selected, computing sentiment features ..."
        )
        (
            sentiment_features,
            sentiment_features_name,
        ) = analysis_sentiment.load_sentiment_features(data, resources_path)
        features = features.merge(
            sentiment_features[sentiment_features_name],
            left_index=True,
            right_index=True,
        )
        features_names.append(sentiment_features_name)

    if "morph" in features_choices:
        _logger.info(
            "Morph features were selected, computing morphological features ..."
        )

        morph_features, morph_features_name = analysis_morphosynthax.load_morph(data)
        features = features.merge(
            morph_features[morph_features_name], left_index=True, right_index=True
        )
        features_names.append(morph_features_name)

    if "tag" in features_choices:
        _logger.info("Tag features were selected, computing posTag features ...")

        tag_features, tag_features_name = analysis_morphosynthax.load_tag(data)

        features = features.merge(
            tag_features[tag_features_name], left_index=True, right_index=True
        )
        features_names.append(tag_features_name)

    if "custom_ner" in features_choices:
        from src.features.analysis_ner import load_ner_features

        _logger.info("Computing custom NER model scores...")
        ner_features, ner_features_names = load_ner_features(data, _logger, config)
        features = features.merge(
            ner_features[ner_features_names], left_index=True, right_index=True
        )
        features_names.append(ner_features_names)

    if "dysfluences" in features_choices:
        _logger.info("Computing dysfluence  features")
        (
            features_dys,
            dysfluence_features_names,
        ) = analysis_dysfluences.load_dysfluences_scores(data)
        features = features.merge(
            features_dys[dysfluence_features_names], left_index=True, right_index=True
        )
        features_names.append(dysfluence_features_names)

    if "coherence" in features_choices:
        _logger.info("Computing coherence features")
        chunk_size = config["features"]["coherence"]["chunk_size"]
        model_type = config["features"]["coherence"]["model_type"]
        feature_coherence, coh_features_names = analysis_coherence.load_coherence(
            data,
            chunk_size,
            model_type,
            config,
            train=config["features"]["coherence"]["train"],
            logger=_logger,
        )
        features = features.merge(
            feature_coherence[coh_features_names], left_index=True, right_index=True
        )
        features_names.append(coh_features_names)

    features = features.fillna(0)

    return features, flatten(features_names)
