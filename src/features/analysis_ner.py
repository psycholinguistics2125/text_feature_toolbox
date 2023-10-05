import sys
import pandas as pd
import os
import spacy
import torch

from src.utils import flatten
from thinc.api import set_gpu_allocator, require_gpu


def make_one_prediction(
    text: str, model: spacy.Language, init_dic: dict, use_gpu: bool = True
) -> list:
    """
    Predict the entities in the text, using the model

    Args:
        text (str): _description_
        model (spacy.Language): _description_

    Returns:
        list: _description_
    """
    doc = model(text)
    ent_list = []
    for ent in doc.ents:
        ent_list.append((ent.text, ent.start, ent.end, ent.label_))

    # store result in dict
    df_ent = pd.DataFrame(ent_list, columns=["text", "start", "end", "label"])
    result = {**init_dic, **df_ent["label"].value_counts()}

    # empty memory of gpu
    doc._.trf_data = None
    if use_gpu:
        torch.cuda.empty_cache()

    return df_ent, result


def add_one_ner_features(data, task_name, logger, config):
    logger.info("Seting up GPU..")

    if config["features"]["ner"]["use_gpu"]:
        set_gpu_allocator("pytorch")
        require_gpu()

    task_labels = config["features"]["ner"]["task_labels"][f"{task_name}_labels"]
    model_path = os.path.join(
        config["features"]["ner"]["models_folder"], f"model_{task_name}", "model-best"
    )

    init_dic = {elt: 0 for elt in task_labels}

    predictions = pd.DataFrame(columns=list(init_dic.keys()))

    logger.info(f"Loading_model from {model_path}")
    model = spacy.load(model_path)
    logger.info(f"Model loaded ! ")

    logger.info(f"Beginning the inference of {(len(data))} documents ..")

    for i in range(len(data)):
        line = data.iloc[i]
        code = line["code"]

        text = line["text"]
        try:
            df_ents, ents = make_one_prediction(
                text,
                model,
                init_dic=init_dic,
                use_gpu=config["features"]["ner"]["use_gpu"],
            )
            predictions.loc[i] = pd.Series(ents)
        except Exception as e:
            logger.error(f"{code} NOT done because of {e}, continuing...")
            predictions.loc[i] = pd.Series(init_dic)
            continue

    result = data.merge(predictions, left_index=True, right_index=True)

    return result, task_labels


def load_ner_features(data: pd.DataFrame, logger, config) -> pd.DataFrame:
    task_names = config["features"]["ner"]["task_names"]
    logger.info(f"Computing ner for {str(task_names)}")

    names = []
    for task_name in task_names:
        data, name = add_one_ner_features(data, task_name, logger, config)
        names.append(name)

    # normalisation
    for name in flatten(names):
        data["n"] = data["token"].apply(len)
        data[name] = data[name] / data["n"] * 100
    return data, flatten(names)
