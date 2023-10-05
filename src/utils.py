import pandas as pd
import yaml
import os


def load_config(file_path: str = "config.yaml") -> dict:
    """
    load config file into dict python oject
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Fail to load config file because of {e}")
        config = {}
    return config


def load_nlp_data(config: dict) -> pd.DataFrame:
    """
    Load the data  from stanza

    Args:
        config (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    try:
        folder = config["data"]["nlp_folder"]
        file_path = os.path.join(folder, config["data"]["nlp_filename"])
        data = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Fail to load config file because of {e}")
        data = pd.DataFrame()
    return data


def flatten(list_of_lists: list) -> list:
    """
    Flatten a list of lists to a combined list

    Args:
        list_of_lists (list): _description_

    Returns:
        list: _description_
    """

    return [item for sublist in list_of_lists for item in sublist]


def filter_data(data, config, logger=False):
    subgroup = config["data"]["filter"]["subgroup"]
    max_len = config["data"]["filter"]["max_number_word"]
    filter_data = data[data[subgroup[0]].isin(subgroup[1])].reset_index()
    if logger:
        logger.info(f"there are {len(filter_data)} individuals in our filter dataset")
    else:
        print(f"there are {len(filter_data)} individuals in our filtered dataset")
    annotation_data = pd.read_csv(
        config["data"]["filter"]["annotation_file_manual_cut"]
    )
    filter_data["max_len"] = filter_data.apply(
        lambda x: compute_max_len(x, annotation_data), axis=1
    )
    if max_len == "manual_cut":
        for col in config["data"]["filter"]["selected_col"]:
            filter_data[col] = filter_data.apply(lambda x: cut_col(x, col), axis=1)
    else:
        for col in config["data"]["filter"]["selected_col"]:
            filter_data[col] = filter_data[col].apply(lambda x: x[:max_len])
    filter_data["text"] = filter_data["token"].apply(lambda x: " ".join(x))

    return filter_data


def compute_max_len(line, annotation_data):
    code = line["code"]
    token = line["token"]
    if code in annotation_data["code"].tolist():
        max_len = annotation_data[annotation_data["code"] == code]["token_len"].values[
            0
        ]

    else:
        max_len = len(token) / 5

    return max_len


def cut_col(line, col):
    return line[col][: int(line["max_len"])]
