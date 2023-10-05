import warnings
#warnings.filterwarnings("ignore")

import pandas as pd
import os, logging, logging.config

from src import features_builder, utils
from src.interpret import verbatim

logging.config.dictConfig(utils.load_config("logging_config.yaml"))

def compute_features_path(config:dict)-> str:
    """

    Args:
        config (dict): _description_

    Returns:
        str: _description_
    """
    features_folder = config['data']['features_folder']
    subgroups = '_'.join(config['data']['filter']['subgroup'][1])
    n = str(config['data']["filter"]['max_number_word'])
    features_choices =  config['features']['features_choices']
    features_filename = "_".join(features_choices) + "_"+ config['data']["features_filename"]
    
    feature_path = os.path.join(features_folder, f"{n}_{subgroups}_{features_filename}")
    
    return feature_path

if __name__ =="__main__" :
    config = utils.load_config("config.yaml")
    
    logger = logging.getLogger()
    saving_path = compute_features_path(config)
    features_choices =  config['features']['features_choices']
    resources_path = config['features']["sentiments"]["resources_path"]
    filter_data = config['data']['filter_data']
    logger.info(f"Data will be saved in {saving_path}")
    logger.info(f"Lading the data..")
    nlp_data = utils.load_nlp_data(config)
    if filter_data:
        filter_data = utils.filter_data(nlp_data, config, logger=logger)
    else :
        filter_data = nlp_data
        # testing_purposes
    data = filter_data[["code","text","token","lemma","morph","pos"]].reset_index()

    data_copy = data.copy()
    features, features_name = features_builder.extract_features(data_copy, _logger = logger, config=config,features_choices=features_choices, resources_path = resources_path)
    
    features.to_csv(saving_path, sep = "\t")
    logger.info(f"features saved in {saving_path}")


    logger.info(f"Saving Verbatim for interpretations in {config['data']['interpretation_folder']}")

    verbatim.save_interpretation_lexicon(config,k=100)
    verbatim.save_interpretation_morph(config,k=100)
    verbatim.save_interpretation_truncation(config, k=100)


