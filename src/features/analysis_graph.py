"""
Auhtor :  Quillivic Robin
inspired by this repo : https://github.com/facuzeta/speechgraph 
"""
# -*- coding: utf8 -*-
from collections import Counter
import re
import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
import pandas as pd
from src import utils


class _graphStatistics:
    def __init__(self, graph):
        self.graph = graph

    def statistics(self):
        res = {}
        graph = self.graph
        res["number_of_nodes"] = graph.number_of_nodes()
        res["number_of_edges"] = graph.number_of_edges()
        res["PE"] = (
            np.array(list(Counter(graph.edges()).values())) > 1
        ).sum()  # nombre de noeud s
        res["LCC"] = nx.algorithms.components.number_weakly_connected_components(
            graph
        )  # nombre de noeuds faiblement connectés
        res["LSC"] = nx.algorithms.components.number_strongly_connected_components(
            graph
        )  # nombre de noeux connectés fortement
        degrees = list(dict(graph.degree()).values())
        res["degree_average"] = np.mean(degrees)
        res["degree_std"] = np.std(degrees)

        # adj_matrix = nx.linalg.adj_matrix(graph).toarray()
        adj_matrix = nx.adjacency_matrix(graph).toarray()
        adj_matrix2 = np.dot(adj_matrix, adj_matrix)
        adj_matrix3 = np.dot(adj_matrix2, adj_matrix)

        res["L1"] = np.trace(adj_matrix)  # number of triangle in the graph
        res["L2"] = np.trace(adj_matrix2)
        res["L3"] = np.trace(adj_matrix3)

        # to slow
        graph = nx.Graph(graph)
        """
        # small world analysis
        res['sigma'] = nx.algorithms.smallworld.sigma(graph)
        res['omega'] = nx.algorithms.smallworld.omega(graph)
        """
        # clustering
        res["transitivity"] = nx.transitivity(graph)
        res["average_clustering"] = nx.average_clustering(graph)

        # compute larger connected_graph
        Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph0 = graph.subgraph(Gcc[0])
        res["diameter_g0"] = nx.diameter(graph0)
        res["average_shrotest_path_g0"] = nx.average_shortest_path_length(graph0)

        return res

    def draw_graph(self):
        G = self.graph
        nx.draw(G, with_labels=True, font_weight="bold")
        plt.show()


class naiveGraph(object):
    def __init__(self):
        self.logger_ = logging.getLogger(self.__class__.__name__)

    def _text2graph(self, token_list):
        gr = nx.MultiDiGraph()
        try:
            gr.add_edges_from(zip(token_list[:-1], token_list[1:]))
            self.logger_.info("Text grpah built succesfully")
        except Exception as e:
            self.logger_.warning(f"Fail to build text graph because of {e}")

        return gr

    def analyzeText(self, token_list, draw=False):
        dgr = self._text2graph(token_list)
        if draw:
            _graphStatistics(dgr).draw_graph()

        return _graphStatistics(dgr).statistics()


def convert_tag(tags):
    tag_list = []
    for tag in tags:
        tag_list.append(tag[1])
    return tag_list


def load_graph(data, col):
    if col == "tag":
        data["tag"] = data["pos"].apply(convert_tag)
    elif col == "text":
        col = "token"
    # init
    g = naiveGraph()
    init_graph = {}
    for k in g.analyzeText(["Je", "suis", "un", "exemple", "pour", "commencer"]).keys():
        init_graph[k] = 0
    df = pd.DataFrame(columns=list(init_graph.keys()))
    # compute graph
    for i in range(len(data)):
        result = g.analyzeText(data.iloc[i][col])
        df.loc[i] = pd.Series(result)

    result = data.merge(df, left_index=True, right_index=True)
    return result, init_graph


def get_X_y(data, col, ptsd=False):
    result, ini_graph = load_graph(data, col)

    if ptsd == True:
        result["exposition"] = data["PTSD"].apply(utils.encode_ptsd)
    else:
        result["exposition"] = data["cercle"].apply(lambda x: 1 if x == 1 else 0)

    X = result[ini_graph.keys()]
    y = result["exposition"]

    return X, y


if __name__ == "__main__":
    import os

    config = utils.load_config_file()
    saving_folder = os.path.join(config["local"]["1000_data_folder"], "features")
    print("#loading ptsd data ..")
    data_ptsd = utils.load_merged_data_ptsd(
        method="spacy_trf", part="1", select=["text", "token", "lemma", "pos", "morph"]
    )
    # data_ptsd = data_ptsd.loc[:2]
    print("# build lemma graph")
    X, y = get_X_y(data_ptsd, col="lemma", ptsd=True)
    X.to_csv(os.path.join(saving_folder, "X_ptsd_graph_lemma.csv"))
    y.to_csv(os.path.join(saving_folder, "y_ptsd_graph_lemma.csv"))

    print("# build pos graph")
    X, y = get_X_y(data_ptsd, col="tag", ptsd=True)
    X.to_csv(os.path.join(saving_folder, "X_ptsd_graph_tag.csv"))
    y.to_csv(os.path.join(saving_folder, "y_ptsd_graph_tag.csv"))

    print("#loading all data ..")
    data = utils.load_merged_data(
        method="spacy_trf", part="1", select=["text", "token", "lemma", "pos", "morph"]
    )
    # data = data.loc[:2]
    print("# build lemma graph")
    X, y = get_X_y(data, col="lemma", ptsd=False)
    X.to_csv(os.path.join(saving_folder, "X_graph_lemma.csv"))
    y.to_csv(os.path.join(saving_folder, "y_graph_lemma.csv"))

    print("# build pos graph")
    X, y = get_X_y(data, col="tag", ptsd=False)
    X.to_csv(os.path.join(saving_folder, "X_graph_tag.csv"))
    y.to_csv(os.path.join(saving_folder, "y_graph_tag.csv"))

    print("done !")
