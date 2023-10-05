import numpy as np
import logging
import gensim
from src.features.pipeline_embeddings import BuildEmbeddings
from collections.abc import MutableMapping

from src import utils
import pandas as pd


class coherenceAnalisys:
    def __init__(self, model, dictionary=None):
        self.model = model
        self.dictionary = dictionary
        self.logger_ = logging.getLogger(self.__class__.__name__)

    def _unitvec(self, v):
        return v / np.linalg.norm(v)

    def vectorize_corpus(self, corpus):
        vect_corpus = []
        if self.model.__class__ in [gensim.models.ldamulticore.LdaMulticore]:
            for sentence in corpus:
                new = self.dictionary.doc2bow(sentence)
                topics = self.model.get_document_topics(new, minimum_probability=0.0)
                vect = [topics[i][1] for i in range(self.model.num_topics)]
                vect_corpus.append([vect])

        elif self.model.__class__ == gensim.models.LsiModel:
            for sentence in corpus:
                new = self.dictionary.doc2bow(sentence)
                topics = self.model[new]
                if len(topics) > 0:
                    vect = [topics[i][1] for i in range(self.model.num_topics)]
                    vect_corpus.append([vect])

        elif self.model.__class__ == gensim.models.doc2vec.Doc2Vec:
            for sentence in corpus:
                vect = self.model.infer_vector(sentence)
                vect_corpus.append([vect])

        else:
            for sentence in corpus:
                vect_sentence = []
                for word in sentence:
                    try:
                        vect = self.model.wv[word]
                        vect_sentence.append(vect)
                    except:
                        self.logger_.info(f"the word {word} not in model ")
                    vect_corpus.append(vect_sentence)

        return vect_corpus

    def analysis_text(self, corpus, max_order=10):
        vectorized_corpus = self.vectorize_corpus(corpus)

        mean_and_len = [
            (np.mean(vec_sent, 0), len(vec_sent)) for vec_sent in vectorized_corpus
        ]

        try:
            mean_vectors_series, len_words_per_vectors = zip(
                *[t for t in mean_and_len if t[1] > 0]
            )  # we separate the mean vector from the len of the sentences
        except Exception as e:
            self.logger_.warning(f"Fail to compute mean vector because of {e}")
            return {}

        m = np.array(list(map(self._unitvec, mean_vectors_series)))  # normalisation
        max_order = min(m.shape[0], max_order)

        similarity_matrix = np.dot(m, m.T)  # distance cosinus
        similarity_orders = [np.diag(similarity_matrix, i) for i in range(1, max_order)]
        similarity_metrics = {
            "order_" + str(i): self._get_statistics(s)
            for i, s in enumerate(similarity_orders)
        }

        # normalized_coeff=[ list(map(np.mean,zip(len_words_per_vectors[:-i],len_words_per_vectors[i:]))) for i in range(1,max_order)]
        # similarity_orders_normalized = [ s/ np.array(coeff_list) for s, coeff_list in zip(similarity_orders,normalized_coeff)]
        # similarity_metrics_normalized = { 'normalized_order_'+str(i):self._get_statistics(s) for i,s in enumerate(similarity_orders_normalized) }

        # similarity_metrics.update(similarity_metrics_normalized)
        # similarity_metrics.update({ 'vector_serie_'+str(i):s for i,s in enumerate(similarity_orders)} )

        return similarity_metrics

    def _get_statistics(self, s):
        res = {"mean": np.mean(s), "std": np.std(s), "min": np.min(s), "max": np.max(s)}
        for i in range(0, 110, 10):
            res["percentile_" + str(i)] = np.percentile(s, i)
        return res


def build_chunks(token_list, n=100):
    chunks = [token_list[x : x + n] for x in range(0, len(token_list), n)]
    return chunks


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def load_coherence(
    data, chunk_size, model_type, config, train=False, logger=logging.getLogger()
):
    builder = BuildEmbeddings(config=config["embeddings"])
    if train:
        logger.info(
            f"Training a model a the selected corpus with  {len(data['token'])} documents..."
        )
        builder.train_and_save(
            corpus=data["token"].tolist(),
            method=model_type,
            model_name=config["embeddings"]["model_name"],
        )

    if model_type in ["lda", "elda", "lsi"]:
        model, c, d = builder.load_model(
            model_name=config["embeddings"]["model_name"], method=model_type
        )
    else:
        model = builder.load_model(
            model_name=config["embeddings"]["model_name"], method=model_type
        )
        d = None
    analyser = coherenceAnalisys(model=model, dictionary=d)

    data["chunks"] = data["token"].apply(lambda x: build_chunks(x, chunk_size))
    flat_dict = flatten_dict(analyser.analysis_text(data.iloc[0]["chunks"]))
    init_coherence = {elt: 0 for elt in flat_dict.keys()}
    df = pd.DataFrame(columns=list(init_coherence.keys()))
    for i in range(len(data)):
        result = flatten_dict(analyser.analysis_text(data.loc[i]["chunks"]))

        df.loc[i] = pd.Series(result)

    result = data.merge(df, left_index=True, right_index=True)
    return result, init_coherence
