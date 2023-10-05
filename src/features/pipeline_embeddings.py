"""
Author :  Quillivic Robin
# todo :
- implement LDA2vec: https://github.com/cemoody/lda2vec 
- add pretrained loading embeding (transformers etc.)
- add transformer embedings
- add spacy pre-trained
"""

from asyncio.log import logger
import os
import logging

from gensim.models import (
    FastText,
    Word2Vec,
    LdaMulticore,
    doc2vec,
    LsiModel,
    EnsembleLda,
    HdpModel,
)
from gensim import corpora

import shutil

current_folder = os.getcwd()


class BuildEmbeddings(object):
    def __init__(self, config):
        self.config = config
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.saving_folder = config["saving_folder"]

    def train_model(self, corpus, method="fasttext"):
        """
        corpus is a list of tokenized text
        """
        self.method = method
        self.model = None
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(
            no_below=self.config["no_below"], no_above=self.config["no_above"]
        )
        self.corpora = [self.dictionary.doc2bow(doc) for doc in corpus]

        if self.method == "doc2vec":
            try:
                model = doc2vec.Doc2Vec(**self.config[self.method])
                tagged_corpus = [
                    doc2vec.TaggedDocument(words=_d, tags=[str(i)])
                    for i, _d in enumerate(corpus)
                ]
                model.build_vocab(tagged_corpus)
                model.train(
                    tagged_corpus,
                    total_examples=model.corpus_count,
                    epochs=model.epochs,
                )
                self.model = model
                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        elif self.method == "fasttext":
            try:
                model = FastText(**self.config[self.method])  # instantiate
                print(model)
                model.build_vocab(corpus)
                print(model)
                model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
                self.model = model
                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        elif self.method == "word2vec":
            try:
                model = Word2Vec(**self.config[self.method])  # instantiate
                model.build_vocab(corpus)
                model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
                self.model = model
                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        elif self.method == "lsi":
            try:
                model = LsiModel(
                    self.corpora, id2word=self.dictionary, **self.config["lsi"]
                )
                self.model = model

                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        elif self.method == "lda":
            try:
                model = LdaMulticore(
                    self.corpora, id2word=self.dictionary, **self.config["lda"]
                )
                self.model = model

                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )
        elif self.method == "elda":
            try:
                model = EnsembleLda(
                    corpus=self.corpora, id2word=self.dictionary, **self.config["elda"]
                )
                model = model.generate_gensim_representation()
                self.model = model

                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        elif self.method == "hdp":
            try:
                model = HdpModel(
                    corpus=self.corpora, id2word=self.dictionary, **self.config["hdp"]
                )
                self.model = model

                self.logger_.info(f"{self.method} model trained successfully !")
            except Exception as e:
                self.logger_.warning(
                    f"Fail to train {self.method} model because of {e} "
                )

        else:
            self.logger_.warning(
                """ The method you specified is not supported yet !
                supported method are : fasttext,word2vec,doc2vec, lda, lsi, elda, hdp """
            )

    def save(self, model_name=None):
        if model_name == None:
            model_name = self.config["model_name"]
        if self.model is None:
            self.logger_.warning(
                "You are about to save a model that is not instancidated"
            )
            return False
        else:
            folder = os.path.join(self.saving_folder, self.method)
            if not os.path.exists(folder):
                os.mkdir(folder)
            spe_folder = os.path.join(folder, model_name)
            try:
                shutil.copy(
                    os.path.join(current_folder, "config.yaml"),
                    os.path.join(spe_folder, "config.yaml"),
                )
            except Exception as e:
                self.logger_.warning(f"fail to save the config file because of {e}")
            if not os.path.exists(spe_folder):
                os.mkdir(spe_folder)
            if self.method in ["fasttext", "word2vec", "doc2vec"]:
                self.model.save(os.path.join(spe_folder, model_name + ".model"))
                self.logger_.info(f"{model_name} Model saved in {spe_folder}")
                return True
            elif self.method in ["lda", "lsi", "elda", "hdp"]:
                self.model.save(os.path.join(spe_folder, model_name + ".model"))
                corpora.MmCorpus.serialize(
                    os.path.join(spe_folder, model_name + ".mm"), self.corpora
                )
                self.dictionary.save(os.path.join(spe_folder, model_name + ".dict"))
                self.logger_.info(f"{model_name} Model saved in {spe_folder}")
                return True
            else:
                return False

    def train_and_save(self, corpus, method, model_name):
        self.train_model(corpus=corpus, method=method)
        self.save(model_name)

    def load_model(self, model_name, method):
        model_path = os.path.join(
            self.saving_folder, method, model_name, model_name + ".model"
        )
        dictionary = None
        corpus = None

        if method == "fasttext":
            model = FastText.load(model_path)

        elif method == "word2vec":
            model = Word2Vec.load(model_path)

        elif method == "doc2vec":
            model = doc2vec.Doc2Vec.load(model_path)

        elif method == "lsi":
            model = LsiModel.load(model_path)
            dictionary = corpora.Dictionary.load(model_path.replace(".model", ".dict"))
            corpus = corpora.MmCorpus(model_path.replace(".model", ".mm"))

        elif method == "lda":
            model = LdaMulticore.load(model_path)
            dictionary = corpora.Dictionary.load(model_path.replace(".model", ".dict"))
            corpus = corpora.MmCorpus(model_path.replace(".model", ".mm"))

        elif method == "hdp":
            model = HdpModel.load(model_path)
            dictionary = corpora.Dictionary.load(model_path.replace(".model", ".dict"))
            corpus = corpora.MmCorpus(model_path.replace(".model", ".mm"))

        elif method == "elda":
            model = LdaMulticore.load(model_path)
            dictionary = corpora.Dictionary.load(model_path.replace(".model", ".dict"))
            corpus = corpora.MmCorpus(model_path.replace(".model", ".mm"))
        else:
            self.logger_.warning("The method specified is not implemented.")
            model = None

        if dictionary != None:
            return model, corpus, dictionary
        else:
            return model


# Exemple
"""
from pipeline_embeddings import BuildEmbeddings
from utils import load_config_file
config = load_config_file()['embeddings']
builder = BuildEmbeddings(config=config)
builder.train_and_save(corpus = test, method='lda',model_name = 'test')
model, c, d = builder.load_model(model_name='test',method='lda')

"""
