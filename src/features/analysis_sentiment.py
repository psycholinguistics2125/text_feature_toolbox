import pandas as pd
import numpy as np
import logging, logging.config
import os, sys


from labMTsimple.storyLab import emotionFileReader, emotion
import liwc
from collections import Counter
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer


class SentimentAnalysis(object):
    def __init__(self, resources_path, text=None, token=None, lemma=None):
        self.text = text
        self.lemma = lemma
        self.token = token
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.resources_path = resources_path
        self.load_resources()

    def load_resources(self):
        self.load_liwc()
        self.load_empath()
        self.load_feel()
        self.load_polarimot()
        self.load_labMT()
        self.load_gobin_2017()

    def compute_emotions(self, text, lemma, token):
        liwc_result = self.compute_liwc(token)
        labMT_result = self.compute_labMT(text)
        textblob_result = self.compute_textblob(text)
        feel_result = self.compute_feel(lemma)
        polarimot_result = self.compute_polarimot(lemma)
        empath_result = self.compute_empath(lemma)
        gobin_2017_result = self.compute_gobin_2017(lemma)

        return {
            **liwc_result,
            **labMT_result,
            **textblob_result,
            **feel_result,
            **polarimot_result,
            **empath_result,
            **gobin_2017_result,
        }

    def load_liwc(self):
        try:
            liwc_path = os.path.join(
                self.resources_path, "French_LIWC2007_Dictionary.dic"
            )
            self.liwc_parser, self.liwc_category_names = liwc.load_token_parser(
                liwc_path
            )
            self.init_liwc_lexicon = {key: 0 for key in self.liwc_category_names}
            self.logger_.info("LIWC loaded succesfully !")

        except Exception as e:
            self.logger_.warning(f"Faill to load LIWC lexicon because {e}")

    def compute_liwc(self, token):
        try:
            self.token = token
            counts = dict(
                Counter(
                    category
                    for token in self.token
                    for category in self.liwc_parser(token)
                )
            )
            total = sum(counts.values(), 0.0)
            norm_counts = {k: v / total for k, v in counts.items()}
            self.logger_.info("LIWC computed successfully")
            result = {**self.init_liwc_lexicon, **dict(sorted(norm_counts.items()))}
            result = {"liwc_" + key: result[key] for key in result.keys()}
            return result

        except Exception as e:
            self.logger_.warning(f"Faill to compute LIWC because {e}")
            result = self.init_liwc_lexicon
            result = {"liwc_" + key: result[key] for key in result.keys()}
            return result

    def load_labMT(self):
        try:
            lang = "french"
            self.labMT, self.labMTvector, self.labMTwordList = emotionFileReader(
                stopval=0.0, lang=lang, returnVector=True
            )
            self.logger_.info("LabMT loaded succesfully !")
        except Exception as e:
            self.logger_.warning(f"Faill to load LabMT parser because {e}")

    def compute_labMT(self, text):
        self.text = text.lower()
        try:
            valence, fvec = emotion(
                text, self.labMT, shift=True, happsList=self.labMTvector
            )
            self.logger_.info("LabMT computed succesfully !")
            return {"labMT": valence}

        except Exception as e:
            self.logger_.warning(f"Faill to compute LabMT  because {e}")
            return {"labMT": 0}

    def compute_textblob(self, text):
        self.text = text
        try:
            polarity, subjectivity = TextBlob(
                text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()
            ).sentiment
            self.logger_.info("TextBlob measures computed succesfully !")
        except Exception as e:
            polarity, subjectivity = 0
            self.logger_.warning(
                f"Faill to compute textblob polarity and subjectivity measure because {e}"
            )
        result = {"textblob_polarity": polarity, "textblob_subjectivity": subjectivity}
        return result

    def load_feel(self):
        try:
            feel_path = os.path.join(self.resources_path, "FEEL.csv")
            feel_data = pd.read_csv(feel_path, sep=";")
            lexicon = feel_data[
                [
                    "word",
                    "joy",
                    "polarity",
                    "fear",
                    "sadness",
                    "anger",
                    "surprise",
                    "disgust",
                ]
            ].set_index("word", drop=True)
            lexicon = lexicon.drop(["bien"])
            lexicon = lexicon.to_dict(orient="index")
            new_lexicon = {}
            for word in lexicon.keys():
                affect_dict = lexicon[word]
                affect_list = []
                for emo in affect_dict.keys():
                    if emo == "polarity":
                        affect_list.append(affect_dict[emo])
                    elif affect_dict[emo] == 1:
                        affect_list.append(emo)
                new_lexicon[word] = affect_list
            self.feel_lexicon = new_lexicon
            self.init_feel_lexicon = {
                "fear": 0.0,
                "anger": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "joy": 0.0,
            }

            self.logger_.info("FEEL loaded succesfully !")

        except Exception as e:
            self.logger_.warning(f"Faill to load FEEl lexicon because {e}")

    def compute_feel(self, lemma):
        self.lemma = lemma
        try:
            result = self.build_word_affect(
                self.lemma, self.feel_lexicon, self.init_feel_lexicon
            )
            self.logger_.info(f"FEEL computed succesfully")
        except Exception as e:
            result = self.init_feel_lexicon
            self.logger_.warning(f"Faill to compute FEEL measures because {e}")
        result = {"feel_" + key: result[key] for key in result.keys()}
        return result

    def load_empath(self):
        try:
            augustin_path = os.path.join(self.resources_path, "Augustin-emo.txt")
            df_augustin = pd.read_csv(augustin_path, sep=";")
            self.init_augustin_lexicon = {
                elt: 0 for elt in df_augustin["emo"].unique().tolist()
            }
            df_augustin["emo"] = df_augustin["emo"].apply(lambda x: [x])
            self.augustin_lexicon = (
                df_augustin[["words", "emo"]]
                .groupby("words", as_index=False)
                .sum()
                .set_index("words")["emo"]
                .to_dict()
            )
            self.logger_.info("EMPATH loaded succesfully !")
        except Exception as e:
            self.logger_.warning(f"Faill to load EMPATH lexicon because {e}")

    def compute_empath(self, lemma):
        try:
            result = self.build_word_affect(
                lemma, self.augustin_lexicon, self.init_augustin_lexicon
            )
            self.logger_.info(f"EMPATH computed succesfully")
        except Exception as e:
            result = self.init_augustin_lexicon
            self.logger_.warning(f"Faill to compute EMPATH measures because {e}")
        result = {"empath_" + key: result[key] for key in result.keys()}
        return result

    def load_polarimot(self):
        try:
            polarimot_path = os.path.join(self.resources_path, "Polarimots.txt")
            df_polarimot = pd.read_csv(polarimot_path, sep=";")
            self.init_polarimot_lexicon = {
                elt: 0 for elt in df_polarimot["polarity"].unique().tolist()
            }
            df_polarimot["polarity"] = df_polarimot["polarity"].apply(lambda x: [x])
            self.polarimot_lexicon = (
                df_polarimot[["words", "polarity"]]
                .groupby("words", as_index=False)
                .sum()
                .set_index("words")["polarity"]
                .to_dict()
            )
            self.logger_.info("EMPATH loaded succesfully !")
        except Exception as e:
            self.logger_.warning(f"Faill to load POLARIMOTS lexicon because {e}")

    def compute_polarimot(self, lemma):
        try:
            result = self.build_word_affect(
                lemma, self.polarimot_lexicon, self.init_polarimot_lexicon
            )
            self.logger_.info(f"POLARIMOT computed succesfully")
        except Exception as e:
            result = self.init_polarimot_lexicon
            self.logger_.warning(f"Faill to compute POLARIMOT measures because {e}")
        result = {"polarimot_" + key: result[key] for key in result.keys()}
        return result

    def load_gobin_2017(self):
        try:
            self.gobin_2017 = pd.read_excel(
                os.path.join(self.resources_path, "ValEmo_Arous_1286.xlsx"),
                sheet_name="Open Lexicon",
            )
            self.logger_.info("Succesfully load Emo Gobin 2017")
        except Exception as e:
            self.logger_.warning(f"Fail to load Gobin 2017 because of {e}")

    def compute_gobin_2017(self, lemma):
        self.lemma = lemma
        col = [
            "Valence",
            "Arousal",
            "PCjoie",
            "PCsurprise",
            "PCcolère",
            "PCdégoût",
            "PCpeur",
            "PCtristesse",
            "PCpos",
            "PCneg",
        ]
        df_result = pd.DataFrame(columns=col)
        i = 0
        for word in lemma:
            if word in self.gobin_2017["Word"].tolist():
                df_result.loc[i] = pd.Series(
                    self.gobin_2017[self.gobin_2017["Word"] == word][col].to_dict(
                        orient="list"
                    )
                )
                i += 1
        result = dict(df_result.applymap(lambda x: x[0]).mean())
        result = {"gobin_" + key: result[key] for key in result.keys()}

        return result

    def build_word_affect(self, words, lexicon, init_lexicon):
        """ """
        # Build word affect function
        affect_list = []
        affect_dict = dict()
        affect_frequencies = Counter()
        lexicon_keys = lexicon.keys()
        for word in words:
            if word in lexicon_keys:
                affect_list.extend(lexicon[word])
                affect_dict.update({word: lexicon[word]})
        for word in affect_list:
            affect_frequencies[word] += 1
        sum_values = sum(affect_frequencies.values())
        affect_percent = init_lexicon
        for key in affect_frequencies.keys():
            affect_percent.update(
                {key: float(affect_frequencies[key]) / float(sum_values)}
            )

        return affect_percent


def load_sentiment_features(data, resources_path):
    sentiment_pipeline = SentimentAnalysis(resources_path)
    text = "exemple de text"
    token = ["exemple", "de", "text"]
    lemma = token
    result = sentiment_pipeline.compute_emotions(text, lemma, token)
    init_sentiment = list(result.keys())
    df_result = pd.DataFrame(columns=init_sentiment)

    for i in range(len(data)):
        text, token, lemma = (
            data["text"].iloc[i],
            data["token"].iloc[i],
            data["lemma"].iloc[i],
        )
        result = sentiment_pipeline.compute_emotions(text, lemma, token)

        df_result.loc[i] = pd.Series(result)

    result = data.merge(df_result, left_index=True, right_index=True)
    # normalization
    for col in init_sentiment:
        result[col] = result[col] / result["token"].apply(len) * 1000
    return result, init_sentiment
