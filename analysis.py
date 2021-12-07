import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.tokenize import TweetTokenizer
import math
from utils import Attributes_flat


class WordSegregator():
    def __init__(self, only_nouns=False):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords.append(['.', ',', 'us', '...', '!', 'x x x x', 'x x x x x'])
        self.tknzr = TweetTokenizer()  # different tokenizer
        self.symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
        self.only_nouns=only_nouns

    def transform(self, reviews):
        corpus_review = re.sub(self.symbols, "", reviews)
        words_token = self.tknzr.tokenize(corpus_review)
        filtered_words = [w.lower() for w in words_token if not w.lower() in self.stopwords]
        if self.only_nouns is True:
            pos_tagged = nltk.pos_tag(filtered_words)
            nouns = filter(lambda x: x[1] == 'NN', pos_tagged)
            filtered_words = list(map(lambda x: x[0], nouns))
        return filtered_words


class TfIdfCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        freq_by_res = self.freq_matrix(X.text, X.id)
        Res_tf = self.create_tf_matrix(freq_by_res)
        Res_idf = self.create_idf_matrix(freq_by_res, len(X))
        Res_weight = self.create_tf_idf_matrix(Res_tf, Res_idf)
        return Res_weight

    def freq_matrix(self, token_text, review_id):
        freq_matrix = {}
        for text, key in zip(token_text, review_id):
            freq_matrix[key] = self.get_freq(text)
        return freq_matrix

    def get_freq(self, token_text):
        freq = {}
        for word in token_text:
            freq[word] = freq.get(word, 0) + 1
        return freq

    def create_tf_matrix(self, freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = sum(f_table.values())
            for word, count in f_table.items():
                #             modified here to get a better result
                tf_table[word] = np.arctan(count / count_words_in_sentence)

            tf_matrix[sent] = tf_table

        return tf_matrix

    def create_idf_matrix(self, freq_matrix, total_documents):
        idf_matrix = {}
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word])) ** 2

            idf_matrix[sent] = idf_table

        return idf_matrix

    def create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix


def first_text(business):
    text = ""
    if business.stars > 4.2:
        text = "Great! Peoplr are unstinting in their praise. But it's not time to pop the champagne. Some guests are complaining about such kind of things."
    elif business.stars > 3.2:
        text = "Mmmm....Your bussiness are in good condition. But you still need to look carefully at guest's reviews. Let's see why some have criticisms on this business."
    else:
        text = "If you don't want to close your business tomorrow, you must look carefully on the following result and make some changes now! See how much they are complaining about your business"
    return text


def second_text(business):
    text = "Besides, you can try to "
    count = 0
    attributes = Attributes_flat(business.attributes)
    if not attributes.get('GoodForKids', False):
        text += "improve you facility to make it better for kids, "
        count = 1
    if not attributes.get('ByAppointmentOnly', False):
        text += "open your seats to walk-in customers, "
        count = 1
    if not attributes.get('BusinessParking.street', False):
        text += "get some parking lot on the streets. "
        count = 1
    if count == 0:
        return ""
    return text + "That will improve your customer's rating."
