from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import numpy as np


class FeatureExtraction:

    def __init__(self):

        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def malay_stop_words(self):

        return ['yang', 'ini', 'itu', '.', ',', ':', ';', '-', '?', '\'', '"', '(', ')', '!']

    def extract_count_feat(self, sentence_per_doc_file, stop_word_language='english'):

        reader = open(sentence_per_doc_file, "r", encoding='utf8')
        docs = reader.readlines()
        reader.close()

        if stop_word_language == 'english':
            vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
        elif stop_word_language == 'malay':
            vectorizer = CountVectorizer(stop_words=self.malay_stop_words(), ngram_range=(1, 1))

        X = vectorizer.fit_transform(docs)

        return X.toarray(), vectorizer.get_feature_names()

    def extract_tfidf_feat(self, sentence_per_doc_file):

        transformer = TfidfTransformer(smooth_idf=False)
        feat_counts, feat_names = self.extract_count_feat(sentence_per_doc_file)
        tfidf = transformer.fit_transform(feat_counts)
        print(feat_names)

        return tfidf.toarray(), feat_names

    def extract_txt_tfidf_feat(self, txt_file, label_file):

        X_source, _ = self.extract_tfidf_feat(txt_file)

        X = list()
        for i, arr in enumerate(X_source):
            values = arr.tolist()
            X.append(values)

        print('ok!')
        Y = list()
        with open(label_file, 'r', encoding='utf8') as reader:
            for value in reader:
                Y.append(float(value))

        return np.array(X), np.array(Y)

