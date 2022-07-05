
import pandas as pd
import numpy as np
import argparse


class ClassifierUtils:

    def __init__(self):
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'


    def save_prediction(self, hypotheses, prediction_file):
        """
        Save prediction in a file
        :param hypotheses: list, hypothesis
        :param prediction_file: str, text file
        :return: None
        """

        with open(prediction_file, 'w', encoding='utf8') as writer:
            for value in hypotheses:
                writer.write(str(value) + "\n")

    def read_data(self, train_csv, test_csv):
        """
        Read training and test feature file
        :param train_csv: training feature file. The last column is the category
        :param test_csv: test feature file. The last column is the category
        :return: training feature vectors, training class vector, test feature vectors, test class vector
        """
        df = pd.read_csv(train_csv, sep=',', header=None)
        # X_train = df.iloc[:, :-1]
        # Y_train = df[df.columns[-1]]
        X_train = df.values[:, :-1]
        Y_train = df[df.columns[-1]].to_numpy()

        df = pd.read_csv(test_csv, sep=',', header=None)
        # X_test = df.iloc[:, :-1]
        # Y_test = df[df.columns[-1]]
        X_test = df.values[:, :-1]
        Y_test = df[df.columns[-1]].to_numpy()

        return X_train, Y_train, X_test, Y_test

    def read_test_data(self, test_csv):
        """
        Read test feature file
        :param test_csv: test feature file. The last column is the category
        :return: test feature vectors, test class vector
        """""

        df = pd.read_csv(test_csv, sep=',', header=None)
        #X_test = df.iloc[:, :-1]
        #Y_test = df[df.columns[-1]]

        X_test = df.values[:, :-1]
        Y_test = df[df.columns[-1]].to_numpy()

        return X_test, Y_test



    def read_cnn_csv(self, csv_file, feat_per_word):
        """
        Read a csv file and convert to a 3D numpy arrays for 1d-cnn text classification.
        For every row in a csv, number_of_values = number_of_words * feat_per_word
        Assume the last column is the label Y.
        :param csv_file: csv file
        :param number_of_words: Number of words in a row (csv file)
        :param feat_per_word: Number of features in a word
        :return: features and label
        """

        feat_3d = list()
        label = list()

        with open(csv_file, 'r') as reader:
            line = reader.readline()
            while line and line != "":
                values = line.split(',')
                float_values = [float(i) for i in values]
                x_values = float_values[: -1]
                label.append(float_values[-1])
                words = [x_values[x:x + feat_per_word] for x in range(0, len(x_values), feat_per_word)]
                feat_3d.append(words)
                line = reader.readline()

        return np.array(feat_3d), np.array(label)


    def read_word_embedding(self, word_embedding_file, add_pad=False):

        word_embed = dict()

        with open(word_embedding_file, 'r') as reader:
            line = reader.readline()
            info = line.split(' ')
            if info is not None and len(info) == 2:
                print('Vocab: ' + info[0])
                print('Embedding vector size: ' + info[1])
            else:
                print('ERROR: Unexpected word embedding file format.')
                return None

            if add_pad:
                word_embed[self.PAD] = [0.0 for i in range(int(info[1]))]

            line = reader.readline().rstrip()
            while line and line != "":
                values = line.split(' ')
                word_embed[values[0]] = [float(i) for i in values[1:]]
                line = reader.readline().rstrip()

        return word_embed, int(info[1])



    def parse_arguments(self):

        parser = argparse.ArgumentParser(description='Training a classier to parallel source/target word')

        # load the dataset
        parser.add_argument("-t", "--train", help="The file containing features for feed forward neural network " +
                                                  "training/modeling. If 'none', then only evaluation ", default="none")
        parser.add_argument("-e", "--eval", help="The file containing features text for evaluation/prediction.")
        parser.add_argument("-m", "--model", help="The path of the model file to save/load")
        parser.add_argument("-c", "--checkpoint", help="Number of steps before checkpoint. Save only if validation result is better.", default=2000)
        parser.add_argument("-o", "--output", help="The path of the prediction file", default="prediction.txt")
        parser.add_argument("-p", "--plot", help="The path of the pyplot loss training and validation", default="figure.png")

        args = parser.parse_args()
        return args
