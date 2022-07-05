

import argparse
import yaml
import numpy
from nltk.tokenize import word_tokenize


class MultiSourceClassifierUtils():

    def __init__(self):
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'


    def load_conf_file(self, config_file):

        with open(config_file, "r") as reader:
            parsed_yaml = yaml.load(reader, Loader=yaml.FullLoader)

        return parsed_yaml['setting'], parsed_yaml['train'], parsed_yaml['test'],\
               parsed_yaml['validation'], parsed_yaml['ensemble']

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

    def parse_arguments(self):

        parser = argparse.ArgumentParser(description='Training a classier to parallel source/target word')

        # load the dataset
        parser.add_argument("-s", "--select", help="Select option: train, test, all", default="all")
        parser.add_argument("-c", "--config", help=".yaml configuration file")

        args = parser.parse_args()
        return args

    def read_data(self, source_file, target_file, label_file, src_sentence_length, tgt_sentence_length,
                  padding=True):
        """
        Read parallel text, and return word vectors
        :param source_file:
        :param target_file:
        :param label_file:
        :param src_sentence_length: only sentence that are more or equal to the sentence_length will be used.
        :param tgt_sentence_length: target sentence length
        :param padding: whether to pad the sentence
        :return:
        """

        source = dict()
        target = dict()
        label = dict()

        with open(source_file, 'r', encoding='utf8') as sReader, open(target_file, 'r', encoding='utf8') as tReader, \
                open(label_file, 'r', encoding='utf8') as lReader:

            sline = sReader.readline()
            tline = tReader.readline()
            lline = lReader.readline()

            id = 0
            while(sline):

                src_words = sline.strip().split(' ')
                tgt_words = tline.strip().split(' ')
                if len(src_words) < src_sentence_length and len(tgt_words) < tgt_sentence_length: 
                    # pad the sentence
                    if padding:
                        while len(src_words) < src_sentence_length:
                            src_words.append(self.PAD)

                    source[id] = src_words
                    label[id] = int(lline.rstrip())

                    # pad the sentence
                    if padding:
                        while len(tgt_words) < tgt_sentence_length:
                            tgt_words.append(self.PAD)

                    target[id] = tgt_words
                    label[id] = int(lline.rstrip())
                    id += 1


                sline = sReader.readline()
                tline = tReader.readline()
                lline = lReader.readline()

        return source, target, label


    def get_max_sentence(self, txt_file):

        with open(txt_file, 'r', encoding='utf8') as reader:
            line = reader.readline()

            sentence_length = 0
            while (line):
                src_words = line.strip().split()
                if len(src_words) > sentence_length:
                    sentence_length = len(src_words)
                line = reader.readline()

        return sentence_length



    def read_word_embedding(self, word_embedding_file, add_pad=False):
        """
        Reading Google word2vec word embedding file.
        :param word_embedding_file:
        :param add_pad:
        :return:
        """

        word_embed = dict()

        with open(word_embedding_file, 'r+', encoding='utf-8') as reader:
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

    def read_glove_embedding(self, word_embedding_file, add_pad=False, add_unk=True):
        """
        Reading Glove word embedding file
        :param word_embedding_file:
        :param add_pad:
        :param add_unk:
        :return:
        """

        word_embed = dict()
        vecs = list()

        with open(word_embedding_file, 'r+', encoding='utf-8') as reader:

            line = reader.readline().rstrip()
            while line and line != "":
                values = line.split(' ')
                vec = [float(i) for i in values[1:]]
                word_embed[values[0]] = vec
                vecs.append(vec)
                line = reader.readline().rstrip()

            embedding_size = len(word_embed.get(values[0]))
            print('Vocab: ' + str(len(vecs)))
            print('Embedding vector size: ' + str(embedding_size))

            if add_pad:
                word_embed[self.PAD] = [0.0 for i in range(embedding_size)]

            if add_unk:
                average_vec = numpy.mean(numpy.asarray(vecs, dtype=numpy.float32), axis=0)
                word_embed[self.UNK] = average_vec.tolist()

        return word_embed, embedding_size

    def calculate_unk(self, word_embedding_file):

        # Get number of vectors and hidden dim
        with open(word_embedding_file, 'r') as f:
            for i, line in enumerate(f):
                pass
        n_vec = i + 1
        hidden_dim = len(line.split(' ')) - 1

        vecs = numpy.zeros((n_vec, hidden_dim), dtype=numpy.float32)

        with open(word_embedding_file, 'r') as f:
            for i, line in enumerate(f):
                vecs[i] = numpy.array([float(n) for n in line.split(' ')[1:]], dtype=numpy.float32)

        average_vec = numpy.mean(vecs, axis=0)
        return average_vec

    def get_vocab2id(self, txt_file):

        count = 0
        vocab2id = dict()
        vocab = set()
        with open(txt_file, 'r', encoding='utf8') as sReader:
            for line in sReader:
                words = line.strip().split()
                for w in words:
                    if w not in vocab2id:
                        vocab2id[w] = count
                        count += 1
                        vocab.add(w)

        return vocab2id, vocab

    def interpolate(self, input_file1, input_file2, output_file, w1=0.5, w2=0.5):

        with open(input_file1, 'r', encoding='utf8') as r1,  open(input_file2, 'r', encoding='utf8') as r2, \
                open(output_file, 'w', encoding='utf8') as writer:

            line1 = r1.readline().rstrip()
            line2 = r2.readline().rstrip()
            while line1:
                f1 = float(line1)
                f2 = float(line2)
                value = w1 * f1 + w2 * f2
                writer.write(str(value) + '\n')
                line1 = r1.readline().rstrip()
                line2 = r2.readline().rstrip()

    def calculate_2_class_accuracy(self, hypothesis_file, reference_file):

        with open(hypothesis_file, 'r', encoding='utf8') as h_reader,  \
                open(reference_file, 'r', encoding='utf8') as r_reader:

            hyp = h_reader.readline().rstrip()
            ref = r_reader.readline().rstrip()
            count = 0
            all = 0
            while hyp:
                h_value = round(float(hyp))
                r_value = float(ref)
                if h_value == r_value:
                    count += 1

                hyp = h_reader.readline().rstrip()
                ref = r_reader.readline().rstrip()
                all += 1

            print('Accuracy: ' + str(count/all))


    def tokenize_text(self, input_file, output_file, language='english'):

        with open(input_file, 'r', encoding='utf8') as reader, open(output_file, 'w', encoding='utf8') as writer:
            line = reader.readline().rstrip()
            while line:
                tokens = word_tokenize(line, language=language)
                writer.write(' '.join(tokens).lower() + '\n')
                line = reader.readline().rstrip()

