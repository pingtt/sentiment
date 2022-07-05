# Ref: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
#      https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import classifier_utils
import os
import pickle


class ParallelTxtDataGenerator(Sequence):
    '''
    Generates parallel text data for Keras. Required word embeddings to convert words to vectors
    '''

    def __init__(self, src_file, tgt_file, label_file, data_dir, src_embed_file, tgt_embed_file, batch_size=32,
                 sentence_length=50, n_classes=2, shuffle=True, overwrite_data=True):
        '''
        Initialization
        :param src_file: source parallel text
        :param tgt_file: target parallel text
        :param label_file: label/class
        :param data_dir: the records in src_file and tgt_file will be converted to .npy to allow
        easy access during training.
        :param src_embed_file: embedding file train using word2vec
        :param tgt_embed_file: embedding file train using word2vec
        :param batch_size:
        :param sentence_length: maximum number of words in a sentence
        :param n_classes:
        :param shuffle:
        '''

        self.sentence_length = sentence_length
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.label_file = data_dir + '/label.pkl'

        if src_embed_file is not None or tgt_embed_file is not None:
            self.utils = classifier_utils.ClassifierUtils()
            source_embed, self.embed_size = self.utils.read_word_embedding(src_embed_file, add_pad=True)
            target_embed, _ = self.utils.read_word_embedding(tgt_embed_file, add_pad=True)
        else:
            source_embed = None
            target_embed = None

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        # convert text to feature if directory is empty
        if overwrite_data is True or len(os.listdir(self.data_dir)) == 0:
            last_id, self.label = self.save_txt_as_npy(src_file, tgt_file, label_file, self.data_dir, source_embed,
                                                       target_embed, sentence_length)
            self.list_IDs = list(range(last_id))
            print(self.list_IDs)
        else:
            self.list_IDs = os.listdir(self.src_dir).sort()
            reader = open(self.label_file, 'r')
            self.label = pickle.load(reader)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_src, X_tgt, Y = self.__data_generation(list_IDs_temp)

        return X_src, X_tgt, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_src = np.empty((self.batch_size, self.sentence_length, self.embed_size))
        X_tgt = np.empty((self.batch_size, self.sentence_length, self.embed_size))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            filename = self.data_dir + '/' + str(ID) + '.npy'
            record = np.load(filename)

            X_src[i,] = record[0]
            X_tgt[i,] = record[1]
            Y[i] = self.label[ID]
            #print(str(self.label[ID]))
            #print(str(X_tgt[i, ]))

        return X_src, X_tgt, Y

    def save_txt_as_npy(self, src_file, tgt_file, label_file, data_dir, src_embed, tgt_embed,
                        sentence_length=50, start_id=0):
        '''
        Save each line as a .npy file for easy access during training
        :param src_file:
        :param tgt_file:
        :param label_file:
        :param data_dir:
        :param tgt_dir:
        :param sentence_length: if more than the sentence length, the sentence will be skipped.
        :param start_id:
        :return:
        '''

        current_id = start_id
        # we will need to save the label separately
        label = dict()
        with open(src_file, 'r') as src_reader, open(tgt_file, 'r') as tgt_reader, open(label_file, 'r') as lbl_reader:
            sline = src_reader.readline().rstrip()
            tline = tgt_reader.readline().rstrip()
            lline = lbl_reader.readline().rstrip()
            while sline:
                src_words = sline.split(' ')
                tgt_words = tline.split(' ')

                if len(src_words) <= sentence_length and len(tgt_words) <= sentence_length:
                    # pad the sentence
                    while len(src_words) < sentence_length:
                        src_words.append(self.utils.PAD)

                    while len(tgt_words) < sentence_length:
                        tgt_words.append(self.utils.PAD)

                    src_vec = list()
                    for word in src_words:
                        embed_vec = src_embed.get(word)
                        if embed_vec is not None:
                            src_vec.append(embed_vec)
                        else:
                            src_vec.append(src_embed.get(self.utils.UNK))

                    tgt_vec = list()
                    for word in tgt_words:
                        embed_vec = tgt_embed.get(word)
                        if embed_vec is not None:
                            tgt_vec.append(embed_vec)
                        else:
                            tgt_vec.append(tgt_embed.get(self.utils.UNK))

                    label[current_id] = int(lline)

                    np_words = np.array([src_vec, tgt_vec], dtype='float')
                    filename = data_dir + '/' + str(current_id) + '.npy'
                    np.save(filename, np_words)

                    current_id += 1
                sline = src_reader.readline().rstrip()
                tline = tgt_reader.readline().rstrip()
                lline = lbl_reader.readline().rstrip()

        with open(data_dir + '/label.pkl', 'wb') as writer:
            pickle.dump(label, writer)

        return current_id, label
