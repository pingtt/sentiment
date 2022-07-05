
import numpy as np

from tensorflow.keras.utils import Sequence
from parallel_data_generator import ParallelTxtDataGenerator
from multisource_classifier_utils import MultiSourceClassifierUtils
from tensorflow.keras.utils import to_categorical

class FastParallelGenerator(Sequence):
    '''
    Generates parallel text data for Keras. Required word embeddings to convert words to vectors. Instead of
    saving the record in disk, we load the data into memory.
    '''

    def __init__(self, src_file, tgt_file, label_file, src_embed, tgt_embed, src_sentence_length, tgt_sentence_length,
                 embed_size, batch_size=32, n_classes=3, shuffle=True):
        '''
        Initialization
        :param src_file: source parallel text
        :param tgt_file: target parallel text
        :param label_file: label/class
        :param src_embed: embedding dict train using word2vec
        :param tgt_embed: embedding dict train using word2vec
        :param src_sentence_length: maximum number of words in a sentence
        :param tgt_sentence_length: maximum number of words in a sentence
        :param embed_size: size of embedding vector
        :param batch_size:
        :param n_classes:
        :param shuffle:
        '''

        self.src_sentence_length = src_sentence_length
        self.tgt_sentence_length = tgt_sentence_length
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.source_embed = src_embed
        self.target_embed = tgt_embed
        self.embed_size = embed_size

        self.utils = MultiSourceClassifierUtils()
        self.source, self.target, self.label = self.utils.read_data(src_file, tgt_file, label_file,
                                                                    src_sentence_length, tgt_sentence_length)
        self.list_IDs =  list(self.source.keys())

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
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_src = np.empty((self.batch_size, self.src_sentence_length, self.embed_size))
        X_tgt = np.empty((self.batch_size, self.tgt_sentence_length, self.embed_size))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            src_words= self.source[ID]
            tgt_words = self.target[ID]

            src_vec = list()
            for word in src_words:
                embed_vec = self.source_embed.get(word)
                if embed_vec is not None:
                    src_vec.append(embed_vec)
                else:
                    src_vec.append(self.source_embed.get(self.utils.UNK))

            tgt_vec = list()
            for word in tgt_words:
                embed_vec = self.target_embed.get(word)
                if embed_vec is not None:
                    tgt_vec.append(embed_vec)
                else:
                    tgt_vec.append(self.target_embed.get(self.utils.UNK))

            X_src[i,] = np.array(src_vec, dtype='float')
            X_tgt[i,] = np.array(tgt_vec, dtype='float')
            Y[i] = self.label[ID]
            #print(str(self.label[ID]))
            #print(str(X_tgt[i, ]))

        return [X_src, X_tgt], to_categorical(Y, num_classes=self.n_classes)


    def get_some_data(self, number_batches):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_src = np.empty((self.batch_size * number_batches, self.src_sentence_length, self.embed_size))
        X_tgt = np.empty((self.batch_size * number_batches, self.tgt_sentence_length, self.embed_size))
        Y = np.empty((self.batch_size * number_batches), dtype=int)

        list_IDs_temp = np.arange(0, self.batch_size * number_batches, 1)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            src_words = self.source[ID]
            tgt_words = self.target[ID]

            src_vec = list()
            for word in src_words:
                embed_vec = self.source_embed.get(word)
                if embed_vec is not None:
                    src_vec.append(embed_vec)
                else:
                    src_vec.append(self.source_embed.get(self.utils.UNK))

            tgt_vec = list()
            for word in tgt_words:
                embed_vec = self.target_embed.get(word)
                if embed_vec is not None:
                    tgt_vec.append(embed_vec)
                else:
                    tgt_vec.append(self.target_embed.get(self.utils.UNK))

            X_src[i,] = np.array(src_vec, dtype='float')
            X_tgt[i,] = np.array(tgt_vec, dtype='float')
            Y[i] = self.label[ID]
            # print(str(self.label[ID]))
            # print(str(X_tgt[i, ]))

        return [X_src, X_tgt], to_categorical(Y, num_classes=self.n_classes)


    def get_data(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_src = np.empty((len(self.list_IDs), self.src_sentence_length, self.embed_size))
        X_tgt = np.empty((len(self.list_IDs), self.tgt_sentence_length, self.embed_size))
        Y = np.empty((len(self.list_IDs)), dtype=int)

        # Generate data
        for i, ID in enumerate(self.list_IDs):
            src_words= self.source[ID]
            tgt_words = self.target[ID]

            src_vec = list()
            for word in src_words:
                embed_vec = self.source_embed.get(word)
                if embed_vec is not None:
                    src_vec.append(embed_vec)
                else:
                    src_vec.append(self.source_embed.get(self.utils.UNK))

            tgt_vec = list()
            for word in tgt_words:
                embed_vec = self.target_embed.get(word)
                if embed_vec is not None:
                    tgt_vec.append(embed_vec)
                else:
                    tgt_vec.append(self.target_embed.get(self.utils.UNK))

            X_src[i,] = np.array(src_vec, dtype='float')
            X_tgt[i,] = np.array(tgt_vec, dtype='float')
            Y[i] = self.label[ID]
            #print(str(self.label[ID]))
            #print(str(X_tgt[i, ]))

        return [X_src, X_tgt], to_categorical(Y, num_classes=self.n_classes)


    def get_data_X(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        print('Dataset Size: ' + str(len(self.list_IDs)))
        X_src = np.empty((len(self.list_IDs), self.src_sentence_length, self.embed_size))
        X_tgt = np.empty((len(self.list_IDs), self.tgt_sentence_length, self.embed_size))

        # Generate data
        for i, ID in enumerate(self.list_IDs):
            src_words= self.source[ID]
            tgt_words = self.target[ID]

            src_vec = list()
            for word in src_words:
                embed_vec = self.source_embed.get(word)
                if embed_vec is not None:
                    src_vec.append(embed_vec)
                else:
                    src_vec.append(self.source_embed.get(self.utils.UNK))

            tgt_vec = list()
            for word in tgt_words:
                embed_vec = self.target_embed.get(word)
                if embed_vec is not None:
                    tgt_vec.append(embed_vec)
                else:
                    tgt_vec.append(self.target_embed.get(self.utils.UNK))

            X_src[i,] = np.array(src_vec, dtype='float')
            X_tgt[i,] = np.array(tgt_vec, dtype='float')

        return [X_src, X_tgt]
