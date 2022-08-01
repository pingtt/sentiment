# Ref: https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
#    : https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
#    : https://keras.io/api/layers/merging_layers/dot/#dot-class

import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dot, LSTM, Bidirectional, Dropout 
from tensorflow.keras.layers import MaxPooling1D, Conv1D, AveragePooling1D, Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np

from multisource_classifier_utils import MultiSourceClassifierUtils
from fast_parallel_data_generator import FastParallelGenerator
from generator_callback import GeneratorCallback

utils = MultiSourceClassifierUtils()

args = utils.parse_arguments()
setting, train_conf, test_conf, valid_conf, ensemble = utils.load_conf_file(args.config)
if setting['embedding_format'] == 'glove':
    source_embed, embed_size = utils.read_glove_embedding(setting['source_embedding'], add_pad=True, 
                                                          add_unk=True)
    target_embed, _ = utils.read_glove_embedding(setting['target_embedding'], add_pad=True, add_unk=True)
else:
    source_embed, embed_size = utils.read_word_embedding(setting['source_embedding'], add_pad=True)
    target_embed, _ = utils.read_word_embedding(setting['target_embedding'], add_pad=True)


if args.select == 'train' or args.select == 'all':
    # Train the model using generator
    train_generator = FastParallelGenerator(train_conf['source'], train_conf['target'], train_conf['label'],
                                            source_embed, target_embed, setting['source_time_step'],
                                            setting['target_time_step'], embed_size, setting['batch_size'])

    test_generator = FastParallelGenerator(test_conf['source'], test_conf['target'], test_conf['label'],
                                           source_embed, target_embed, setting['source_time_step'],
                                           setting['target_time_step'], embed_size, setting['batch_size'], shuffle=False)

    valid_generator = FastParallelGenerator(valid_conf['source'], valid_conf['target'], valid_conf['label'],
                                            source_embed, target_embed, setting['source_time_step'],
                                            setting['target_time_step'], embed_size, setting['batch_size'], shuffle=False)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    # checkpoint
    gc = GeneratorCallback(valid_generator, setting['checkpoint'], setting['model'] + '.h5')

    n_output = 3 
    n_features = embed_size

    #hidden state dimension
    enc_units = 128
    dec_units = 128
 
    # define source encoder
    encoder_inputs = Input(shape=(setting['source_time_step'], n_features))
    eforward_layer = LSTM(enc_units, return_sequences=True)
    ebackward_layer = LSTM(enc_units, activation='relu', return_sequences=True,
                           go_backwards=True)
    encoder = Bidirectional(eforward_layer, backward_layer=ebackward_layer)
    encoder_outputs = encoder(encoder_inputs)

    # define target decoder
    decoder_inputs = Input(shape=(setting['target_time_step'], n_features))
    decoder_lstm = LSTM(dec_units, return_sequences=True, return_state=True)
    decoder_outputs, d_state_h, d_state_c = decoder_lstm(decoder_inputs)

    flat = Flatten()(encoder_outputs)
    dense = Dense(32, activation='relu')(flat)
    dense2 = Dense(8, activation='relu')(dense)

    outputs = Dense(n_output, activation='softmax')(dense2)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(train_generator, validation_data=valid_generator, epochs=setting['max_epochs'],
              verbose=setting['verbose'], callbacks=[es, gc])

    X_test, Y_test = test_generator.get_data()
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=setting['batch_size'], verbose=0)
    print("Last Epoch Model: ")
    print('Accuracy: %.2f' % (accuracy * 100))
    prob = model.predict(X_test)
    hyp = np.argmax(prob, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro')
    print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))

    utils.save_prediction(prob, setting['hypothesis'])

elif args.select == 'test':

    test_generator = FastParallelGenerator(test_conf['source'], test_conf['target'], test_conf['label'],
                                           source_embed, target_embed, setting['source_time_step'],
                                           setting['target_time_step'], embed_size, setting['batch_size'], shuffle=False)

    X_test, Y_test = test_generator.get_data()

    model = load_model(setting['model'] + '.h5')
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=setting['batch_size'], verbose=0)
    print("Best Epoch Model: ")
    print('Accuracy: %.2f' % (accuracy * 100))

    # evaluate the keras model
    prob = model.predict(X_test)
    hyp = np.argmax(prob, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro')
    print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))

    utils.save_prediction(prob, setting['hypothesis'])

