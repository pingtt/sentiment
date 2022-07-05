# Ref: https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
#    : https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
#    : https://keras.io/api/layers/merging_layers/dot/#dot-class

import tensorflow 
import numpy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dot, LSTM, Conv1D, MaxPooling1D, Softmax, Concatenate, Dropout, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    # The LSTM model is saved as .h5 due to a bug in tf2.4 and tf2.5 that the model cannot be loaded.
    gc = GeneratorCallback(valid_generator, setting['checkpoint'], setting['model'] + '.h5')

    n_output = 3
    n_features = embed_size

    #hidden state dimension
    n_units = 128

    # define source encoder
    encoder_inputs = Input(shape=(setting['source_time_step'], n_features))
    encoder = LSTM(n_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define target decoder
    decoder_inputs = Input(shape=(setting['target_time_step'], n_features))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    #decoder_outputs, _, _ = decoder_lstm(decoder_inputs)
    decoder_outputs = tensorflow.transpose(decoder_outputs, [0, 2, 1])


    # define attention
    # dot product = axes2=column * axes1=row ?
    dot = Dot(axes=(2, 1))
    join_outputs = dot([encoder_outputs, decoder_outputs])
    soft = Softmax()(join_outputs)
    merged = Concatenate()([encoder_outputs, soft])
    dense = Dense(32, activation='relu')(merged)

    conv = Conv1D(filters=32, kernel_size=3, activation='relu')(dense)
    pool = MaxPooling1D(pool_size=2)(conv)
    flat = Flatten()(pool)
    dense2 = Dense(16, activation='relu')(flat)
    #drop = Dropout(0.2)(dense)

    outputs = Dense(n_output, activation='softmax')(dense2)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(train_generator, validation_data=valid_generator, epochs=setting['max_epochs'],
              batch_size=setting['batch_size'], verbose=setting['verbose'], callbacks=[es, gc])

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(setting['accuracy_plot'])

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(setting['loss_plot'])

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
elif args.select == 'ensemble':

    test_generator = FastParallelGenerator(test_conf['source'], test_conf['target'], test_conf['label'],
                                           source_embed, target_embed, setting['source_time_step'],
                                           setting['target_time_step'], embed_size, shuffle=False)
    X_test, Y_test = test_generator.get_data()

    # make ensemble predictions
    # https://machinelearningmastery.com/horizontal-voting-ensemble/
    y_hyp = list()
    for model_path in ensemble:
        model = load_model(model_path)
        y_hyp.append(model.predict(X_test))
    y_hyp = numpy.array(y_hyp)

    # sum across ensemble members
    summed = numpy.sum(y_hyp, axis=0)
    # argmax across classes
    result = numpy.argmax(summed, axis=1)

    utils.save_prediction(result, setting['hypothesis'])





