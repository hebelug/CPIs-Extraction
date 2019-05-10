MODEL = 'BGRU_GA'

import numpy as np
from tensorflow import set_random_seed
np.random.seed(1234)
set_random_seed(12345)

import os, pickle
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from LossHistory import LossHistory
from keras.layers import Dense, Input, Embedding, GRU, Bidirectional, TimeDistributed, concatenate, Dropout, BatchNormalization, Reshape, \
    RepeatVector, multiply, Permute
from keras.models import Model
from granularAtt import Granular_Attention_layer
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 70
EMBEDDING_DIM = 200
POS_EMBEDDING_DIM = 10
MODEL_DIR = './models/'
EMBED_DIR = './embedding/'

def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish)})

def build_model():
    pos_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name='pos_input1')

    word_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name='aux_input')
    word_inputE = Embedding(len(word_index) + 1, EMBEDDING_DIM, mask_zero=False, weights=[word_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH)(word_input)
    pos_input11 = RepeatVector(EMBEDDING_DIM)(pos_input1)
    pos_input11 = Permute((2, 1))(pos_input11)
    xx1 = multiply([word_inputE, pos_input11])
    x = Dropout(0.5)(xx1)
    y = Bidirectional(GRU(256, return_sequences=True, activation=swish))(x)
    att = Granular_Attention_layer()(y)
    d2 = Dropout(0.5)(att)
    d2 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(d2)
    main_output = Dense(6, kernel_regularizer=regularizers.l2(0.02), activation='softmax', name='main_output')(d2)
    model = Model(inputs=[word_input, pos_input1], outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.summary()
    return model

def train_model(model, x_train, y_train, x_dev, y_dev):
    i = 100
    dir_model = os.path.join(MODEL_DIR, MODEL)
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    filepath = dir_model + "/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
    print("The " + str(i) + "-th iteration.")
    history = LossHistory()
    model.fit({'aux_input': x_train, 'pos_input1': pos_chem_train}, {'main_output': y_train},
                     validation_data=([x_dev, pos_chem_dev], y_dev),
              epochs=i, batch_size=64, verbose=2, callbacks=[history, checkpoint,
                EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5, verbose=0, mode='min')]
              )



