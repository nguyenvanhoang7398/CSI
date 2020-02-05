'''
matrix_main is used for LSTM input.
matrix_sub is used for the scoring module.
'''

from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Dense, Input, Dropout, Lambda, LSTM, Embedding, Conv1D, TimeDistributed, Add, Concatenate

from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K


def build_csi(user2ind, eid2ind, nb_feature_sub, task):

    acc=0

    nb_users = len(user2ind)
    nb_events = len(eid2ind)
    nb_features = 2+20+100    # (#temporal, #user, #doc)
    dim_hidden = 50
    text_feature_dim = 100
    news_feature_dim = text_feature_dim * 2

    ##### News (event) feature part #####
    ef_inputs = Input(batch_shape=(None, news_feature_dim)) ### news text + source text
    ef = Dense(100, activation='tanh')(ef_inputs)


    ##### Main part #####
    inputs = Input(shape=(None, nb_features))
    emb_out = TimeDistributed(Dense(100, activation='tanh'))(inputs)    # W_e
    emb_out = Dropout(0.2)(emb_out)
    rnn_out = LSTM(dim_hidden, activation='tanh', return_sequences=False)(emb_out)    #(None, dim_hidden)
    rnn_out = Dense(100, activation='tanh')(rnn_out)     # (None, 100) W_r
    rnn_out = Dropout(0.2)(rnn_out)


    ##### Sub part #####
    nb_score = 1
    nb_expand = 100
    sub_input = Input(shape=(None, nb_feature_sub + text_feature_dim))
    user_vec = TimeDistributed(Dense(nb_expand, activation='tanh',
                                     W_regularizer=regularizers.l2(0.01)))(sub_input)   # (None, None, nb_expand)
    sub_h = TimeDistributed(Dense(nb_score, activation='sigmoid'))(user_vec)    # (None, None, nb_score)
    z = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(sub_h)    #(None, nb_score)

    ##### Concatenate #####
    # out1 = Dense(1, activation='sigmoid')(rnn_out)
    # concat_out = Add()([out1, z])
    concat_out = Concatenate(axis=1)([rnn_out, z, ef])
    # concat_out = merge([rnn_out, z], mode='concat', concat_axis=1)
    # concat_out = concatenate([rnn_out, z], axis=1)

    ##### Classifier #####
    # outputs = Dense(1, activation='sigmoid')(concat_out)
    outputs = Dense(1, activation='sigmoid')(concat_out)
    # outputs = concat_out


    ##### Model #####
    hvector = Model(input=[inputs, sub_input, ef_inputs], output=concat_out)
    # zscore = Model(input=sub_input, output=sub_h)
    model = Model(input=[inputs, sub_input, ef_inputs], output=outputs)
    # uvector = Model(input=sub_input, output=user_vec)
    # model = Model(input=inputs, output=outputs)


    ##### Compile #####
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if task=="regression":
        model.compile(optimizer=adam,
                      loss='mean_squared_error')
    elif task=="classification":
        model.compile(optimizer=adam,
                      loss='binary_crossentropy')
    print("Model is compiled.")

    return model