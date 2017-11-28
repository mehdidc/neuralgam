from functools import partial

import numpy as np

import keras.backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers.merge import average
from keras.models import Sequential
from keras.models import Model


def build_fully_connected_gam(input_dim, 
                              output_dim=1, 
                              hidden_units=[], 
                              hidden_activation='relu', 
                              output_activation='linear', 
                              features=None):
    _build = partial(
        _build_single_fully_connected_model, 
        output_dim=output_dim,
        hidden_units=hidden_units, 
        hidden_activation=hidden_activation,
        output_activation=output_activation,
    )
    return build_gam(input_dim, _build, features=features)


def _build_single_fully_connected_model(input_dim,
                                        output_dim,
                                        hidden_units=[], 
                                        hidden_activation='relu', 
                                        output_activation='linear'):
    model = Sequential()
    input_shape = input_dim,
    for units in hidden_units:
        model.add(Dense(units, activation=hidden_activation, input_shape=input_shape))
        input_shape = (units,)
    model.add(Dense(output_dim, activation=output_activation, input_shape=input_shape))
    return model
    

def build_gam(input_dim, build_single_model_func, features=None):
    if features is None:
        features = [(i,) for i in range(input_dim)]
    models = []
    for i, feature_list in enumerate(features):
        dim = len(feature_list)
        model = build_single_model_func(dim)
        model.name = 'model_{}'.format(i)
        models.append(model)
    
    y_outs = []
    x = Input(shape=(input_dim,), name='input')
    for i, (model, feature_list) in enumerate(zip(models, features)):
        x_feat = Lambda(
            lambda m: m[:, feature_list], 
            output_shape=(dim,),
            name='component_{}'.format(i),
        )(x)
        y_out = model(x_feat)
        y_outs.append(y_out)
    y = average(y_outs)
    model = Model(x, y)
    
    _funcs = {}
    model.predict_component = {}
    for i, y_out in enumerate(y_outs):
        pred = K.function([x], y_out)
        _funcs[i] = pred
        model.predict_component[i] = lambda x:_funcs[i]([x])
    pred = K.function([x], y_outs)
    model.predict_components = lambda x:pred([x])
    return model
