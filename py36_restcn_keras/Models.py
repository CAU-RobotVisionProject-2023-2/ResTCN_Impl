import numpy as np
import sys
# sys.path.append('/home-2/jhou16@jhu.edu/.local/lib/python3.6/site-packages')

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.layers.merge import add,concatenate
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial
import pdb

def TCN_simple(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           kernel_regularizer=l2(1.e-4),
           activation="relu"):
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  config = [ 
             [(1,8,64)]
           ]
  
  input = Input(shape=(max_len,feat_dim))
  model = input
  
  for depth in range(0,len(config)):
    for stride,filter_dim,num in config[depth]:
      conv = Conv1D(num, 
                    filter_dim,
                    strides=stride,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=kernel_regularizer)(model)
      bn = BatchNormalization(axis=CHANNEL_AXIS)(conv)
      dr = Dropout(dropout)(bn)
      model = dr

  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           strides=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(units=n_classes, 
                activation="softmax",
                kernel_initializer="he_normal")(flatten)

  model = Model(inputs=input, outputs=dense)
  return model


def TCN_plain(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           kernel_regularizer=l2(1.e-4),
           activation="relu"):
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  config = [ 
             [(1,8,64)]
           ]
  
  input = Input(shape=(max_len,feat_dim))
  model = input
  
  for depth in range(0,len(config)):
    for stride,filter_dim,num in config[depth]:
      conv = Conv1D(num, 
                    filter_dim,
                    strides=stride,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=kernel_regularizer)(model)
      bn = BatchNormalization(axis=CHANNEL_AXIS)(conv)
      relu = Activation(activation)(bn)
      dr = Dropout(dropout)(relu)
      model = dr

  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           strides=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(units=n_classes, 
                activation="softmax",
                kernel_initializer="he_normal")(flatten)

  model = Model(inputs=input, outputs=dense)
  return model


def TCN_resnet(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           kernel_regularizer=l1(1.e-4),
           activation="relu"):
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  config = [ 
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(2,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(2,8,256)],
             [(1,8,256)],
             [(1,8,256)]
           ]
  initial_stride = 1
  initial_filter_dim = 8 # kernel size
  initial_num = 64 # filters
  # input shape -> 300 150
  # input_channel = 150
  
  # kernel_shape -> (8, 1) + (150//1, 64) -> 

  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Conv1D(initial_num, 
                 initial_filter_dim,
                 strides=initial_stride,
                 padding="same",
                 kernel_initializer="he_normal",
                 kernel_regularizer=kernel_regularizer)(model)
  cnt = 0
  # print(cnt, 'conv', K.int_shape(model)); cnt += 1
  
  for depth in range(0,len(config)):
    for stride,filter_dim,num in config[depth]:
      bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
      # print(cnt, 'bn', bn.get_weights()); cnt += 1
      relu = Activation(activation)(bn)
      # print(cnt, 'relu', K.int_shape(relu)); cnt += 1
      dr = Dropout(dropout)(relu)
      # print(cnt, 'dr', K.int_shape(dr)); cnt += 1
      res = Conv1D(num, 
                   filter_dim,
                   strides=stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(dr)
      # print(cnt, 'res', K.int_shape(res)); cnt += 1

      res_shape = K.int_shape(res)
      model_shape = K.int_shape(model)
      if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
        model = Conv1D(num, 
                       1,
                       strides=stride,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=kernel_regularizer)(model)
        
        # print(cnt, 'downsample', K.int_shape(model)); cnt += 1

      model = add([model,res])
  # print(cnt, 'skip conn', K.int_shape(model)); cnt += 1

  bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
  # print(cnt, 'bn', K.int_shape(bn)); cnt += 1
  model = Activation(activation)(bn)
  # print(cnt, 'relu', K.int_shape(model)); cnt += 1

  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           strides=1)(model)
    # print(cnt, 'gap', K.int_shape(gap)); cnt += 1
    
    flatten = Flatten()(gap)
    # print(cnt, 'flat', K.int_shape(flatten)); cnt += 1
  else:
    flatten = Flatten()(model)
  dense = Dense(units=n_classes, 
                activation="softmax",
                kernel_initializer="he_normal")(flatten)
  # print(cnt, 'dense', K.int_shape(dense)); cnt += 1

  model = Model(inputs=input, outputs=dense)
  return model


def TCN_simple_resnet(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           kernel_regularizer=l1(1.e-4),
           activation="relu"):
  
  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1

  config = [ 
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(2,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(2,8,256)],
             [(1,8,256)],
             [(1,8,256)]
           ]
  initial_stride = 1
  initial_filter_dim = 8
  initial_num = 64
  
  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Conv1D(initial_num, 
                 initial_filter_dim,
                 strides=initial_stride,
                 padding="same",
                 kernel_initializer="he_normal",
                 kernel_regularizer=kernel_regularizer)(model)

  for depth in range(0,len(config)):
    for stride,filter_dim,num in config[depth]:
      bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
      dr = Dropout(dropout)(bn)
      res = Conv1D(num, 
                   filter_dim,
                   strides=stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(dr)

      res_shape = K.int_shape(res)
      model_shape = K.int_shape(model)
      if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
        model = Conv1D(num, 
                       1,
                       strides=stride,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=kernel_regularizer)(model)

      model = add([model,res])

  model = BatchNormalization(axis=CHANNEL_AXIS)(model)

  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           strides=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(units=n_classes, 
                activation="softmax",
                kernel_initializer="he_normal")(flatten)

  model = Model(inputs=input, outputs=dense)
  return model

