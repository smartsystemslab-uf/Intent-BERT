import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def build_time_model(context_size=102+96+96, out_vector_size=96):
    input_layer = layers.Input(context_size)
    finish_time = layers.Dense(1, name='time_to_end_task')(input_layer)
    next_time = layers.Dense(1, name='time_to_next_task')(input_layer)
    next_meta = layers.Dense(out_vector_size, name='next_meta_task')(input_layer)

    model = keras.Model(input_layer, (next_meta, finish_time, next_time), name='time_only')
    return model