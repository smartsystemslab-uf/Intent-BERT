import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder, PositionEmbedding
from Prepared_Data.dataset_params import EMBED_DETPH

def encoder(key, query, value, key_dim, dropout, reduced_embed_dim):
    # print(key.shape, query.shape, value.shape)
    x1 = layers.MultiHeadAttention(num_heads=4, key_dim=key_dim)(query=query, key=key, value=value)
    x1 = x1 + query
    x1 = layers.LayerNormalization()(x1)
    if dropout > 0:
        x1 = layers.Dropout(dropout)(x1)
    x2 = layers.Dense(reduced_embed_dim)(x1)
    x2 = layers.LayerNormalization()(x2)
    if dropout > 0:
        x2 = layers.Dropout(dropout)(x2)
    # print(x2.shape, x1.shape)
    x = x2 + x1
    return x

def decoder(key, query, value, key_dim, dropout, reduced_embed_dim):
    x1 = layers.MultiHeadAttention(num_heads=4, key_dim=key_dim)(query=query, key=query, value=query)
    x1 = x1 + query
    x1 = layers.LayerNormalization()(x1)
    if dropout > 0:
        x1 = layers.Dropout(dropout)(x1)
    x2 = layers.MultiHeadAttention(num_heads=4, key_dim=key_dim)(query=query, key=key, value=value)
    x2 = x1 + x2
    x2 = layers.LayerNormalization()(x2)
    if dropout > 0:
        x2 = layers.Dropout(dropout)(x2)
    x3 = layers.Dense(reduced_embed_dim)(x2)
    x3 = layers.LayerNormalization()(x3)
    if dropout > 0:
        x3 = layers.Dropout(dropout)(x3)
    return x3 + x2

def build_sentence_model(vocab_size, seq_len, pose_3_size=17, pose_2_size=17, reduced_embed_dim=1024):
    # generator X
    # X = (poses_3D, vels_3D, poses_2D, current_task['sequence_output'], current_task['pooled_output'],
    #             current_meta['sequence_output'], current_meta['pooled_output'], pose_2D_names,
    #              self.pose_3D_names)
    pose_3_input = layers.Input((pose_3_size*seq_len, 6), name='pose_3_input', dtype=tf.float32)
    vel_3_input = layers.Input((pose_3_size*seq_len, 6), name='vel_3_input', dtype=tf.float32)

    pose_3_name_input = layers.Input((pose_3_size, seq_len, EMBED_DETPH), name='pose_3_name_input', dtype=tf.float32)
    pose_3_name = layers.Reshape((-1, EMBED_DETPH))(pose_3_name_input)
    pose_3_name = layers.Dense(reduced_embed_dim)(pose_3_name)

    pose_2_input = layers.Input((None, 3), name='pose_2_input', dtype=tf.float32)
    pose_2_name_input = layers.Input((None, EMBED_DETPH), name='pose_2_name_input', dtype=tf.float32)
    pose_2_name = layers.Dense(reduced_embed_dim)(pose_2_name_input)

    task_embed_input = layers.Input((seq_len, EMBED_DETPH), name='task_feature_input', dtype=tf.float32)
    task_embed = layers.Dense(reduced_embed_dim)(task_embed_input)

    meta_embed_input = layers.Input((seq_len, EMBED_DETPH), name='meta_feature_input', dtype=tf.float32)
    meta_embed = layers.Dense(reduced_embed_dim)(meta_embed_input)

    image_feature_input = layers.Input((23, 40, 2048), name='image_feature_input', dtype=tf.float32)

    image_feature_embed = layers.Dense(reduced_embed_dim)(image_feature_input)
    image_feature_embed = layers.Reshape((-1, reduced_embed_dim))(image_feature_embed)
    image_feature_spatial = PositionEmbedding(sequence_length=23*40)(image_feature_embed)



    pose_3_encoded_task = encoder(key=pose_3_name, query=task_embed, value=pose_3_input, key_dim=reduced_embed_dim,
                                  reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    pose_3_encoded_meta = encoder(key=pose_3_name, query=meta_embed, value=pose_3_input, key_dim=reduced_embed_dim,
                                  reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    # print(pose_3_encoded_task.shape)

    vel_3_encoded_task = encoder(key=pose_3_name, query=task_embed, value=vel_3_input, key_dim=reduced_embed_dim,
                                 reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    vel_3_encoded_meta = encoder(key=pose_3_name, query=meta_embed, value=vel_3_input, key_dim=reduced_embed_dim,
                                 reduced_embed_dim=reduced_embed_dim, dropout=0.05)

    pose_2_encoded_task = encoder(key=pose_2_name, query=task_embed, value=pose_2_input, key_dim=reduced_embed_dim,
                                  reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    pose_2_encoded_meta = encoder(key=pose_2_name, query=meta_embed, value=pose_2_input, key_dim=reduced_embed_dim,
                                  reduced_embed_dim=reduced_embed_dim, dropout=0.05)

    task_encoded = encoder(key=task_embed, query=task_embed, value=task_embed, key_dim=reduced_embed_dim,
                           reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    meta_encoded = encoder(key=meta_embed, query=meta_embed, value=meta_embed, key_dim=reduced_embed_dim,
                           reduced_embed_dim=reduced_embed_dim, dropout=0.05)

    image_feature_encoded_task = encoder(key=image_feature_spatial, value=image_feature_embed, query=task_embed,
                                         key_dim=reduced_embed_dim, reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    image_feature_encoded_meta = encoder(key=image_feature_spatial, value=image_feature_embed,
                                         query=meta_embed, key_dim=reduced_embed_dim,
                                         reduced_embed_dim=reduced_embed_dim, dropout=0.05)

    combined_task = task_encoded + pose_2_encoded_task + vel_3_encoded_task + pose_3_encoded_task + image_feature_encoded_task
    combined_task = layers.LayerNormalization()(combined_task)
    combined_meta = meta_encoded + pose_2_encoded_meta + vel_3_encoded_meta + pose_3_encoded_meta + image_feature_encoded_meta
    combined_meta = layers.LayerNormalization()(combined_meta)

    total_combined = combined_task + combined_meta
    total_combined = layers.LayerNormalization()(total_combined)


    next_meta_latent = decoder(key=total_combined, query=total_combined, value=meta_embed, key_dim=reduced_embed_dim,
                               reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    next_meta_latent_out = layers.Dense(EMBED_DETPH, name='next_meta_embed')(next_meta_latent)

    next_task_latent = decoder(key=total_combined, query=next_meta_latent, value=task_embed, key_dim=reduced_embed_dim,
                               reduced_embed_dim=reduced_embed_dim, dropout=0.05)
    next_task_latent_out = layers.Dense(EMBED_DETPH, name='next_task_embed')(next_task_latent)


    # next_meta = layers.Dense(embed_depth, kernel_regularizer='l1', activation='tanh')(next_meta_latent)
    # next_meta = layers.Dropout(0.1)(next_meta_latent)
    next_meta = layers.Dense(vocab_size, name='next_meta_pred')(next_task_latent_out)


    # next_task = layers.Dense(embed_depth, kernel_regularizer='l1', activation='tanh')(next_task_latent)
    # next_task = layers.Dropout(0.1)(next_task)
    next_task = layers.Dense(vocab_size, name='next_task_pred')(next_meta_latent_out)

    combined = layers.Concatenate()([next_task_latent_out, next_meta_latent_out])
    combined = layers.Flatten()(combined)

    finish_time = layers.Dropout(0.1)(combined)
    finish_time = layers.Dense(1)(finish_time)
    finish_time = layers.LeakyReLU(name='time_to_end_task')(finish_time)

    next_start = layers.Dropout(0.1)(combined)
    next_start = layers.Concatenate()([next_start, finish_time])
    next_start = layers.Dense(1)(next_start)
    next_start = layers.LeakyReLU(name='time_to_next_task')(next_start)


    # generator X
    # X = (poses_3D, vels_3D, poses_2D, current_task['sequence_output'], current_task['pooled_output'],
    #             current_meta['sequence_output'], current_meta['pooled_output'], pose_2D_names,
    #              self.pose_3D_names)

    # Y = (next_task['sequence_output'], next_task['pooled_output'], next_meta['sequence_output'],
    #        next_meta['pooled_output'], time_to_finish, time_to_next)

    model = keras.Model((pose_3_input, vel_3_input, pose_3_name_input,
                         pose_2_input, pose_2_name_input,
                         task_embed_input,
                         meta_embed_input,
                         image_feature_input),
                        (next_task, next_meta_latent_out,
                         next_meta, next_task_latent_out, finish_time, next_start),
                        name='objectless')
    return model

def build_objectless_model(pose_size=102, token_size=96):
    pose_input = layers.Input(pose_size, name='pose_input')

    task_input = layers.Input((None, token_size), name='task_input')
    meta_input = layers.Input((None, token_size), name='meta_input')

    ragged_inputs = tf.concat([task_input, meta_input], axis=1)

    task_encoded = TransformerEncoder(token_size, 8, name='task_encode')(task_input)
    # print(task_encoded)

    meta_encoded = TransformerEncoder(token_size, 8, name='meta_encode')(meta_input)


    combined = tf.concat([task_encoded, meta_encoded], name='recombine', axis=1)


    shared_output = tf.reduce_max(combined, axis=1)
    shared_output = layers.Concatenate()([shared_output, pose_input])
    # TODO adjust decoder sequence length by latent space
    decoder_space = layers.Dense(token_size)(shared_output)
    decoder_space = layers.Reshape((1, -1))(decoder_space)


    finish_time = layers.Dense(1)(shared_output)
    finish_time = layers.LeakyReLU(name='time_to_end_task')(finish_time)

    next_meta = TransformerDecoder(token_size, token_size+pose_size, activation='tanh')(decoder_space, ragged_inputs)
    next_meta = next_meta * 2

    next_start = layers.Flatten()(next_meta)

    next_start = layers.Concatenate()([next_start, shared_output])
    next_start = layers.Dense(128, activation='tanh')(next_start)
    next_start = layers.Dense(1)(next_start)
    next_start = layers.LeakyReLU(name='time_to_next_task')(next_start)
    # TODO pass tasks as string tensor

    model = keras.Model((pose_input, task_input, meta_input), (next_meta, finish_time, next_start), name='objectless')
    return model