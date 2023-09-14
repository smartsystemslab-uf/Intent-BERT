import numpy as np
import tensorflow as tf
import keras_nlp
from generators import tokenizer
from Prepared_Data.dataset_params import NLP_MODEL
# import spacy
# tf.config.run_functions_eagerly(True)

# nlp = spacy.load(NLP_MODEL)

mse_loss_base = tf.keras.losses.MeanSquaredError(reduction='auto')
cosine_loss_base = tf.keras.losses.CosineSimilarity(reduction='none')
sparse_loss_base = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

bleu = keras_nlp.metrics.Bleu(tokenizer=tokenizer, max_order=3, smooth=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
edit_distance = keras_nlp.metrics.EditDistance()
sampler = keras_nlp.samplers.TopPSampler(p=0.5)



def LGL(y_true, y_pred, epsilon=0.001):
    pred_corrector = tf.abs(tf.clip_by_value(y_pred, -5, 5))
    numerator = tf.abs(y_true - y_pred)
    loss = (numerator)/(tf.abs(y_true) + epsilon) + numerator/(pred_corrector + epsilon)
    return loss

def cosine_loss(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    loss = cosine_loss_base(y_true, y_pred) + 1
    weights = (0 * loss) + 1
    weights = tf.cumsum(weights, axis=1, reverse=True)
    loss = loss * weights
    return loss

def embed_loss(y_true, y_pred):
    cos_loss = tf.expand_dims(cosine_loss(y_true, y_pred), -1)
    mse = LGL(y_true, y_pred)
    return (mse * cos_loss) + mse

def max_diff_loss(y_true, y_pred):
    mse_true = tf.reduce_mean(y_true, axis=1)
    mse_pred = tf.reduce_mean(y_pred, axis=1)
    diff = mse_true - mse_pred
    max_diff = tf.abs(tf.reduce_max(diff))

    return max_diff


def combined_loss(y_true, y_pred):
    return cosine_loss(y_true, y_pred) + embed_loss(y_true, y_pred) + max_diff_loss(y_true, y_pred)

def edit_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tokenizer.detokenize(y_pred)
    # print(y_pred)
    y_true = tokenizer.detokenize(y_true)
    # print(y_true)
    return edit_distance(y_true, y_pred)

def perplexity_metric(y_true, y_pred):
    return perplexity(y_true, y_pred)

def bleu_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    return bleu(y_true, y_pred)


def sparse_loss(y_true, y_pred):
    loss = sparse_loss_base(y_true, y_pred) #+ (perplexity(y_true, y_pred)/1000)
    weights = (0.0 * loss) + 1.0
    weights = tf.cumsum(weights, axis=1, reverse=True)
    loss = loss * weights
    return loss
