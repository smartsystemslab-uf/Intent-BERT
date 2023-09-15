from models import build_objectless_model, build_sentence_model
from generators import get_full_train_val_gens
from Prepared_Data.dataset_params import EMBED_DETPH
# import tensorflow as tf
# import os
from typing import Union
from tensorflow import keras
import argparse
from losses import LGL, embed_loss
from losses import sparse_loss, perplexity_metric, true_recovery, top_k_recovery, cosine_loss_base
# import keras_nlp
# from generators import tokenizer



# class TopKTextGenerator(keras.callbacks.Callback):
#     """A callback to generate text from a trained model using top-k."""
#
#     def __init__(self, k):
#         self.sampler = keras_nlp.samplers.TopKSampler(k)
#         self.token_storage = tf.zeros((1, 20))
#
#     def next_prompt(self, prompt, cache, index):
#         logits = self.model(prompt)[:, index - 1, :]
#         # Ignore hidden states for now; only needed for contrastive search.
#         hidden_states = None
#         return logits, hidden_states, cache
#
#     def on_epoch_end(self, epoch, logs=None):
#         output_tokens = self.sampler(
#             next=self.next_prompt,
#             prompt=self.token_storage,
#             index=1,
#         )
#         txt = tokenizer.detokenize(output_tokens)
#         print(f"Top-K search generated text: \n{txt}\n")


def main(args, train_gen, val_gen, vocab_size, metrics:Union[list, dict, iter] = ()):
    # model = build_objectless_model()
    model = build_sentence_model(seq_len=args.seq_len, vocab_size=vocab_size, reduced_embed_dim=args.embed_dim)
    model.summary()
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, epsilon=0.01, global_clipnorm=3)
    model.compile(loss={'next_meta_pred': [sparse_loss], 'next_task_pred': [sparse_loss],
                        'time_to_end_task': ['MSLE'], 'time_to_next_task': ['MSLE'],
                        'next_meta_embed': [embed_loss], 'next_task_embed': [embed_loss]},
                  run_eagerly=False, jit_compile=False, optimizer=optimizer,
                  metrics=metrics)
    # tboard = keras.callbacks.TensorBoard()
    reduceLR = keras.callbacks.ReduceLROnPlateau()
    checkpoints = keras.callbacks.ModelCheckpoint('models/val_{val_loss}.keras')
    callbacks = [reduceLR, checkpoints]
    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks,
              steps_per_epoch=10000, validation_steps=100)
    # model.fit(train_gen, validation_data=val_gen)#, steps_per_epoch=100, validation_steps=100)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int, help="Number of instances in each batch")
    parser.add_argument('--epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--seq_len', default=10, type=int, help="Max number of tokens in a sentence")
    parser.add_argument('--embed_dim', default=128, type=int, help="Depth of embedding space")
    parser.add_argument('train_path', type=str, help='Path to training data')

    args = parser.parse_args()

    train_gen, val_gen = get_full_train_val_gens(args.train_path, 0.1, args.batch_size, args.seq_len, EMBED_DETPH)
    vocab_size = train_gen.tokenizer.vocabulary_size()



    metrics = {'next_meta_pred': ['SparseCategoricalAccuracy', perplexity_metric, true_recovery, top_k_recovery],
               'next_meta_embed': ['MSE', LGL, 'CosineSimilarity'],
               'next_task_pred': ['SparseCategoricalAccuracy', perplexity_metric, true_recovery, top_k_recovery],
               'next_task_embed': ['MSE', LGL, 'CosineSimilarity'],
               'time_to_end_task': ['MAE', 'MAPE'], 'time_to_next_task': ['MAE', 'MAPE']}

    trained_model = main(args, train_gen, val_gen, vocab_size, metrics)
    trained_model.save('../models/trained.keras')
