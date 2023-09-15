from models import build_sentence_model
from generators import get_full_train_val_gens, TextAsStringGenerator
from Prepared_Data.dataset_params import EMBED_DETPH
from typing import Union
from tensorflow import keras
import argparse
from losses import LGL, embed_loss
from losses import sparse_loss, perplexity_metric, true_recovery, top_k_recovery, cosine_loss_base, perfect_recovery




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


def main(args, train_gen, metrics:Union[list, dict, iter] = (), val_gen=None, resume: str=None):
    # model = build_objectless_model()
    vocab_size = train_gen.tokenizer.vocabulary_size()
    if resume is None:
        model = build_sentence_model(seq_len=args.seq_len, vocab_size=vocab_size, reduced_embed_dim=args.embed_dim)
    else:
        model = keras.models.load_model(resume, compile=False)
    model.summary()
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, epsilon=0.01, global_clipnorm=3)
    model.compile(loss={'next_meta_pred': [sparse_loss], 'next_task_pred': [sparse_loss],
                        'time_to_end_task': ['MSLE'], 'time_to_next_task': ['MSLE'],
                        'next_meta_embed': [embed_loss], 'next_task_embed': [embed_loss]},
                  run_eagerly=False, jit_compile=False, optimizer=optimizer,
                  metrics=metrics)
    # tboard = keras.callbacks.TensorBoard()
    
    
    if val_gen is not None:
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
        checkpoints = keras.callbacks.ModelCheckpoint('models/val_{val_loss}.keras')
        callbacks = [reduceLR, checkpoints]
        model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks,
                  steps_per_epoch=10000, validation_steps=100)
    else:
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss')
        checkpoints = keras.callbacks.ModelCheckpoint('models/loss_{loss}.keras')
        callbacks = [reduceLR, checkpoints]
        model.fit(train_gen, epochs=args.epochs, callbacks=callbacks)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--seq_len', default=10, type=int, help="Max number of tokens in a sentence")
    parser.add_argument('--embed_dim', default=128, type=int, help="Depth of embedding space")
    parser.add_argument('--full_train', default=True, type=bool, help="Train on the full training set")
    parser.add_argument('--resume_path', default=None, type=str, help="Path to paritally trained model")
    parser.add_argument('--split', default=0.1, type=float, help="Split to use when training")
    parser.add_argument('train_path', type=str, help='Path to training data')

    args = parser.parse_args()

    metrics = {'next_meta_pred': ['SparseCategoricalAccuracy', perplexity_metric, true_recovery, top_k_recovery(3), perfect_recovery],
               'next_meta_embed': ['MSE', 'MAPE', LGL, 'CosineSimilarity'],
               'next_task_pred': ['SparseCategoricalAccuracy', perplexity_metric, true_recovery, top_k_recovery(3), perfect_recovery],
               'next_task_embed': ['MSE', 'MAPE', LGL, 'CosineSimilarity'],
               'time_to_end_task': ['MAE', 'MAPE'], 'time_to_next_task': ['MAE', 'MAPE']}

    if not args.full_train:
        train_gen, val_gen = get_full_train_val_gens(args.train_path, split=args.split, batch_size=1, seq_dim=args.seq_len,
                                                     embed_dim=EMBED_DETPH)
        trained_model = main(args, train_gen, metrics, val_gen, resume=args.resume_path)
    else:
        train_gen = TextAsStringGenerator(args.train_path, 1, seq_len=args.seq_len, embed_dim=EMBED_DETPH)
        trained_model = main(args, train_gen, metrics, resume=args.resume_path)


    trained_model.save('models/trained.keras')
