from models import build_time_model
from generators import DataGenerator, get_train_val_gens
import tensorflow as tf
import os
from tensorflow import keras
import argparse

def main(args, train_gen, val_gen):
    model = build_time_model()
    model.summary()
    model.compile(loss='MSE', run_eagerly=False, jit_compile=False)
    # model.fit(train_gen, use_multiprocessing=True, workers=4)
    model.fit(train_gen, validation_data=val_gen) #, steps_per_epoch=100, validation_steps=100)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int, help="number of instances in each batch")
    parser.add_argument('train_path', type=str, help='Path to training data')

    args = parser.parse_args()

    train_gen, val_gen = get_train_val_gens(args.train_path, 0.1, args.batch_size)
    trained_model = main(args, train_gen, val_gen)
    trained_model.save('../models/trained.h5')