import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import json
from tqdm import tqdm
import sys

def get_train_val_gens(train_path, split=0.1, batch_size=1):
    temp_gen = DataGenerator(train_path, batch_size)
    num_val = int(split * len(temp_gen))
    train_set = temp_gen.data_list[num_val:]
    val_set = temp_gen.data_list[:num_val]

    train_gen = DataGenerator(train_path, batch_size)
    train_gen.data_list = train_set
    train_gen.count = len(train_set)
    train_gen.on_epoch_end()

    val_gen = DataGenerator(train_path, batch_size)
    val_gen.data_list = val_set
    val_gen.count = len(val_set)
    val_gen.on_epoch_end()


    return train_gen, val_gen

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_path, batch_size=1, shuffle=True):
        'Initialization'
        self.data_list = []
        for session in tqdm(os.listdir(root_path)):
            for frame in os.listdir(os.path.join(root_path, session)):
                full_path = os.path.join(root_path, session, frame)
                self.data_list.append(full_path)

        self.count = len(self.data_list)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.count)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.data_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        current_contexts = np.empty((self.batch_size, 102+96+96)) # 198 = 102 poses + 2 x 96 vector embedding
        time_to_finish = np.empty((self.batch_size))
        time_to_next = np.empty((self.batch_size))
        next_task_metas = np.empty((self.batch_size, 96))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            temp_f = open(ID, 'r')
            current_context = json.load(temp_f)
            temp_f.close()
            try:
                poses = np.asarray_chkfinite(current_context['poses'])
                current_embed = np.asarray_chkfinite(current_context['current_task_embed'])
                current_meta_embed = np.asarray_chkfinite(current_context['current_task_meta'])
                next_task_meta = np.asarray_chkfinite(current_context['next_task_meta'])
            except ValueError:
                print(ID)
                sys.exit()

            combined_context = np.concatenate([poses, current_embed, current_meta_embed])

            current_contexts[i] = combined_context
            time_to_next[i] = current_context['time_to_next']
            time_to_finish[i] = current_context['time_to_end']
            next_task_metas[i] = next_task_meta

        return current_contexts, (next_task_metas, time_to_finish, time_to_next)


if __name__ == "__main__":
    tester = DataGenerator(os.path.join("Prepared_Data","training"))
    print(tester.__getitem__(0))