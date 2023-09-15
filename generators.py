import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import json
from tqdm import tqdm
import sys
import keras_nlp
from Prepared_Data.dataset_params import train_set, BERT_MODEL, BODY_PART_LIST_3D, BODY_PART_LIST_2D, SEQ_LEN, EMBED_DETPH
tokenizer = keras_nlp.models.BertTokenizer.from_preset(BERT_MODEL)

def get_objectless_train_val_gens(train_path, split=0.1, batch_size=1):
    val_sessions = int(split*len(train_set.split(', ')))
    train_sessions = train_set[:val_sessions]
    val_sessions = train_set[val_sessions:]
    print('Train Sessions', train_sessions)
    print('Val Sessions', val_sessions)

    train_gen = TextAsStringGenerator(train_path, batch_size, ignore_sessions=val_sessions)

    val_gen = TextAsStringGenerator(train_path, batch_size, ignore_sessions=train_sessions)

    return train_gen, val_gen

def get_full_train_val_gens(train_path, split=0.1, batch_size=1, seq_dim=20, embed_dim=1024):
    all_train_sessions = train_set.split(', ')
    val_sessions = int(split * len(all_train_sessions)) + 1
    train_sessions = all_train_sessions[val_sessions:]
    val_sessions = all_train_sessions[:val_sessions]
    print('Train Sessions', train_sessions)
    print('Val Sessions', val_sessions)

    train_gen = TextAsStringGenerator(train_path, batch_size, seq_len=seq_dim, embed_dim=embed_dim, ignore_sessions=val_sessions)

    val_gen = TextAsStringGenerator(train_path, batch_size, seq_len=seq_dim, embed_dim=embed_dim, ignore_sessions=train_sessions)

    return train_gen, val_gen


class PreparedDataGenerator(keras.utils.Sequence):
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
        current_embeds = np.array([])
        current_metas = np.array([])
        time_to_finish = np.array([])
        time_to_next = np.array([])
        next_task_metas = np.array([])
        next_task_embeds = np.array([])
        poses = np.array([])
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            temp_f = open(ID, 'r')
            current_context = json.load(temp_f)
            temp_f.close()
            try:
                pose = np.asarray_chkfinite(current_context['poses'])
                current_embed = np.asarray_chkfinite(current_context['current_task_embed'])
                current_meta_embed = np.asarray_chkfinite(current_context['current_task_meta'])
                next_task_meta = np.asarray_chkfinite(current_context['next_task_meta'])
                next_task_embed = np.asarray_chkfinite(current_context['next_task_embed'])
            except ValueError:
                print(ID)
                sys.exit()

            pose = pose[np.newaxis, :]
            poses = np.vstack([poses, pose]) if poses.size else pose

            current_embed = current_embed[np.newaxis, :]
            current_embeds = np.vstack([current_embeds, current_embed]) if current_embeds.size else current_embed

            current_meta_embed = current_meta_embed[np.newaxis, :]
            current_metas = np.vstack([current_metas, current_meta_embed]) if current_metas.size else current_meta_embed

            temp = np.asarray_chkfinite(current_context['time_to_next'])
            temp = temp.reshape((1, 1))
            time_to_next = np.vstack([time_to_next, current_context['time_to_next']]) if time_to_next.size else temp

            temp = np.asarray_chkfinite(current_context['time_to_end'])
            temp = temp.reshape((1, 1))
            time_to_finish = np.vstack([time_to_finish, current_context['time_to_end']]) if time_to_finish.size else temp

            next_task_meta = next_task_meta[np.newaxis, :]
            next_task_metas = np.vstack([next_task_metas, next_task_meta]) if next_task_metas.size else next_task_meta

            next_task_embed = next_task_embed[np.newaxis, :]
            next_task_embeds = np.vstack([next_task_embeds, next_task_embed]) if next_task_embeds.size else next_task_embed

        # print(current_embeds.shape, current_metas.shape, 'gen')
        return (poses, current_embeds, current_metas),\
               (next_task_metas, time_to_finish, time_to_next)


class TextAsStringGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_path, batch_size=1, shuffle=True, seq_len=SEQ_LEN, embed_dim=128, ignore_sessions=[],
                 include_no_action=True):
        'Initialization'
        self.data_list = []
        for session in tqdm(os.listdir(root_path)):
            if session in ignore_sessions:
                continue
            for frame in os.listdir(os.path.join(root_path, session)):
                # don't double count frames
                if 'json' in frame:
                    if not include_no_action:
                        with open(os.path.join(root_path, session, frame), 'r') as temp:
                            temp = json.load(temp)['current_task_meta_str']
                        if temp != 'No action':
                            full_path = os.path.join(root_path, session, frame)
                        else:
                            continue
                    else:
                        full_path = os.path.join(root_path, session, frame)
                    self.data_list.append(full_path)

        self.count = len(self.data_list)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        preprocessor = keras_nlp.models.BertPreprocessor(tokenizer=tokenizer, sequence_length=seq_len)
        embedding = keras_nlp.models.BertBackbone.from_preset(BERT_MODEL)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.BODY_PART_LIST_3D = np.asarray(BODY_PART_LIST_3D, dtype=str)
        self.pose_3D_names = tf.constant(self.BODY_PART_LIST_3D, dtype=tf.string)
        self.pose_3D_names = preprocessor(self.pose_3D_names)
        self.pose_3D_names = embedding(self.pose_3D_names)['sequence_output']
        self.pose_3D_names = tf.expand_dims(self.pose_3D_names, 0)

        self.pose_2D_names = [BODY_PART_LIST_2D[i] for i in range(len(BODY_PART_LIST_2D))]
        self.pose_2D_names = tf.constant(self.pose_2D_names)
        self.pose_2D_names = preprocessor(self.pose_2D_names)
        self.pose_2D_names = embedding(self.pose_2D_names)['sequence_output']
        self.pose_2D_names = tf.reshape(self.pose_2D_names, (1, -1, embed_dim))
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
        poses_3D = np.array([])
        vels_3D = np.array([])
        poses_2D = np.array([])
        time_to_next = np.array([])
        time_to_finish = np.array([])
        # next_task_metas = np.zeros((self.batch_size, self.seq_len, 96))
        # next_task_embeds = np.zeros((self.batch_size, self.seq_len, 96))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            feats_file = ID.split('.')[0] + '.npz'
            feats = np.load(feats_file)['feats']
            feats = feats[np.newaxis, :]
            temp_f = open(ID, 'r')
            current_context = json.load(temp_f)
            temp_f.close()
            try:
                pose_3D = np.asarray_chkfinite(current_context['3D_poses'])[:, 1:]
                vel_3D = np.asarray_chkfinite(current_context['3D_pose_vels'])
                pose_2D = np.asarray_chkfinite(current_context['2D_poses'])[:, :, 1:]

                current_task_embed = np.asarray_chkfinite(current_context['current_task_embed'])
                current_meta_embed = np.asarray_chkfinite(current_context['current_task_meta_embed'])
                next_meta_embed = np.asarray_chkfinite(current_context['next_task_meta_embed'])
                next_task_embed = np.asarray_chkfinite(current_context['next_task_embed'])

                # current_task_token = np.asarray_chkfinite(current_context['current_task_token'])
                # current_meta_token = np.asarray_chkfinite(current_context['current_meta_token'])
                next_task_token = np.asarray_chkfinite(current_context['next_task_token'])[np.newaxis, :]
                next_meta_token = np.asarray_chkfinite(current_context['next_meta_token'])[np.newaxis, :]
            except ValueError:
                print(ID)
                sys.exit()
            # current_task = tf.expand_dims(tf.constant(current_context['current_task_str']), 0)
            # current_meta = tf.expand_dims(tf.constant(current_context['current_task_meta_str']), 0)
            # next_task = tf.expand_dims(current_context['next_task_str'], 0)
            # next_meta = tf.expand_dims(tf.constant(current_context['next_task_meta_str']), 0)

            # pose_3D = pose_3D[np.newaxis, :]
            pose_3D = pose_3D.reshape((-1, 6))
            head_loc = np.where(pose_3D[9] == 0, 1, pose_3D[9])
            pose_3D = pose_3D/head_loc
            pose_3D[9] = head_loc
            pose_3D = np.reshape(pose_3D, (1, len(BODY_PART_LIST_3D), 6))
            pose_3D = np.tile(pose_3D, (1, self.seq_len, 1))
            poses_3D = np.vstack([poses_3D, pose_3D]) if poses_3D.size else pose_3D


            vel_3D = np.reshape(vel_3D, (1, -1, 6))
            vel_3D = np.tile(vel_3D, (1, self.seq_len, 1))
            vels_3D = np.vstack([vels_3D, vel_3D]) if vels_3D.size else vel_3D

            # print(self.pose_2D_names.shape, pose_2D.shape)
            pose_2D_names = tf.tile(self.pose_2D_names, (1, pose_2D.shape[0], 1))

            pose_2D = np.reshape(pose_2D, (1, -1, 3))
            pose_2D = np.tile(pose_2D, (1, self.seq_len, 1))
            poses_2D = np.vstack([poses_2D, pose_2D]) if poses_2D.size else pose_2D

            # current_task = self.preprocessor(current_task)
            # current_task = self.embedding(current_task)
            #
            # current_meta = self.preprocessor(current_meta)
            # current_meta = self.embedding(current_meta)
            #
            # next_task_token = self.preprocessor(next_task)
            # # next_task_token = tokenizer(next_task)
            # next_task = self.embedding(next_task_token)
            #
            # next_meta_token = self.preprocessor(next_meta)
            # # next_meta_token = tokenizer(next_meta)
            # next_meta = self.embedding(next_meta_token)

            pose_3D_names = self.pose_3D_names

            temp = np.asarray_chkfinite(current_context['time_to_next'])
            temp = temp.reshape((1, 1))
            time_to_next = np.vstack([time_to_next, current_context['time_to_next']]) if time_to_next.size else temp

            temp = np.asarray_chkfinite(current_context['time_to_end'])
            temp = temp.reshape((1, 1))
            time_to_finish = np.vstack(
                [time_to_finish, current_context['time_to_end']]) if time_to_finish.size else temp

            # next_task_meta = next_task_meta
            # # next_task_metas = np.vstack([next_task_metas, next_task_meta]) if next_task_metas.size else next_task_meta
            # next_task_metas[i, :next_task_meta.shape[0]] = next_task_meta
            #
            # next_task_embed = next_task_embed
            # # next_task_embeds = np.vstack([next_task_embeds, next_task_embed]) if next_task_embeds.size else next_task_embed
            # next_task_embeds[i, :next_task_embed.shape[0]] = next_task_embed
        X = (poses_3D, vels_3D, pose_3D_names,
             poses_2D, pose_2D_names,
             current_task_embed, current_meta_embed,
             feats)
        Y = (next_task_token, next_task_embed,
             next_meta_token, next_meta_embed,
             time_to_finish, time_to_next)
        return X, Y



if __name__ == "__main__":
    tester, dummy = get_full_train_val_gens(os.path.join("Prepared_Data", "training"), embed_depth=1024)
    inputs, outputs = tester.__getitem__(0)
    p_3, v_3, p3_n, p_2, p2_n, CT_E, CM_E, feats,  = inputs
    NT_S, NT_P, NM_S, NM_P, TF, TN = outputs
    # print('p_2', p_2.shape)
    print('p3_n', p3_n.shape)
    print('p_3', p_3.shape)
    print('p_2', p_2.shape)
    print('p2_n', p2_n.shape)
    # print('NM', NM)
    # print('TF', TF)
    # print('TN', TN)
    print('feats', feats.shape)
