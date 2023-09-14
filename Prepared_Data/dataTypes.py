import PyMO
import numpy as np
import os
import spacy
from typing import Union
import json
import keras_nlp
from dataset_params import *
import tensorflow as tf

tokenizer = keras_nlp.models.BertTokenizer.from_preset(BERT_MODEL)
preprocessor = keras_nlp.models.BertPreprocessor(tokenizer=tokenizer, sequence_length=SEQ_LEN)
embedding = keras_nlp.models.BertBackbone.from_preset(BERT_MODEL)

meta_file = open('meta_lookup.json', 'r')
META_CONVERTER = json.load(meta_file)

class ObjectList:
    def __init__(self, objects: np.ndarray, buffer_size: int = 5):
        '''
        :param objects: a Nx6 array where N is the number of objects, containing bounding box info for each object
        class_id, x1, y1, x2, y2, confidence
        :param buffer_size: number of objects to retain
        '''
        # select only the most confident objects
        objects_to_select = objects.argsort(axis=-1)[-buffer_size:]
        objects = objects[objects_to_select]
        self.objects = np.zeros((buffer_size, 6))
        for i, object in enumerate(objects):
            if i > buffer_size:
                break
            self.objects[i] = object


class Task:
    def __init__(self, task_str: str, meta_task: str, start_time: Union[float, np.float32, np.float16, np.float64] = 0.0,
                 stop_time: Union[float, np.float32, np.float16, np.float64] = 0.0):
        self.start_time = start_time
        self.stop_time = stop_time
        self.task_str = task_str
        self.meta_task = meta_task

        self.task_token = self.to_tokens(task_str.lower())
        self.task_embed = self.to_array(self.task_token)
        self.task_token = self.task_token['token_ids'][0].numpy()
        print(self.task_token.shape)

        self.meta_token = self.to_tokens(meta_task.lower())
        self.meta_embed = self.to_array(self.meta_token)
        self.meta_token = self.meta_token['token_ids'][0].numpy()

        self.__dict__['string'] = self.task_str
        self.__dict__['embed'] = self.task_embed
        self.__dict__['meta_task'] = self.meta_task
        self.__dict__['meta_embed'] = self.meta_embed

    def to_array(self, token):
        temp = embedding(token)
        return np.asarray(temp['sequence_output'])

    def to_tokens(self, string: str):
        temp = tf.expand_dims(tf.constant(string), 0)
        temp = preprocessor(temp)
        return temp

    def __str__(self):
        return self.task_str


class Context:
    def __init__(self, task: Task, objects: ObjectList, pose):
        self.task = task
        self.objects = objects
        self.pose = pose
        self.__dict__['task'] = task
        self.__dict__['objects'] = objects
        self.__dict__['pose'] = pose
