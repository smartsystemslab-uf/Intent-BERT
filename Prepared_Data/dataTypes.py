import PyMO
import numpy as np
import os
import spacy
from typing import Union
import json

nlp = spacy.load("en_core_web_sm")

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
    def __init__(self, task_str: str, start_time: Union[float, np.float32, np.float16, np.float64] = 0.0,
                 stop_time: Union[float, np.float32, np.float16, np.float64] = 0.0):
        self.start_time = start_time
        self.stop_time = stop_time
        if '(' in task_str:
            task_str = task_str[:task_str.find('(')-1]
        self.task_str = task_str.lower()
        try:
            self.meta_task = META_CONVERTER[self.task_str]
        except KeyError:
            self.meta_task = self.task_str.split()[0]
        self.task_embedded = self.get_vector(self.task_str)
        self.meta_embedded = self.get_vector(self.meta_task)
        self.__dict__['string'] = self.task_str
        self.__dict__['embed'] = self.task_embedded
        self.__dict__['meta'] = self.meta_embedded

    def get_vector(self, string: str):
        tokenizer = nlp(string)
        return tokenizer.vector

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
