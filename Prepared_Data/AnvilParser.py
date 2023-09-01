from typing import Union
import os

class AnvilParser():
    def __init__(self,  file_path: Union[str, os.PathLike]):
        self.file_path = file_path
        self.open_file = open(self.file_path, 'r')
        self.read_head()

    def read_head(self):
        currentLine = ''
        while 'head' not in currentLine:
            currentLine = self.open_file.readline()
        self.spec = self.get_value(self.open_file.readline())
        self.video_file = self.get_value(self.open_file.readline())
        self.mocap_file = self.get_value(self.open_file.readline())
        # skip to end of head
        self.open_file.readline()
        self.open_file.readline()

    def get_value(self, line: str, key: str = None):
        if key:
            temp = line[line.find(key):]
        else:
            temp = line
        temp = temp[temp.find('='):]
        temp = temp[:temp.find(' ')]
        return temp


