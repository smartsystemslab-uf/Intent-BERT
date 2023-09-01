from typing import Union
import os

class AnvilParser():
    def __init__(self,  file_path: Union[str, os.PathLike]):
        self.file_path = file_path
        self.open_file = open(self.file_path, 'r')
        self.read_head()
        self.start_time = 0
        self.current_action = 'No Action'
        self.stop_time = 0

    def read_head(self):
        currentLine = ''
        while 'head' not in currentLine:
            currentLine = self.open_file.readline()
        self.spec = self.get_value(self.open_file.readline())
        self.video_file = self.get_value(self.open_file.readline())
        self.mocap_file = self.get_value(self.open_file.readline())
        # skip to end of head
        self.open_file.readline()
        self.current_line = self.open_file.readline()

    def get_value(self, line: str, key: str = None):
        if key:
            temp = line[line.find(key):]
        else:
            temp = line
        temp = temp[temp.find('='):]
        temp = temp[:temp.find(' ')]
        return temp

    def next_action(self):
        while '<el' not in self.current_line:
            self.current_line = self.open_file.readline()
        self.start_time = self.get_value(self.current_line, 'start')
        self.stop_time = self.get_value(self.current_line, 'stop')
        self.current_line = self.open_file.readline()
        self.current_action = self.current_line[self.current_line.find('>'):self.current_line.find('<')]
        self.current_line = self.open_file.readline()

    def __next__(self):
        self.next_action()
        return self.start_time, self.stop_time, self.current_action

    def next(self):
        return self.__next__()

    def __del__(self):
        self.open_file.close()

if __name__ == "__main__":



