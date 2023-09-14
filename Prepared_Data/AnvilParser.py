from typing import Union
import os

class AnvilParser():
    def __init__(self,  file_path: Union[str, os.PathLike]):
        self.file_path = file_path
        self.open_file = open(self.file_path, 'r', encoding='UTF-16')
        self.read_head()
        self.start_time = 0
        # self.current_action = 'No Action'
        self.stop_time = 0
        self.actions = []
        self.meta_actions = []
        self.current_index = 0
        self.parse_doc()

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
        temp = temp[temp.find('=')+1:]
        temp = temp[:temp.find(' ')-1]
        return temp

    def next_action(self):
        while '<el' not in self.current_line:
            self.current_line = self.open_file.readline()
            # print(self.current_line)
            if not self.current_line:
                raise IndexError
            elif '/track' in self.current_line:
                return
        self.start_time = self.get_value(self.current_line, 'start')
        self.stop_time = self.get_value(self.current_line, 'end')
        self.current_line = self.open_file.readline()
        current_action = self.current_line[self.current_line.find(']')+2:self.current_line.find('/')-1]
        self.actions.append((current_action, self.start_time, self.stop_time))
        self.current_line = self.open_file.readline()

    def next_meta_action(self):
        while '<el' not in self.current_line:
            self.current_line = self.open_file.readline()
            # print(self.current_line)
            if not self.current_line:
                raise IndexError
            elif '/track' in self.current_line:
                return
        self.start_time = self.get_value(self.current_line, 'start')
        self.stop_time = self.get_value(self.current_line, 'end')
        self.current_line = self.open_file.readline()
        current_action = self.current_line[self.current_line.find(']')+2:self.current_line.find('/')-1]
        self.meta_actions.append((current_action, self.start_time, self.stop_time))
        self.current_line = self.open_file.readline()

    def parse_doc(self):
        while '/track' not in self.current_line:
            self.next_action()
        # print(self.current_line)
        # move to next
        while '<track' not in self.current_line:
            self.current_line = self.open_file.readline()
            # print(self.current_line)
        while '/track' not in self.current_line:
            # print('before', self.current_line)
            self.next_meta_action()
        # print(len(self.actions), self.actions)
        # print(len(self.meta_actions), self.meta_actions)
        if len(self.actions) != len(self.meta_actions):
            raise RuntimeError

    def time_as_float(self, old_time):
        try:
            new_time = float(old_time[1:])
        except ValueError:
            new_time = float(old_time[1:-1])
        return new_time

    def __next__(self):
        action, start, stop = self.actions[self.current_index]
        meta_action = self.meta_actions[self.current_index][0]
        self.current_index += 1
        print(self.current_index, len(self.actions))
        if self.current_index >= len(self.actions):
            raise IndexError
        return action, meta_action, self.time_as_float(start), self.time_as_float(stop)

    def next(self):
        try:
            out = self.__next__()
        except IndexError:
            raise IndexError
        return out

    def __del__(self):
        self.open_file.close()

if __name__ == "__main__":
    test_file = '../InHARD/Online/Labels/P01_R01.anvil'
    tester = AnvilParser(test_file)
    print(tester.spec)
    print(tester.next())


