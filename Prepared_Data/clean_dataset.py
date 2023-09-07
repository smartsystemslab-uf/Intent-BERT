import json

import numpy as np
import ffmpeg
from pymo.parsers import BVHParser
import argparse
from tqdm import tqdm
import os
import cv2
from AnvilParser import AnvilParser
from dataTypes import Task, Context

BODY_PART_LIST = ['Hips', 'LeftFoot', 'LeftHand', 'LeftLeg', 'LeftUpLeg', 'LeftForeArm', 'LeftArm', 'LeftShoulder',
                  'RightFoot', 'RightHand', 'RightLeg', 'RightUpLeg', 'RightForeArm', 'RightArm', 'RightShoulder',
                  'Spine', 'Head']
POSE_COMPONENTS = ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']

FIELD_NAMES = []
for part in BODY_PART_LIST:
    for comp in POSE_COMPONENTS:
        temp = '_'.join([part, comp])
        FIELD_NAMES.append(temp)

def label_to_Task(label_out):
    action_label = label_out[2]
    try:
        start_time = float(label_out[0][1:-1])
    except ValueError:
        start_time = float(label_out[0][1:])
    try:
        stop_time = float(label_out[1][1:-1])
    except ValueError:
        stop_time = float(label_out[1][1:])
    return Task(action_label, start_time, stop_time)

def prepare_data(args):
    in_data_path = args.in_data_path
    data_type = args.data_type
    out_data_path = os.path.join(args.out_data_path, data_type)
    try:
        os.mkdir(out_data_path)
    except FileExistsError:
        pass
    if data_type == 'training': #
        mask = 'P01_R01, P01_R03, P03_R01, P03_R03, P03_R04, P04_R02, P05_R03, P05_R04, P06_R01, P07_R01, P07_R02, ' \
               'P08_R02, P08_R04, P09_R01, P09_R03, P10_R01, P10_R02, P10_R03, P11_R02, P12_R01, P12_R02, P13_R02,' \
               ' P14_R01, P15_R01, P15_R02, P16_R02'
    else:
        mask = 'P01_R02, P02_R01, P02_R02, P04_R01, P05_R01, P05_R02, P08_R01, P08_R03, P09_R02, P11_R01, P14_R02,' \
               ' P16_R01'
    mask = mask.split(', ')
    vid_root_path = os.path.join(in_data_path, 'RGB')
    pose_root_path = os.path.join(in_data_path, 'Skeleton')
    label_root_path = os.path.join(in_data_path, 'Labels')
    parser = BVHParser()

    for session in tqdm(mask):
        print('starting', session)
        out_root = os.path.join(out_data_path, session)
        try:
            os.mkdir(out_root)
        except FileExistsError:
            pass

        print('loading poses')
        pose_path = os.path.join(pose_root_path, session+'.bvh')
        poses = parser.parse(pose_path)
        pose_times = poses.values.index
        pose_times = pose_times.total_seconds()
        pose_values = poses.values[FIELD_NAMES]


        anvil_path = os.path.join(label_root_path, session+'.anvil')
        session_labels = AnvilParser(anvil_path)
        current_task = Task('No Action')
        next_task = label_to_Task(session_labels.next())


        session_vid = os.path.join(vid_root_path, session+'.mp4')
        session_vid = cv2.VideoCapture(session_vid)
        frame_rate = session_vid.get(cv2.CAP_PROP_FPS)
        seconds_per_frame = 1/frame_rate

        current_frame = 0
        current_time = 0
        last_time = 0
        last_task_switch = 0

        ret = True
        while ret:
            # print(current_frame)
            ret, img = session_vid.read()
            try:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('test', img)
            except:
                cv2.destroyAllWindows()
                ret = False
                break
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                ret = False
                break
            current_frame += 1
            current_time += seconds_per_frame

            poses_for_frame = last_time <= pose_times
            poses_for_frame = poses_for_frame & (pose_times < current_time)
            pose_readings = pose_values.loc[poses_for_frame]
            # take average of all readings across time frame
            pose_readings = pose_readings.mean(axis=0)
            # print(pose_readings)
            pose_readings = np.asarray(pose_readings)
            # print(pose_readings)
            pose_readings = pose_readings.tolist()
            # print(pose_readings)

            # if we are past the end of the last task
            if current_time > current_task.stop_time:
                # get the next task
                current_task = next_task
                try:
                    next_task = label_to_Task(session_labels.next())
                except IndexError:
                    cv2.destroyAllWindows()
                    break
                last_task_switch = current_time


            if current_time < current_task.start_time:
                actual_task = Task('Consult Sheets', start_time=last_task_switch, stop_time=current_task.start_time)

            else:
                actual_task = current_task


            # print(pose_readings)
            # print(actual_task.task_str, current_time, next_task.task_str)

            out_file_name = os.path.join(out_root, str(current_frame)+'.json')
            if np.all(np.isfinite(pose_readings)):
                with open(out_file_name, 'w') as f_temp:
                    dict_to_write = {}
                    dict_to_write['poses'] = pose_readings
                    dict_to_write['current_task_str'] = actual_task.task_str
                    dict_to_write['current_task_embed'] = actual_task.task_embedded.tolist()
                    dict_to_write['current_task_meta'] = actual_task.meta_embedded.tolist()
                    dict_to_write['time_to_end'] = actual_task.stop_time - current_time
                    dict_to_write['time_to_next'] = next_task.start_time - current_time
                    dict_to_write['next_task_str'] = next_task.task_str
                    dict_to_write['next_task_embed'] = next_task.task_embedded.tolist()
                    dict_to_write['next_task_meta'] = next_task.meta_embedded.tolist()
                    json.dump(dict_to_write, f_temp)
                    f_temp.close()

            last_time = current_time

        cv2.destroyAllWindows()







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data_path', type=str)
    parser.add_argument('out_data_path', type=str)
    parser.add_argument('data_type', type=str, choices=['training', 'testing'])

    args = parser.parse_args()
    prepare_data(args)


