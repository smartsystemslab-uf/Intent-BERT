import json
import sys

import numpy as np
import ffmpeg
from pymo.parsers import BVHParser
import argparse
from tqdm import tqdm
import os
import cv2
from AnvilParser import AnvilParser
from dataTypes import Task, Context
import tensorflow as tf
from ultralytics import YOLO
import keras_nlp

from dataset_params import *

pose_model = YOLO(POSE_MODEL)

for part in BODY_PART_LIST_3D:
    for comp in POSE_COMPONENTS:
        temp = '_'.join([part, comp])
        FIELD_NAMES.append(temp)

def label_to_Task(label_out):
    action_label = label_out[0]
    meta_action_label = label_out[1]
    start_time = label_out[2]
    stop_time = label_out[3]
    return Task(action_label, meta_action_label, start_time, stop_time)

def prepare_data(args):
    in_data_path = args.in_data_path
    data_type = args.data_type
    out_data_path = os.path.join(args.out_data_path, data_type)
    try:
        os.mkdir(out_data_path)
    except FileExistsError:
        pass
    if data_type == 'training': #
        mask = train_set
    else:
        mask = test_set
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
        current_task = Task('No Action', 'No Action')
        next_task = label_to_Task(session_labels.next())


        session_vid = os.path.join(vid_root_path, session+'.mp4')
        session_vid = cv2.VideoCapture(session_vid)
        frame_rate = session_vid.get(cv2.CAP_PROP_FPS)
        seconds_per_frame = 1/frame_rate
        height = int(session_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(session_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        feat_model = tf.keras.applications.ResNet50(include_top=False,
                                                    input_shape=(height,
                                                                 width,
                                                                 3),
                                                    )

        current_frame = 0
        current_time = 0
        last_time = 0
        last_pose_3d = np.array([])

        ret = True
        while ret:
            # print(current_frame)
            current_frame += 1
            current_time += seconds_per_frame
            if current_time < current_task.start_time:
                continue

            ret, img = session_vid.read()
            try:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB
                pose_2D_pred = pose_model.predict(img, conf=0.7)[0]
                feats = feat_model.predict(tf.keras.applications.resnet.preprocess_input(img[np.newaxis, ::-1]))
                print(feats.shape)
                cv2.imshow(session, pose_2D_pred.plot())


            except cv2.error as e:
                print(e)
                cv2.destroyAllWindows()
                break
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

            # if we are past the end of the last task
            print(current_task.stop_time, current_time)
            if current_time > current_task.stop_time:
                print(current_task.stop_time, current_time, 'moving to next task')
                # get the next task
                current_task = next_task
                try:
                    next_task = label_to_Task(session_labels.next())
                except IndexError:
                    cv2.destroyAllWindows()
                    break
                if next_task.start_time < current_task.stop_time:
                    next_task.start_time = current_task.stop_time

            try:
                pose_2D_pred = pose_2D_pred.cpu().numpy().keypoints
                # invalid_detections = np.nonzero(pose_2D_pred.conf < threshold)
                # print(pose_2D_pred.conf.shape, pose_2D_pred.xyn.shape)
                confs = pose_2D_pred.conf[:, :, np.newaxis]
            except TypeError:
                continue
            norm_preds = pose_2D_pred.xyn
            new_inds = np.ones_like(confs)
            new_inds = new_inds.cumsum(axis=1)
            # print(new_inds.shape, confs.shape, norm_preds.shape)
            pose_2D_pred = np.concatenate([new_inds, confs, norm_preds], axis=-1)
            # print(pose_2D_pred)

            if current_time > next_task.start_time or current_time > current_task.stop_time:
                print('invalid timing, exiting')
                print(current_time > next_task.start_time)
                print(current_time > current_task.stop_time)
                print(current_time, next_task.start_time, next_task.stop_time, current_task.start_time, current_task.stop_time, current_frame, session)
                sys.exit()


            # print(pose_readings)
            # print(actual_task.task_str, current_time, next_task.task_str)

            poses_for_frame = last_time <= pose_times
            poses_for_frame = poses_for_frame & (pose_times < current_time)
            pose_readings = pose_values.loc[poses_for_frame]
            # take average of all readings across time frame
            pose_readings = pose_readings.mean(axis=0)
            # print(pose_readings)
            pose_readings = np.asarray(pose_readings)
            # print(pose_readings)
            # pose_readings = pose_readings
            # print(pose_readings)

            out_file_name = os.path.join(out_root, str(current_frame)+'.json')
            if np.all(np.isfinite(pose_readings)):
                with open(out_file_name, 'w') as f_temp:
                    if not last_pose_3d.size:
                        last_pose_3d = pose_readings

                    pose_vel_3D = pose_readings - last_pose_3d

                    inds = np.ones_like(pose_readings)
                    inds = inds.cumsum(axis=0)
                    pose_readings = np.concatenate([inds, pose_readings])
                    pose_readings = np.reshape(pose_readings, (-1, 2), order='F')

                    dict_to_write = {}
                    dict_to_write['3D_poses'] = pose_readings.tolist()
                    dict_to_write['3D_pose_vels'] = pose_vel_3D.tolist()
                    dict_to_write['2D_poses'] = pose_2D_pred.tolist()
                    dict_to_write['current_task_str'] = current_task.task_str
                    dict_to_write['current_task_embed'] = current_task.task_embed.tolist()
                    dict_to_write['current_task_token'] = current_task.task_token.tolist()
                    dict_to_write['current_task_meta_embed'] = current_task.meta_embed.tolist()
                    dict_to_write['current_task_meta_str'] = current_task.meta_task
                    dict_to_write['current_meta_token'] = current_task.meta_token.tolist()
                    dict_to_write['time_to_end'] = current_task.stop_time - current_time
                    dict_to_write['time_to_next'] = next_task.start_time - current_time
                    dict_to_write['next_task_str'] = next_task.task_str
                    dict_to_write['next_task_embed'] = next_task.task_embed.tolist()
                    dict_to_write['next_task_token'] = next_task.task_token.tolist()
                    dict_to_write['next_task_meta_embed'] = next_task.meta_embed.tolist()
                    dict_to_write['next_task_meta_str'] = next_task.meta_task
                    dict_to_write['next_meta_token'] = next_task.meta_token.tolist()
                    np.savez_compressed(os.path.join(out_root, str(current_frame)), feats=feats[0])
                    json.dump(dict_to_write, f_temp)
                    f_temp.close()

            last_time = current_time
            last_pose_3d = pose_readings[:, 1]

        cv2.destroyAllWindows()







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data_path', type=str)
    parser.add_argument('out_data_path', type=str)
    parser.add_argument('data_type', type=str, choices=['training', 'testing'])

    args = parser.parse_args()
    prepare_data(args)


