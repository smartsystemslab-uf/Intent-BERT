import numpy as np
import ffmpeg
from pymo.parsers import BVHParser
import argparse
import tqdm
import os
import cv2
from AnvilParser import AnvilParser

def prepare_data(args):
    in_data_path = args.in_data_path
    data_type = args.data_type
    out_data_path = os.path.join(args.out_data_path, data_type)
    try:
        os.mkdir(out_data_path)
    except FileExistsError:
        pass
    if data_type == 'training':
        mask = 'P01_R01, P01_R03, P03_R01, P03_R03, P03_R04, P04_R02, P05_R03, P05_R04, P06_R01, P07_R01, P07_R02, ' \
               'P08_R02, P08_R04, P09_R01, P09_R03, P10_R01, P10_R02, P10_R03, P11_R02, P12_R01, P12_R02, P13_R02,' \
               ' P14_R01, P15_R01, P15_R02, P16_R02'
    else:
        mask = 'P01_R02, P02_R01, P02_R02, P04_R01, P05_R01, P05_R02, P08_R01, P08_R03, P09_R02, P11_R01, P14_R02,' \
               ' P16_R01'
    mask = mask.split(' ,')
    vid_path = os.path.join(in_data_path, 'RGB')
    pose_path = os.path.join(in_data_path, 'Skeleton')
    label_path = os.path.join(in_data_path, 'Labels')
    parser = BVHParser()

    for session in mask:
        out_root = os.path.join(out_data_path, session)

        pose_path = os.path.join(pose_path, session+'.bvh')
        poses = parser.parse(pose_path)


        # anvil_path = os.path.join(label_path, session+'.anvil')
        # session_labels = AnvilParser(anvil_path)

        session_vid = os.path.join(vid_path, session+'.mp4')
        session_vid = cv2.VideoCapture(session_vid)
        frame_rate = session_vid.get(cv2.cv.CV_CAP_PROP_FPS)
        seconds_per_frame = 1/frame_rate

        current_frame = 0
        current_time = 0

        ret = True
        while ret:
            img, ret = session_vid.read()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('test', img)
            if cv2.waitKey(5) == ord('q'):
                cv2.destroyAllWindows()
                break
            current_frame += 1
            current_time += seconds_per_frame







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data_path', type=str)
    parser.add_argument('out_data_path', type=str)
    parser.add_argument('data_type', type=str)

    args = parser.parse_args()
    prepare_data(args)


