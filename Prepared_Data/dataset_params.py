import json

BODY_PART_LIST_3D = ['Hips', 'LeftFoot', 'LeftHand', 'LeftLeg', 'LeftUpLeg', 'LeftForeArm', 'LeftArm', 'LeftShoulder',
                  'RightFoot', 'RightHand', 'RightLeg', 'RightUpLeg', 'RightForeArm', 'RightArm', 'RightShoulder',
                  'Spine', 'Head']

BODY_PART_LIST_2D = {0: "Nose", 1: "LeftEye",2: "RightEye",3: "LeftEar",
                    4: "RightEar",
                    5: "LeftShoulder",
                    6: "RightShoulder",
                    7: "LeftElbow",
                    8: "RightElbow",
                    9: "LeftWrist",
                    10: "RightWrist",
                    11: "LeftHip",
                    12: "RightHip",
                    13: "LeftKnee",
                    14: "Rightknee",
                    15: "LeftAnkle", 16: "RightAnkle"}
POSE_COMPONENTS = ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']

FIELD_NAMES = []

POSE_MODEL= 'yolov8x-pose'

threshold = 0.85
#P01_R01, P01_R03, P03_R01, P03_R03, P03_R04,
train_set = 'P04_R02, P05_R03, P05_R04, P06_R01, P07_R01, P07_R02, ' \
               'P08_R02, P08_R04, P09_R01, P09_R03, P10_R01, P10_R02, P10_R03, P11_R02, P12_R01, P12_R02, P13_R02, ' \
               'P14_R01, P15_R01, P15_R02, P16_R02'

test_set = 'P01_R02, P02_R01, P02_R02, P04_R01, P05_R01, P05_R02, P08_R01, P08_R03, P09_R02, P11_R01, P14_R02,' \
               ' P16_R01'
# 'P01_R02, P02_R01, P02_R02, P04_R01, P05_R01, P05_R02, P08_R01, P08_R03, P09_R02, P11_R01, P14_R02,' \
#                ' P16_R01'

NLP_MODEL = "en_core_web_lg"

BERT_MODEL = "bert_large_en_uncased"

EMBED_DETPH = 1024

SEQ_LEN = 10