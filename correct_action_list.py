import json

import pandas as pd
import numpy as np
import math
import spacy


meta_file = 'InHARD/rsc/Action-Meta-action-list.xlsx'
meta_list = pd.read_excel(meta_file)

cols = meta_list.columns
out_dict = {}
last_label = 0
corrected_labels = []
for item in meta_list['ID']:
    if not math.isnan(item):
        last_label = item
    else:
        item = last_label
    corrected_labels.append(item)

meta_list['ID'] = corrected_labels

last_label = 'No action'
corrected_actions = []
for item in meta_list['Meta action label']:
    try:
        if not math.isnan(item):
            last_label = item
        else:
            item = last_label
    except TypeError:
        last_label = item
    corrected_actions.append(item)

meta_list['Meta action label'] = corrected_actions

for i, entry in meta_list.iterrows():
   entry = entry.values
   last_store = out_dict.get(i, [])
   last_store.append(list(entry))
   out_dict[i] = last_store

print(out_dict)
out_file = open('corrected_action_dict.json', 'w+')
out_dict_as_str = json.dumps(out_dict)
out_file.write(out_dict_as_str)
out_file.close()


