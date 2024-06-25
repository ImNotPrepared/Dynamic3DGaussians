import os 
import torch
import json
import numpy as np


data = np.load(os.path.join('tennis', 'init_pt_cld.npz'))

# Inspect the keys in the .npz file
for key in data.files:
    array_shape = data[key].shape
    print(f"Shape of the array under key '{key}': {array_shape}")

'''
/data3/kaihuac/ImageNet_LT_open/ImageNet_LT_test_output.csv
with open(os.path.join('tennis', 'train_meta.json')) as json_file:
  data=json.load(json_file)

for key, data_item in data.items():
  try:
    print(key, len(data_item))
    print(t(data_item[0]))
  except:
    continue
'''