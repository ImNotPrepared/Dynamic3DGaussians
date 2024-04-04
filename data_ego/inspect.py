import os 
import torch
import json
import numpy as np
with open(os.path.join('tennis', 'train_meta.json')) as json_file:
  data=json.load(json_file)

for key, data_item in data.items():
  try:
    print(key, len(data_item))
    print(t(data_item[0]))
  except:
    continue


with open(os.path.join('tennis', 'init_pt_cld.npz')) as json_file:
  data=np.load(json_file)

print(data)


'''
/data3/kaihuac/ImageNet_LT_open/ImageNet_LT_test_output.csv
'''