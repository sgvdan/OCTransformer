from data.kermany_data import build_kermany_dataset

import os
import re
from pathlib import Path
#
# dir = Path('/home/nih/nih-dannyh/data/oct/DR_TIFF')
#
# count = 0
# regex = r'\d*_\d*_(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)_.*_(?P<eye>[R|L])'
# for patient in dir.glob('DR_*'):
#     for sample in patient.glob('*'):
#         if ('bscan_36.tiff' in os.listdir(sample)) and ('bscan_37.tiff' not in os.listdir(sample)):
#             matches = re.match(regex, sample.name)
#             if matches is None:
#                 continue
#
#             matches = matches.groupdict()
#
#             print('{pid},{eye},{day}/{month}/{year},{filename},{labels}'
#                   .format(pid=patient.name, eye='OD' if matches['eye'] == 'R' else 'OS',
#                           day=matches['day'], month=matches['month'], year=matches['year'],
#                           filename=sample.name, labels=','.join(['0'] * 16)))
#             count += 1
#
# print('DONE - ', count)

# import numpy as np
# from tqdm import tqdm
# dir = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std')
# for sample in tqdm(dir.glob('*/*/data-volume.npy')):
#     data = np.load(sample)
#     reshaped = np.moveaxis(data, 0, 2)
#     np.save(sample, reshaped)


# import torch
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# from torchvision.io import read_image
#
# image_data = Image.open('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train/DME/DME-30521-17.jpeg')
# image_data = np.array(image_data)
#
# image_data_t = (transforms.ToTensor()(image_data) * 255).to(torch.uint8)
# image_data_r = read_image('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train/DME/DME-30521-17.jpeg')
#
# resize = transforms.Resize((256,256))
# image_data_t = resize(image_data_t)
# image_data_r = resize(image_data_r)
#
# print((image_data_t - image_data_r).max())

# from pathlib import Path
# image_src = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std2')
# ls_src = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-brightness-enhanced')
# dest = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-brightness-enhanced-ls-only')
#
# import shutil
#
# for sample_path in image_src.glob('*/*/data-volume.npy'):
#     sample_id = sample_path.parent.name
#     patient_id = sample_path.parent.parent.name
#
#     ls_path = ls_src / patient_id / sample_id / 'ls-volume.npy'
#
#     dest_path = dest / patient_id / sample_id
#     os.makedirs(dest_path, exist_ok=True)
#
#     shutil.copy2(sample_path, dest_path)
#     shutil.copy2(ls_path, dest_path)
#
#     print('{}:{}'.format(patient_id, sample_id))

# from pathlib import Path
# image_src = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std2')
# ls_src = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-brightness-enhanced')
# dest = Path('/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-brightness-enhanced-ls-only')
#
# import shutil
#
# for sample_path in image_src.glob('*/*/data-volume.npy'):
#     sample_id = sample_path.parent.name
#     patient_id = sample_path.parent.parent.name
#
#     ls_path = ls_src / patient_id / sample_id / 'ls-volume.npy'
#
#     dest_path = dest / patient_id / sample_id
#     os.makedirs(dest_path, exist_ok=True)
#
#     shutil.copy2(sample_path, dest_path)
#     shutil.copy2(ls_path, dest_path)
#
#     print('{}:{}'.format(patient_id, sample_id))

# from filelock import FileLock
#
# with FileLock("temp.txt"), open("temp.txt") as file:
#     # work with the file as it is now locked
#     print("Lock acquired.")
#     print(file)

print('Build Kermany train', flush=True)
build_kermany_dataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train',
                      '/home/projects/ronen/sgvdan/workspace/datasets/kermany/ls-confidence2/train')

print('Build Kermany val', flush=True)
build_kermany_dataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/val',
                      '/home/projects/ronen/sgvdan/workspace/datasets/kermany/ls-confidence2/val')

print('Build Kermany test', flush=True)
build_kermany_dataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/test',
                      '/home/projects/ronen/sgvdan/workspace/datasets/kermany/ls-confidence2/test')
