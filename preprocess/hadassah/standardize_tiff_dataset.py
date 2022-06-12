import argparse
import json
import os
from pathlib import Path

import PIL
from util import glob_re
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle
from torchvision import transforms

# TODO: Standardize dataset
from data.hadassah_data import HadassahDataset


def mask_out_noise(image, snr=10):
    denoised_image = denoise_tv_chambolle(image, weight=1)
    fft = abs(np.fft.fft(denoised_image, axis=-1))  # FFT on the LAST axis

    signal = np.max(fft[:, 1:], axis=-1)  # Take the spectral breakdown "envelope", aside from the "sum" element
    signal /= np.max(signal)

    signal_boundaries = np.argwhere(signal > (1/snr)).squeeze()
    _min, _max = signal_boundaries[0], signal_boundaries[-1]

    _min = max(0, _min - 15)
    _max = min(_max + 15, image.shape[-2] - 1)

    image[:_min, :] = 0
    image[_max:, :] = 0

    return image


def build_patient_dataset(patient_path, dest_path, remove_noise):
    dest_path = Path(dest_path)

    # Iterate through patient's samples
    for sample in tqdm(os.listdir(patient_path)):
        sample_src_path = os.path.join(patient_path, sample)
        sample_dest_path = os.path.join(dest_path, sample)
        if os.path.exists(sample_dest_path):
            continue

        os.makedirs(sample_dest_path, exist_ok=True)

        bscans = glob_re(r'bscan_\d+.tiff', os.listdir(sample_src_path))

        mean = []
        for bscan in tqdm(bscans):
            bscan_src_path = os.path.join(sample_src_path, bscan)
            bscan_dest_path = os.path.join(sample_dest_path, bscan)

            image = Image.open(bscan_src_path)
            denoised_image = Image.fromarray(mask_out_noise(np.array(image)))
            denoised_image.save(bscan_dest_path)


def build(rootdir, destdir, remove_noise):
    rootdir = Path(rootdir)
    destdir = Path(destdir)

    # Iterate through all subdirectories and standardize them in destdir
    for path in rootdir.iterdir():
        if path.is_dir() and path.name in patient_list:
            dest_path = destdir / path.name
            tqdm.write('Saving patient\'s data {} to {}.'.format(path, dest_path))
            build_patient_dataset(path, dest_path, remove_noise)


# def calibrate(dataset_root, dest_path):
#     records = build_hadassah_dataset(dataset_root=dataset_root)  # TODO: make that it doesn't require annotations
#     dataset = HadassahDataset(records.get_samples())
#     mean, stdev = util.get_dataset_stats(dataset)
#
#     with open(dest_path + '/info.json', 'wb+') as file:
#         json.dump({'mean': mean, 'stdev': stdev}, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardize Hadassah Dataset')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input folder path (.xlsx)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output (standardized) folder path (.xlsx)')
    parser.add_argument('-rn', '--remove-noise', action='store_true',
                        help='Choose whether to mask out noise')

    args = parser.parse_args()

    build(args.input, args.output, args.remove_noise)
