import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from oct_converter.readers import E2E
from tqdm import tqdm

# TODO: Standardize dataset
from data.hadassah_data import Sample


def build_patient_dataset(patient_path, dest_path):
    dest_path = Path(dest_path)

    # Iterate through patient's samples
    for e2e_path in tqdm(list(Path(patient_path).rglob("*.E2E"))):
        if e2e_path.is_file():
            e2e = E2E(e2e_path)
            volumes = e2e.read_oct_volume()
            try:
                fundusi = e2e.read_fundus_image()
            except Exception as e:
                tqdm.write('could not read fundus from {}. Exception: {}:{}'.format(e2e_path, type(e), str(e)))
                fundusi = None

            for idx, volume_obj in enumerate(volumes):
                sample_dst_path = dest_path / volume_obj.patient_id
                volume_data = volume_obj.volume

                validity = [isinstance(tomogram, np.ndarray) for tomogram in volume_data]
                if not all(validity):
                    tqdm.write('Invalid volume: {}/{}. Ignored.'.format(e2e_path, volume_obj.patient_id))
                    continue
                # if len(validity) != 37:  # TODO: REMOVE / PUT IN CONFIG!!
                #     tqdm.write('Taking 37-slice volumes only. {}/{} Ignored.'.format(e2e_path, volume_obj.patient_id))
                #     continue

                volume_data.insert(0, volume_data.pop())  # Fix oct-reader bug appending first image to last

                # Save volume
                volume_dst_path = sample_dst_path / 'volume'  # Patient ID are unique
                os.makedirs(volume_dst_path, exist_ok=True)
                with open(volume_dst_path / 'data.npy', 'wb+') as file:
                    np.save(file, np.stack(volume_data, axis=2))  # Save as single tensor

                # Also save Images for debugging-ease
                images_dst_path = volume_dst_path / 'images'
                os.makedirs(images_dst_path, exist_ok=True)

                for i, tomogram in enumerate(volume_data):
                    im = Image.fromarray(tomogram.astype('uint8')).convert('RGB')
                    tomogram_path = images_dst_path / '{}.jpg'.format(i)
                    with open(tomogram_path, 'wb+') as file:
                        im.save(file)

                # Save fundus
                if fundusi is not None:
                    fundus_dst_path = sample_dst_path / 'fundus'
                    os.makedirs(fundus_dst_path, exist_ok=True)
                    with open(fundus_dst_path / 'fundus.npy', 'wb+') as file:
                        np.save(file, fundusi[idx].image)

                    # Also save fundus image for debugging-ease
                    im = Image.fromarray(fundusi[idx].image).convert('RGB')
                    with open(fundus_dst_path / 'fundus.jpg', 'wb+') as file:
                        im.save(file)

                # Keep metadata
                with open(sample_dst_path / '.metadata', 'wb+') as file:
                    pickle.dump({'e2e_file': e2e_path}, file)

    tqdm.write('Saved patient\'s data {} to {}.'.format(patient_path, dest_path))


def build(rootdir, destdir):
    rootdir = Path(rootdir)
    destdir = Path(destdir)

    # Iterate through all subdirectories and standardize them in destdir
    for path in rootdir.iterdir():
        if path.is_dir():
            build_patient_dataset(path, destdir / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardize Hadassah Dataset')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input folder path (.xlsx)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output (standardized) folder path (.xlsx)')

    args = parser.parse_args()

    build(args.input, args.output)
