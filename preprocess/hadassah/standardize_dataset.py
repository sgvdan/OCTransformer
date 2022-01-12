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
            volumes = E2E(e2e_path).read_oct_volume()

            for volume_obj in volumes:
                sample_dst_path = dest_path / volume_obj.patient_id
                volume_data = volume_obj.volume

                validity = [isinstance(tomogram, np.ndarray) for tomogram in volume_data]
                if not all(validity):
                    tqdm.write('Invalid volume: {}/{}. Ignored.'.format(e2e_path, volume_obj.patient_id))
                    continue
                if len(validity) != 37:  # TODO: REMOVE / PUT IN CONFIG!!
                    tqdm.write('Taking 37-slice volumes only. {}/{} Ignored.'.format(e2e_path, volume_obj.patient_id))
                    continue

                volume_data.insert(0, volume_data.pop())  # Fix oct-reader bug appending first image to last
                volume_data = np.moveaxis(volume_data, 0, 2)  # Works best with Pytorch' transformation

                # Save volume
                volume_dst_path = sample_dst_path / 'volume'  # Patient ID are unique
                os.makedirs(volume_dst_path, exist_ok=True)
                with open(volume_dst_path / 'data.npy', 'wb+') as file:
                    np.save(file, volume_data)

                # Also save Images for debugging-ease
                images_dst_path = volume_dst_path / 'images'
                os.makedirs(images_dst_path, exist_ok=True)

                for idx, tomogram in enumerate(volume_data):
                    im = Image.fromarray(tomogram.astype('uint8')).convert('RGB')
                    tomogram_path = images_dst_path / '{}.jpg'.format(idx)
                    with open(tomogram_path, 'wb+') as file:
                        im.save(file)

                # Save fundus
                fundus_dst_path = sample_dst_path / 'fundus'
                os.makedirs(fundus_dst_path, exist_ok=True)
                # with open(fundus_dst_path / 'fundus.npy', 'wb+') as file:
                #     np.save(file, fundus)

                # Also save fundus image for debugging-ease
                # TODO save fundus as image

                # Keep metadata
                with open(sample_dst_path / '.metadata', 'wb+') as file:
                    pickle.dump({'e2e_file': e2e_path})

    tqdm.write('Saved patient\'s data {} to {}.'.format(patient_path, dest_path))


def build():
    # TODO: run through all subdirectories of patients and feed it to the above directory