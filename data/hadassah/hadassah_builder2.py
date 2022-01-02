import os
import pickle
from pathlib import Path

import numpy as np
import pandas
from PIL import Image
from oct_converter.readers import E2E
from tqdm import tqdm


class Records:
    def __init__(self):
        self.patients = []

    def add_patient(self, patient):
        assert isinstance(patient, Patient)
        self.patients.append(patient)

    def __contains__(self, item):
        assert isinstance(item, Patient)
        return any([patient.name == item.name for patient in self.patients])


class Patient:
    def __init__(self, name, path, reside_path, label):
        self.name = name
        self.label = label
        self.path = path
        self.reside_path = reside_path
        self.volumes = []

    def add_volume(self, volume):
        self.volumes.append(volume)

    def get_volumes(self):
        return self.volumes


class Volume:
    def __init__(self, name, path, reside_path, metadata=None):
        self.name = name
        self.path = path
        self.reside_path = reside_path
        self.metadata = metadata if metadata is not None else {}

    def add_metadata(self, metadatum):
        self.metadata += metadatum


def build_records(data_root, annotations_file, labels_to_extract, dataset_root, records_file):
    data_root = Path(data_root)
    dataset_root = Path(dataset_root)
    annotations_path = data_root / annotations_file
    records_path = dataset_root / records_file

    assert annotations_path.exists() and type(labels_to_extract) == list

    records = Records()
    contents = pandas.read_excel(annotations_path)

    for row_idx in range(1, len(contents)):
        patient_name = contents['P.I.D'][row_idx]
        patient_eye = contents['EYE'][row_idx]
        patient_src_path = data_root / contents['PATH'][row_idx]
        patient_dst_path = dataset_root / patient_name
        label = {key: contents[key][row_idx] for key in labels_to_extract}

        # Create Patient
        tqdm.write('Creating patient {}'.format(patient_name))
        os.makedirs(patient_dst_path, exist_ok=True)
        patient = Patient(name=patient_name, path=patient_src_path, reside_path=patient_dst_path, label=label)
        records.add_patient(patient)
        # TODO: THERE ARE TWO LINES - LEFT & RIGHT. NEED TO FIGURE OUT HOW TO TREAT THOSE (ADD RIGHT, THEN ADD LEFT)

        # Iterate through patient's samples
        for sample in tqdm(list(patient_src_path.rglob("*.E2E"))):
            if sample.is_file():
                volumes = E2E(sample).read_oct_volume()

                for volume_obj in volumes:
                    volume_data = volume_obj.volume

                    validity = [isinstance(tomogram, np.ndarray) for tomogram in volume_data]
                    if not all(validity):
                        tqdm.write('Invalid volume: {}/{}. Ignored.'.format(sample, volume_obj.patient_id))
                        continue

                    # Fix oct-reader bug appending first image to last
                    volume_data.insert(0, volume_data.pop())

                    # Save volume
                    volume_dst_path = patient_dst_path / volume_obj.patient_id / 'data.npy'  # Patient ID are unique
                    os.makedirs(volume_dst_path.parent, exist_ok=True)
                    with open(volume_dst_path, 'wb+') as file:
                        np.save(file, volume_data)

                    # Keep record
                    volume = Volume(name=volume_obj.patient_id, path=sample, reside_path=volume_dst_path)
                    patient.add_volume(volume)

                    # Save Images for debugging-ease
                    images_dst_path = patient_dst_path / volume_obj.patient_id / 'images'
                    os.makedirs(images_dst_path, exist_ok=True)

                    for idx, tomogram in enumerate(volume_data):
                        im = Image.fromarray(tomogram.astype('uint8')).convert('RGB')
                        tomogram_path = images_dst_path / '{}.jpg'.format(idx)
                        with open(tomogram_path, 'wb+') as file:
                            im.save(file)

        tqdm.write('Saved patient {} ({} volumes in total).'.format(patient_name, len(patient.get_volumes())))

    with open(records_path, 'wb+') as file:
        pickle.dump(records, file)

    return records


def main():
    records = build_records(data_root='/home/projects/ronen/sgvdan/workspace/datasets/hadassah/original',
                             annotations_file='annotations.xlsx',
                             labels_to_extract=['DME', 'IRF', 'SRF'],
                             dataset_root='/home/projects/ronen/sgvdan/workspace/datasets/hadassah/temp',
                             records_file='records.pkl')


if __name__ == '__main__':
    main()
