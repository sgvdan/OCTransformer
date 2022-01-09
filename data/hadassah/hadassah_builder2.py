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

    def get_patient(self, patient_name):
        for patient in self.patients:
            if patient_name == patient.name:
                return patient

        return None

    def get_patients(self):
        return self.patients


class Patient:

    LEFT_EYE = 0
    RIGHT_EYE = 1

    def __init__(self, name, path):
        """

        :param name:
        :param path:
        :param reside_path:
        """
        self.name = name
        self.path = path
        self.volumes = []
        self.left_label = self.right_label = None

    def set_label(self, side, label):
        if side == Patient.LEFT_EYE:
            self.left_label = label
        elif side == Patient.RIGHT_EYE:
            self.right_label = label
        else:
            raise NotImplementedError

    def get_label(self, side):
        if side == Patient.LEFT_EYE:
            return self.left_label
        elif side == Patient.RIGHT_EYE:
            return self.right_label
        else:
            raise NotImplementedError

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


def build_patient_dataset(patient, dest_path):
    dest_path = Path(dest_path)

    # Iterate through patient's samples
    for sample in tqdm(list(Path(patient.path).rglob("*.E2E"))):
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
                volume_dst_path = dest_path / volume_obj.patient_id / 'data.npy'  # Patient ID are unique
                os.makedirs(volume_dst_path.parent, exist_ok=True)
                with open(volume_dst_path, 'wb+') as file:
                    np.save(file, volume_data)

                # Keep record
                volume = Volume(name=volume_obj.patient_id, path=sample, reside_path=volume_dst_path)
                patient.add_volume(volume)

                # Save Images for debugging-ease
                images_dst_path = dest_path / volume_obj.patient_id / 'images'
                os.makedirs(images_dst_path, exist_ok=True)

                for idx, tomogram in enumerate(volume_data):
                    im = Image.fromarray(tomogram.astype('uint8')).convert('RGB')
                    tomogram_path = images_dst_path / '{}.jpg'.format(idx)
                    with open(tomogram_path, 'wb+') as file:
                        im.save(file)

    tqdm.write('Saved patient\'s data {} to {} ({} volumes in total).'
               .format(patient.name, dest_path, len(patient.get_volumes())))


def build_records(annotations_path, labels_to_extract, records=None):
    annotations_path = Path(annotations_path)
    assert annotations_path.exists() and type(labels_to_extract) == list

    # Create new records if records is None, otherwise update
    if records is None:
        records = Records()

    contents = pandas.read_excel(annotations_path)

    for row_idx in range(1, len(contents)):
        patient_name = contents['P.I.D'][row_idx]
        patient_src_path = Path(contents['PATH'][row_idx])
        label = {key: contents[key][row_idx] for key in labels_to_extract}
        patient_eye = Patient.LEFT_EYE if contents['EYE'][row_idx] == 'OS' else Patient.RIGHT_EYE

        assert contents['EYE'][row_idx] in ['OS', 'OD']

        # Does exist?
        patient = records.get_patient(patient_name)
        if patient is not None:
            # Patient exists - set label and continue
            patient.set_label(patient_eye, label)
            tqdm.write('Set {} eye label for existing patient {}'
                       .format('left' if patient_eye == Patient.LEFT_EYE else 'right', patient.name))
            continue

        # Patient does not exist, create a new one
        patient = Patient(name=patient_name, path=patient_src_path)
        patient.set_label(patient_eye, label)
        tqdm.write('New Patient - {} - created. Set {} eye\'s label'
                   .format(patient.name, 'left' if patient_eye == Patient.LEFT_EYE else 'right'))

        # Add to records
        records.add_patient(patient)
        tqdm.write('Added new patient - {} - to records'.format(patient_name))

    return records


def build(data_root, annotations_file, labels_to_extract, dataset_root, records_file):
    data_root = Path(data_root)
    annotations_path = data_root / annotations_file

    dataset_root = Path(dataset_root)
    records_path = dataset_root / records_file

    records = build_records(annotations_path, labels_to_extract)
    os.makedirs(records_path.parent, exist_ok=True)
    with open(records_path, 'wb+') as file:
        pickle.dump(records, file)

    for patient in records.get_patients():
        dest_path = dataset_root / patient.name
        build_patient_dataset(patient, dest_path)


def main():
    build(data_root='/home/projects/ronen/sgvdan/workspace/datasets/hadassah/original',
          annotations_file='annotations.xlsx',
          labels_to_extract=['DME', 'IRF', 'SRF', 'DME-END', 'IRF-END', 'SRF-END'],
          dataset_root='/home/projects/ronen/sgvdan/workspace/datasets/hadassah/temp2',
          records_file='records.pkl')


if __name__ == '__main__':
    main()
