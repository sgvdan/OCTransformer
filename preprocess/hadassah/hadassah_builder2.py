import os
import pickle
from pathlib import Path

import numpy as np
import pandas
from PIL import Image
from oct_converter.readers import E2E
from tqdm import tqdm

from data.hadassah_data import Sample, Records, Patient


def build_records(annotations_path, labels_to_extract, records=None):
    annotations_path = Path(annotations_path)
    assert annotations_path.exists() and type(labels_to_extract) == list

    # Create new records if records is None, otherwise update
    if records is None:
        records = Records()

    contents = pandas.read_excel(annotations_path)

    for row_idx in range(1, len(contents)):
        patient_name = contents['P.I.D'][row_idx]
        sample_name = contents['S.I.D'][row_idx]
        sample_path = Path(contents['PATH'][row_idx])
        label = {key: contents[key][row_idx] for key in labels_to_extract}

        # Retrieve/Create patient
        patient = records.get_patient(patient_name)
        if patient is None:
            patient = Patient(name=patient_name)
            records.add_patient(patient)
            tqdm.write('Added new patient - {} - to records'.format(patient_name))

        # TODO: how many images? (slices_num)
        slices_num = 0

        # TODO: read sample's metadata -> retrieve e2e_file
        e2e_file_path = 'shit'

        sample = Sample(name=sample_name, label=label,
                        volume_path=sample_path / 'volume/volume.npy', fundus_path=sample_path / 'fundus/fundus.npy',
                        metadata={'e2e_file': e2e_file_path, 'slices_num': slices_num})

    return records


def build(data_root, annotations_file, labels_to_extract, dataset_root, records_file):
    data_root = Path(data_root)
    annotations_path = data_root / annotations_file

    dataset_root = Path(dataset_root)
    records_path = dataset_root / records_file

    records = build_records(annotations_path, labels_to_extract)

    for patient in records.get_patients():
        dest_path = dataset_root / patient.name
        build_patient_dataset(patient, dest_path)

    os.makedirs(records_path.parent, exist_ok=True)
    with open(records_path, 'wb+') as file:
        pickle.dump(records, file)


def main():
    build(data_root='/home/projects/ronen/sgvdan/workspace/datasets/hadassah/original',
          annotations_file='annotations.xlsx',
          labels_to_extract=['DME', 'IRF', 'SRF', 'DME-END', 'IRF-END', 'SRF-END'],
          dataset_root='/home/projects/ronen/sgvdan/workspace/projects/OCTransformer/datasets/temp',
          records_file='records.pkl')


if __name__ == '__main__':
    main()
