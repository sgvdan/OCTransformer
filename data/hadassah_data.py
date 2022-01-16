import pickle
from pathlib import Path

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HadassahDataset(Dataset):
    def __init__(self, samples, chosen_label, transformations):
        self.transformations = transformations
        self.samples = samples
        self.chosen_label = chosen_label

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sample_data = sample.get_data()
        sample_label = sample.get_label()[self.chosen_label]

        sample_data = self.transformations(sample_data)
        sample_data = sample_data.unsqueeze(dim=1).expand(-1, 3, -1, -1)

        return sample_data, sample_label

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [sample.get_label()[self.chosen_label] for sample in self.samples]

    def get_classes(self):
        # TODO: REMOVE THIS! SOMEHOW TREAT IT WITH LABEL NAME
        return {'DME_HEALTHY': 0, 'DME_SICK': 1}


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

    def get_samples(self):
        return [sample for patient in self.patients for sample in patient.get_samples()]

    def slice_samples(self, labels):
        """
        Performs a control-study cut of the samples according to labels
        """
        control, study = [], []
        for patient in self.patients:
            for sample in patient.get_samples():
                correspondence = all([sample.get_label()[key] == value for key, value in labels.items()])
                if correspondence:
                    study.append(sample)
                    print('Pat.{}-Sam.{} was added to study group'.format(patient.name, sample.name))
                else:
                    control.append(sample)
                    print('Pat.{}-Sam.{} was added to control group'.format(patient.name, sample.name))

        return control, study


class Patient:
    def __init__(self, name):
        self.name = name
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)

    def get_samples(self):
        return self.samples

    def __len__(self):
        return len(self.samples)


class Sample:
    def __init__(self, name, label, volume_path=None, fundus_path=None, metadata_path=None):
        self.name = name
        self.label = label
        self.volume_path = volume_path
        self.fundus_path = fundus_path
        self.metadata_path = metadata_path

    def set_metadata(self, metadatum):
        metadata = self.get_metadata() + metadatum

        with open(self.metadata_path, 'wb+') as file:
            pickle.dump(metadata, file)

    def get_label(self):
        return self.label

    def get_metadata(self):
        if self.metadata_path is None:
            return None
        else:
            with open(self.metadata_path, 'rb') as file:
                return pickle.load(file)

    def get_data(self):
        if self.volume_path is None:
            return None
        else:
            return np.load(self.volume_path, mmap_mode='r')

    def get_fundus(self):
        if self.fundus_path is None:
            return None
        else:
            return np.load(self.fundus_path, mmap_mode='r')


def build_hadassah_dataset(dataset_root, annotations_path):
    annotations = pandas.read_excel(annotations_path)
    dataset_root = Path(dataset_root)

    records = Records()
    for row_idx in range(1, len(annotations)):
        metadata = annotations.iloc[row_idx, :7]
        label = annotations.iloc[row_idx, 7:]

        patient_name = metadata['DR_CODE']
        sample_name = '{}_{}_{}'.format(metadata['P.I.D'], metadata['E.I.D'], metadata['S.I.D'])
        path = dataset_root / '{}/{}'.format(patient_name, sample_name)

        print(path)
        assert path.exists()

        patient = records.get_patient(patient_name)
        if patient is None:
            patient = Patient(name=patient_name)
            records.add_patient(patient)

        temp = path / 'volume/images/'
        if len(list(temp.glob('*'))) != 37:
            continue

        sample = Sample(sample_name, label,
                        volume_path=path / 'volume/data.npy',
                        fundus_path=path / 'fundus/fundus.npy',
                        metadata_path=path / '.metadata')
        patient.add_sample(sample)

    return records


def get_hadassah_transform(image_size):
    return transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size)])
