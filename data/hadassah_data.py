import numpy as np
from torch.utils.data import Dataset


class HadassahDataset(Dataset):
    def __init__(self, patients, transformations):
        self.transformations = transformations

        self.samples = []
        for patient in patients:
            self.samples += patient.get_samples()

    def __getitem__(self, idx):
        sample = self.patient.get_samples()[idx]

        sample_data = sample.get_data()
        sample_label = sample.get_label()

        sample_data = self.transformations(sample_data)
        sample_data = sample_data.unsqueeze(dim=1).expand(-1, 3, -1, -1)

        return sample_data, sample_label

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [sample.get_label() for sample in self.samples]

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

    def slice_samples(self, labels):
        """
        Performs a control-study cut of the patients according to labels
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
    def __init__(self, name, label, volume_path=None, fundus_path=None, metadata=None):
        self.name = name
        self.label = label
        self.volume_path = volume_path
        self.fundus_path = fundus_path
        self.metadata = metadata

    def set_metadata(self, metadatum):
        self.metadata += metadatum

    def get_label(self):
        return self.label

    def get_metadata(self):
        return self.metadata

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
