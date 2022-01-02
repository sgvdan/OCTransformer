import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class HadassahDataset(Dataset):
    def __init__(self, patients, transformations):
        self.patients_datasets = [PatientDataset(patient, transformations) for patient in patients]
        self.indices = np.empty_like(patients)

        for idx, patient in enumerate(self.patients):
            self.indices[idx] = np.cumsum(self.indices) + len(patient)

    def __getitem__(self, idx):
        temp = idx - self.indices
        patient_num = np.argmax(temp <= 0)
        volume_index = temp[patient_num]

        return self.patients_datasets[patient_num][volume_index]

    def __len__(self):
        return sum([len(patient) for patient in self.patients])


class PatientDataset(Dataset):
    def __init__(self, patient, transformations):
        super().__init__()
        self.patient = patient
        self.transformations = [transforms.ToTensor(), *transformations]

    def __len__(self):
        return len(self.patient.get_volumes())

    def __getitem__(self, idx):
        volume = self.patient.get_volumes()[idx]

        volume_data = np.load(volume.reside_path, mmap_mode='r')
        volume_data = self.transformations(volume_data)
        volume_data = volume_data.unsqueeze(dim=1).expand(-1, 3, -1, -1)

        return volume_data, self.patient.label


def slice_patients(patients, slice_label):
    control, study = [], []
    for patient in patients:
        correspondences = [value == patient.label[key] for key, value in slice_label]
        if all(correspondences):
            study.append(patient)
        else:
            control.append(patient)

    return control, study
