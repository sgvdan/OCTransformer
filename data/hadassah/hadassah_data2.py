import numpy as np
from torch.utils.data import Dataset


class HadassahDataset(Dataset):
    def __init__(self, patients, label_name, transformations):
        self.patients_datasets = []
        self.indices = np.zeros(1)

        for patient in patients:
            self.patients_datasets.append(PatientDataset(patient, label_name, transformations))
            self.indices = np.append(self.indices, self.indices[-1] + len(patient))

    def cast_index(self, idx):
        temp = idx - self.indices
        patient_num = int(np.argmax(temp < 0) - 1)
        volume_index = int(temp[patient_num])

        return patient_num, volume_index

    def __getitem__(self, idx):
        if idx >= self.indices[-1]:
            raise StopIteration

        patient_num, volume_index = self.cast_index(idx)
        return self.patients_datasets[patient_num][volume_index]

    def get_label(self, idx):
        patient_num, volume_index = self.cast_index(idx)
        return self.patients_datasets[patient_num].get_label(volume_index)

    def __len__(self):
        return sum([len(patient_dataset) for patient_dataset in self.patients_datasets])

    def get_labels(self):
        return [self.get_label(idx) for idx in range(len(self))]

    def get_classes(self):
        # TODO: REMOVE THIS! SOMEHOW TREAT IT WITH LABEL NAME
        return {'DME_HEALTHY': 0, 'DME_SICK': 1}


class PatientDataset(Dataset):
    def __init__(self, patient, label_name, transformations):
        super().__init__()
        self.patient = patient
        self.transformations = transformations
        self.label_name = label_name

    def __len__(self):
        return len(self.patient.get_volumes())

    def get_label(self, idx):
        # Prefer right if exists (FOR NOW)
        label = self.patient.right_label if self.patient.right_label is not None else self.patient.left_label
        # TODO: Per volume, save (when building) whether it is left/right and use the corresponding label here instead
        return label[self.label_name]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration

        volume = self.patient.get_volumes()[idx]
        label = self.get_label(idx)

        volume_data = np.load(volume.reside_path, mmap_mode='r')
        volume_data = np.swapaxes(volume_data, 0, 2)  # conform to torch image resize
        volume_data = self.transformations(volume_data)
        volume_data = volume_data.unsqueeze(dim=1).expand(-1, 3, -1, -1)

        return volume_data, label
