import itertools
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import util

mask_values = [10, 26, 51, 77, 102, 128, 153, 179, 204, 230]


class HadassahDataset(Dataset):
    def __init__(self, samples, chosen_labels, transformations):
        self.transformations = transformations
        self.samples = samples
        self.classes = {label: label_value for label_value, label in enumerate(chosen_labels)}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = util.get_reduced_label(sample.get_label(), self.classes.keys())

        sample_data = self.transformations(sample.get_data())
        sample_data = sample_data.unsqueeze(dim=1).expand(-1, 3, -1, -1)

        return sample_data, label

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return torch.stack([util.get_reduced_label(sample.get_label(), self.classes.keys()) for sample in self.samples])

    def get_classes(self):
        return self.classes

    def get_samples(self):
        return self.samples


class HadassahLayerSegmentationDataset(HadassahDataset):
    def __init__(self, samples, chosen_labels, transformations, layer_segmentation_transform):
        super().__init__(samples, chosen_labels, transformations)
        self.layer_segmentation_transform = layer_segmentation_transform

    def __getitem__(self, idx):
        sample_data, label = super().__getitem__(idx)
        layer_segmentation = self.layer_segmentation_transform(self.samples[idx].get_layer_segmentation_data())

        channel_mask = torch.stack([(layer_segmentation == value) for value in mask_values], dim=1).type(torch.uint8) * 255

        sample_data = (sample_data[:, 0, :, :] * 255).unsqueeze(dim=1).to(torch.uint8)
        sample_data = torch.concat([channel_mask, sample_data], dim=1)

        # import matplotlib.pyplot as plt
        # for channel in sample_data[0, :, :, :]:
        #     plt.figure()
        #     plt.imshow(channel)
        #     plt.show()
        return sample_data, label


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
    def __init__(self, name, label, volume_path=None, layer_segmentation_path=None, fundus_path=None, metadata_path=None):
        self.name = name
        self.label = label
        self.volume_path = volume_path
        self.layer_segmentation_path = layer_segmentation_path
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

    def get_layer_segmentation_data(self):
        if self.layer_segmentation_path is None:
            return None
        else:
            with open(self.layer_segmentation_path, 'rb') as layer_segmentation_file:
                return pickle.load(layer_segmentation_file)

    def get_fundus(self):
        if self.fundus_path is None:
            return None
        else:
            return np.load(self.fundus_path, mmap_mode='r')


def build_hadassah_dataset(dataset_root, layer_segmentation_root, annotations_path):
    annotations = pandas.read_excel(annotations_path)
    dataset_root = Path(dataset_root)
    layer_segmentation_root = Path(layer_segmentation_root)

    records = Records()
    for row_idx in range(1, len(annotations)):
        metadata = annotations.iloc[row_idx, :7]
        label = annotations.iloc[row_idx, 7:]

        patient_name = metadata['DR_CODE']
        sample_name = '{}_{}_{}'.format(metadata['P.I.D'], metadata['E.I.D'], metadata['S.I.D'])
        path = dataset_root / '{}/{}'.format(patient_name, sample_name)
        layer_segmentation_path = layer_segmentation_root / '{}/{}'.format(patient_name, sample_name)

        assert path.exists()

        patient = records.get_patient(patient_name)
        if patient is None:
            patient = Patient(name=patient_name)
            records.add_patient(patient)

        temp = path / 'volume/images/'
        if len(list(temp.glob('*.jpg'))) != 37:  # TODO: relevant only for -ls (layer-segmentation) dataset. Need to change
            continue

        sample = Sample(sample_name, label,
                        volume_path=path / 'volume/data.npy',
                        layer_segmentation_path=layer_segmentation_path / 'volume/layer_segmentation.pkl',
                        fundus_path=path / 'fundus/fundus.npy',
                        metadata_path=path / '.metadata')
        patient.add_sample(sample)

    return records


def get_hadassah_transform(config, mean, stdev):
    return transforms.Compose([transforms.ToTensor(), transforms.Resize(config.input_size),
                               SubsetSamplesTransform(config.num_slices), VolumeNormalize(),
                               # transforms.Normalize(mean, stdev), EnhanceBrightness(3),
                               transforms.RandomHorizontalFlip(p=0.5)])


def setup_hadassah(config):
    records = build_hadassah_dataset(dataset_root=config.hadassah_root,
                                     layer_segmentation_root=config.hadassah_layer_segmentation_root,
                                     annotations_path=config.hadassah_annotations)

    control, study = records.slice_samples(dict(list(zip(config.labels, itertools.repeat(1)))))

    train_control, eval_control, test_control = util.split_list(control, [config.train_size,
                                                                          config.eval_size,
                                                                          config.test_size])
    train_study, eval_study, test_study = util.split_list(study, [config.train_size,
                                                                  config.eval_size,
                                                                  config.test_size])

    if config.layer_segmentation_input:
        layer_segmentation_transform = transforms.Compose([SubsetSamplesTransform(config.num_slices),
                                                           transforms.Resize(config.input_size, interpolation=InterpolationMode.NEAREST),
                                                           transforms.RandomHorizontalFlip(p=1)])
        dataset_handler = partial(HadassahLayerSegmentationDataset, layer_segmentation_transform=layer_segmentation_transform)
    else:
        dataset_handler = HadassahDataset

    # Test Loader
    transform = get_hadassah_transform(config, 7.85263045909125, 19.988803667371428)
    test_dataset = dataset_handler([*test_control, *test_study], chosen_labels=config.labels, transformations=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # Evaluation Loader
    transform = get_hadassah_transform(config, 8.69875640790064, 22.536245302856397)
    eval_dataset = dataset_handler([*eval_control, *eval_study], chosen_labels=config.labels, transformations=transform)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size)

    # Train Loader
    transform = get_hadassah_transform(config, 8.165419910168131, 21.35598863335939)
    train_dataset = dataset_handler([*train_control, *train_study], chosen_labels=config.labels, transformations=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader


class SubsetSamplesTransform(object):
    def __init__(self, num_slices):
        self.num_slices = num_slices

    def __call__(self, sample):
        indices = torch.tensor(range(18-self.num_slices//2, 18 + self.num_slices//2+1))
        new_sample = sample.index_select(dim=0, index=indices)
        return new_sample


class VolumeNormalize(object):
    def __call__(self, sample):
        norm_sample = sample

        norm_sample -= sample.min()
        norm_sample /= sample.max()

        return norm_sample
