import itertools
import pickle
import pandas
import os
from functools import partial
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import util
from torchvision.transforms.functional import adjust_brightness
from mgu_models.net_builder import net_builder
import data.seg_transforms as dt

mask_values = [10, 26, 51, 77, 102, 128, 153, 179, 204, 230]
fg_mask_values = [10, 77, 102, 128, 153, 179, 204, 230]


class HadassahDataset(Dataset):
    def __init__(self, samples, chosen_labels, transformations):
        self.transformations = transformations
        self.samples = samples
        self.classes = {label: label_value for label_value, label in enumerate(chosen_labels)}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = util.get_reduced_label(sample.get_label(), self.classes.keys())

        sample_data = transforms.ToTensor()(sample.get_data())
        sample_data = self.transformations(sample_data)
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
    def __init__(self, samples, chosen_labels, transformations, mode, layer_segmentation_transform):
        super().__init__(samples, chosen_labels, transformations)
        self.layer_segmentation_transform = layer_segmentation_transform
        self.mode = mode

    def __getitem__(self, idx):
        sample_data, label = super().__getitem__(idx)
        sample_data = sample_data[:, 0, :, :].unsqueeze(dim=1)  # One channel is enough

        layer_segmentation = torch.tensor(self.samples[idx].get_layer_segmentation_data()).moveaxis(3, 0)
        layer_segmentation = self.layer_segmentation_transform(layer_segmentation)

        if self.mode == 'none':
            sample_data = sample_data
        elif self.mode == 'zeros':
            mask = torch.zeros_like(layer_segmentation)
            sample_data = torch.concat([mask, sample_data], dim=1)
        elif self.mode == 'confidence':
            sample_data = torch.concat([layer_segmentation, sample_data], dim=1)
        elif self.mode == 'confidence-only':
            sample_data = layer_segmentation
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

    def split_samples(self):
        samples = self.get_samples()
        train = [sample for sample in samples if sample.metadata['GROUP'].lower() == 'train']
        eval = [sample for sample in samples if sample.metadata['GROUP'].lower() == 'eval']
        test = [sample for sample in samples if sample.metadata['GROUP'].lower() == 'test']

        for label in ['DR', 'DME', 'IRF']:
            print('-- ' + label + ' --')
            trn = len([sample for sample in train if sample.get_label()[label] == 1])
            evl = len([sample for sample in eval if sample.get_label()[label] == 1])
            tst = len([sample for sample in test if sample.get_label()[label] == 1])
            print("TRAIN: {}/{}, EVAL: {}/{}, TEST: {}/{}".format(trn, len(train),
                                                                  evl, len(eval),
                                                                  tst, len(test)))

        return train, eval, test


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
    def __init__(self, name, label, volume_path=None, layer_segmentation_path=None, fundus_path=None, metadata=None):
        self.name = name
        self.label = label
        self.volume_path = volume_path
        self.layer_segmentation_path = layer_segmentation_path
        self.fundus_path = fundus_path
        self.metadata = metadata

    def get_label(self):
        return self.label

    def get_metadata(self):
        return self.metadata

    def get_data(self):
        if self.volume_path is None:
            return None
        else:
            return np.load(self.volume_path, mmap_mode='r')

    def get_layer_segmentation_data(self):
        if self.layer_segmentation_path is None:
            return None
        else:
            return np.load(self.layer_segmentation_path, mmap_mode='r')

    def get_fundus(self):
        if self.fundus_path is None:
            return None
        else:
            return np.load(self.fundus_path, mmap_mode='r')


def build_hadassah_dataset(dataset_root, annotations_path, dest_path, add_layer_segmentation=False):
    annotations = pandas.read_excel(annotations_path)
    dataset_root = Path(dataset_root)
    dest_path = Path(dest_path)

    if add_layer_segmentation:
        layer_seg_transform = LayerSegCreator()

    records = Records()
    for row_idx in tqdm(range(1, len(annotations))):
        metadata = annotations.iloc[row_idx, :6]
        label = annotations.iloc[row_idx, 6:]

        patient_name = metadata['DR_CODE']
        file_name = metadata['FileName']

        if file_name != file_name:  # file_name is nan
            continue

        data_path = dataset_root / '{}/{}'.format(patient_name, file_name)
        # assert data_path.exists()

        slices = list(util.glob_re(r'bscan_\d+.tiff', os.listdir(data_path)))
        if len(slices) != 37:
            continue

        patient = records.get_patient(patient_name)
        if patient is None:
            patient = Patient(name=patient_name)
            records.add_patient(patient)

        volume_data_path = dest_path / '{}/{}/data-volume.npy'.format(patient_name, file_name)
        if not volume_data_path.exists():
            volume_data = []
            for slice in util.sorted_nicely(slices):
                # Pre-process image and append
                image = Image.open(data_path / slice)
                # image = adjust_brightness(image, 2)
                image_data = np.array(image)
                # image_data = mask_out_noise(image_data)
                volume_data.append(image_data)
            volume_data = np.stack(volume_data, axis=2)
            print('Created volume {patient_name}:{file_name}'.format(patient_name=patient_name, file_name=file_name))

            os.makedirs(volume_data_path.parent, exist_ok=True)
            with open(volume_data_path, 'wb+') as volume_file:
                np.save(volume_file, volume_data)

        volume_ls_path = dest_path / '{}/{}/ls-volume.npy'.format(patient_name, file_name)
        if add_layer_segmentation and not volume_ls_path.exists():
            with open(volume_data_path, 'rb') as volume_file:
                volume_data = np.load(volume_file)

            volume_ls = layer_seg_transform(volume_data)

            with open(volume_ls_path, 'wb+') as volume_ls_file:
                np.save(volume_ls_file, volume_ls)

            print('Created layer-segmentation volume {patient_name}:{file_name}'
                  .format(patient_name=patient_name, file_name=file_name))

        sample = Sample(file_name, label,
                        volume_path=volume_data_path,
                        layer_segmentation_path=volume_ls_path,
                        metadata=metadata,
                        fundus_path=data_path / 'slo.tiff')
        patient.add_sample(sample)

    return records


def get_hadassah_transform(config):
    return transforms.Compose([transforms.Resize(config.input_size),
                               SubsetSamplesTransform(config.num_slices)])


def get_layer_segmentation_transform(config):
    return transforms.Compose([transforms.Resize(config.input_size, interpolation=InterpolationMode.NEAREST),
                               SubsetSamplesTransform(config.num_slices)])


def setup_hadassah(config):
    records = build_hadassah_dataset(dataset_root=config.hadassah_root, dest_path=config.hadassah_dest,
                                     annotations_path=config.hadassah_annotations,
                                     add_layer_segmentation=config.layer_segmentation_input)

    train_group, eval_group, test_group = records.split_samples()

    if config.layer_segmentation_input:
        dataset_handler = partial(HadassahLayerSegmentationDataset, chosen_labels=config.labels,
                                  transformations=get_hadassah_transform(config), mode=config.layer_segmentation_input,
                                  layer_segmentation_transform=get_layer_segmentation_transform(config))
    else:
        dataset_handler = partial(HadassahDataset, chosen_labels=config.labels,
                                  transformations=get_hadassah_transform(config))

    # Test Loader
    test_dataset = dataset_handler(test_group)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # Evaluation Loader
    eval_dataset = dataset_handler(eval_group)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size)

    # Train Loader
    train_dataset = dataset_handler(train_group)
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


def mask_out_noise(image, snr=10):
    denoised_image = denoise_tv_chambolle(image, weight=1)
    fft = abs(np.fft.fft(denoised_image, axis=-1))  # FFT on the LAST axis

    signal = np.max(fft[:, 1:], axis=-1)  # Take the spectral breakdown "envelope", aside from the "sum" element
    signal /= np.max(signal)

    signal_boundaries = np.argwhere(signal > (1/snr)).squeeze()
    _min, _max = signal_boundaries[0], signal_boundaries[-1]

    _min = max(0, _min - 15)
    _max = min(_max + 15, image.shape[-2] - 1)

    image[:_min, :] = 0
    image[_max:, :] = 0

    return image


class LayerSegCreator:
    def __init__(self):
        # load the model
        net = net_builder('tsmgunet')
        self.model = torch.nn.DataParallel(net).cuda()
        checkpoint = torch.load('/home/projects/ronen/sgvdan/workspace/projects/MGU-Net/result/mat_dataset/train/tsmgunet_0.001_t1/model/model_best.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.transform = dt.Compose([dt.ToTensor(), dt.Normalize(mean=[37.49601251371136, 37.49601251371136, 37.49601251371136],
                                                                 std=[51.04429269367485, 51.04429269367485, 51.04429269367485])])

    def __call__(self, volume_data):
        layer_segmentation = []

        with torch.no_grad():
            for image_data in np.moveaxis(volume_data, 2, 0):
                image = Image.fromarray(image_data).resize((1024, 992))
                image_data = list(self.transform(image))[0]
                image_var = Variable(image_data).cuda().unsqueeze(dim=0)
                _,_,output_seg = self.model(image_var)
                pred_seg = torch.nn.functional.softmax(output_seg, dim=1)
                pred_seg = pred_seg.cpu().data.numpy()
                layer_segmentation.append(pred_seg[0])

        layer_segmentation = np.stack(layer_segmentation, axis=3)

        return layer_segmentation

