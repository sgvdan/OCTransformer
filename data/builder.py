from torch.nn.functional import one_hot
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from oct_converter.readers import E2E
import argparse

from cache import Cache

LABELS = {'HEALTHY': torch.tensor(0), 'SICK': torch.tensor(1)}


def build_volume_cache(cache, path, label):
    counter = 0
    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if sample.is_file():
            for scan in E2E(sample).read_oct_volume():
                # Ignore volumes with 0-size
                validity = [isinstance(tomogram, np.ndarray) for tomogram in scan.volume]
                if not all(validity) or len(validity) != 37:  # TODO: CHANGE THIS TO ALL SIZES
                    continue
                try:
                    volume = np.moveaxis(scan.volume, 0, 2)  # Works best with Pytorch' transformation
                    cache.append((volume, label))
                    counter += 1
                except Exception as ex:
                    print("Ignored volume {0} in sample {1}. An exception of type {2} occurred. \
                               Arguments:\n{1!r}".format(scan.patient_id, sample, type(ex).__name__, ex.args))
                    continue

    print("Overall {counter} proper volumes were added for label [{label}] in cache [{cache_name}]"
          .format(counter=counter, label=label, cache_name=cache.name))


def print_stats(path):
    # Obtain # of E2E files in path
    total_e2e_count = 0
    total_volume_count = 0
    total_scans_count = 0
    corrupt_volumes_count = 0
    volume_count_histogram = {}
    scans_count_histogram = {}

    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if sample.is_file():
            total_e2e_count += 1

            volumes = E2E(sample).read_oct_volume()

            # Keep the histogram of volumes in each E2E file
            number_of_volumes = len(volumes)
            if number_of_volumes not in volume_count_histogram:
                volume_count_histogram[number_of_volumes] = 0
            volume_count_histogram[number_of_volumes] += 1

            # Keep total volume count
            total_volume_count += number_of_volumes

            # How many scans in each volume
            for scan in volumes:
                validity = [isinstance(tomogram, np.ndarray) for tomogram in scan.volume]
                if not all(validity):
                    corrupt_volumes_count += 1
                    continue

                number_of_scans = len(validity)
                if number_of_scans not in scans_count_histogram:
                    scans_count_histogram[number_of_scans] = 0
                scans_count_histogram[number_of_scans] += 1

                # Keep total scan count
                total_scans_count += number_of_scans

    print('TOTAL E2E COUNT: {}\n'
          'TOTAL VOLUME COUNT: {}\n'
          'TOTAL CORRUPT VOLUMES COUNT: {}\n'
          'TOTAL SCANS COUNT: {}\n'
          'VOLUME COUNT HISTOGRAM: {}\n'
          'SCANS COUNT HISTOGRAM: {}\n'.format(total_e2e_count,
                                               total_volume_count,
                                               corrupt_volumes_count,
                                               total_scans_count,
                                               volume_count_histogram,
                                               scans_count_histogram)
          )


def main():
    parser = argparse.ArgumentParser(description='Build new OCT (cached) dataset')
    parser.add_argument('--action', type=str, choices=['create', 'assess'], default='create',
                        help='Choose what to do with the dataset (create/delete cache or assess dataset)')
    parser.add_argument('--path', type=str, nargs='+', required=True,
                        help='Specify the data\'s path(s). Should correspond to `label`')
    parser.add_argument('--label', type=str, nargs='+', default=None,
                        help='Specify the label\'s name. Should correspond to `path`')
    parser.add_argument('--name', type=str, default=None,
                        help='The dataset\'s name')
    parser.add_argument('--label_type', choices=['one_hot', 'scalar'], default='scalar',
                        help='Choose labels kind')

    args = parser.parse_args()

    if args.action == 'create':
        assert args.name is not None and args.label is not None

        cache = Cache(args.name)
        for idx, (path, label_name) in enumerate(zip(args.path, args.label)):
            if args.label_type == 'one_hot':
                label = one_hot(idx, num_classes=len(args.label))
            elif args.label_type == 'scalar':
                label = torch.tensor(idx)
            cache.set_class(label_name, label)

            print('Appending [{path}] to label [{label_name}:{label_value}] in cache [{cache_name}]'
                  .format(path=path, label_name=label_name, label_value=label, cache_name=cache.name))
            build_volume_cache(cache, path, label)

    elif args.action == 'assess':
        for path in args.path:
            print('\n--STATS ({})---\n'.format(path))
            print_stats(path)


if __name__ == '__main__':
    main()
