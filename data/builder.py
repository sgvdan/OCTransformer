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


def main():
    parser = argparse.ArgumentParser(description='Build new OCT (cached) dataset')
    parser.add_argument('--path', type=str, nargs='+', required=True,
                        help='Specify the data\'s path(s). Should correspond to `label`')
    parser.add_argument('--label', type=str, nargs='+', required=True,
                        help='Specify the label\'s name. Should correspond to `path`')
    parser.add_argument('--name', type=str, required=True,
                        help='The dataset\'s name')
    parser.add_argument('--action', type=str, choices=['create', 'delete'], default='create',
                        help='Choose what to do with the cache')
    parser.add_argument('--label_type', choices=['one_hot', 'scalar'], default='scalar',
                        help='Choose labels kind')

    args = parser.parse_args()

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


if __name__ == '__main__':
    main()
