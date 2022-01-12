import torch
from torch.nn.functional import one_hot
from torchvision.io import read_image
from tqdm import tqdm
from pathlib import Path
import argparse
from data.cache import Cache


def build_kermany_cache(cache, path, label):
    counter = 0
    for image_path in tqdm(Path(path).iterdir()):
        image = read_image(str(image_path))
        cache.append((image, label))
        counter += 1

    print("Overall {counter} images were added for label [{label}] in cache [{cache_name}]"
          .format(counter=counter, label=label, cache_name=cache.name))


def main():
    parser = argparse.ArgumentParser(description='Build Kermany (cached) dataset')
    parser.add_argument('--label', type=str, nargs='+', default=None,
                        help='Specify the label\'s name. Should correspond to `path`')
    parser.add_argument('--path', type=str, nargs='+', required=True,
                        help='Specify the data\'s path(s). Should correspond to `label`')
    parser.add_argument('--name', type=str, default=None,
                        help='The dataset\'s name')
    parser.add_argument('--label_type', choices=['one_hot', 'scalar'], default='scalar',
                        help='Choose labels kind')

    args = parser.parse_args()

    assert args.name is not None and args.label is not None

    cache = Cache(args.name)
    labels = cache.get_classes()
    for idx, (path, label_name) in enumerate(zip(args.path, args.label)):
        if label_name not in labels:
            if args.label_type == 'one_hot':
                label = one_hot(idx, num_classes=len(args.label))
            elif args.label_type == 'scalar':
                label = torch.tensor(idx)

            cache.set_class(label_name, label)
            labels[label_name] = label
        else:
            label = labels[label_name]

        print('Appending [{path}] to label [{label_name}:{label_value}] in cache [{cache_name}]'
              .format(path=path, label_name=label_name, label_value=label, cache_name=cache.name))
        build_kermany_cache(cache, path, label)


if __name__ == '__main__':
    main()
