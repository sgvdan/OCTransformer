# import json
# import os
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# import torch
# from pathlib import Path
# from util import move2cpu
#
# from skimage.restoration import denoise_tv_chambolle
#
#
# def read_images(input_dir):
#     images_list = Path(input_dir).glob('*.jpg')
#     to_tensor = transforms.ToTensor()
#
#     images, names = [], []
#     for image_path in images_list:
#         pil_image = Image.open(image_path)
#         image = to_tensor(pil_image)[0]
#         images.append(image.cuda())
#         names.append(image_path.stem)
#
#     return torch.stack(images), names
#
#
# def save_images(images, names, output_dir):
#     to_image = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
#     for idx, image in enumerate(images):
#         to_image(image).save(output_dir + '/' + names[idx] + '.jpg')
#
#
# def mask_out_noise(images, snr=10):
#     denoised_images = denoise_tv_chambolle(images, weight=1)
#     fft = abs(np.fft.fft(denoised_images, axis=-1))  # FFT on the LAST axis
#
#     signal = np.max(fft[:, 1:], axis=-1)  # Take the spectral breakdown "envelope", aside from the "sum" element
#     signal /= np.max(signal)
#
#     signal_boundaries = np.argwhere(signal > (1/snr)).squeeze()
#     _min, _max = signal_boundaries[0], signal_boundaries[-1]
#
#     _min = max(0, _min - 10)
#     _max = min(_max + 10, image.shape[-2] - 1)
#
#     image[:_min, :] = 0
#     image[_max:, :] = 0
#
#     return image
#
#
# if __name__ == '__main__':
#     # input_dir = '../../datasets/hadassah/std/DR_00065/17132_158663_529050/volume/images'
#     # input_dir = '../../datasets/hadassah/std/DR_01380/19801_143118_474628/volume/images'
#     input_dir = '../../datasets/hadassah/std/DR_00385/13673_74702_257508/volume/images'
#
#     output_dir = './output/preprocess/37-clean-code'
#     os.makedirs(output_dir, exist_ok=True)
#
#     images, names = read_images(input_dir)
#     images = move2cpu(images)
#
#     for image in images:
#         mask_out_noise(image, snr=10)
#
#     save_images(images, [name + '-masked' for name in names], output_dir)
#
#     with open(output_dir + '/src.json', 'w+') as f:
#         json.dump(input_dir, f)
