import json
import os

import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from torch.nn.functional import conv2d
from PIL import Image, ImageEnhance
import numpy as np
from torchvision import transforms
import torch
from pathlib import Path


def smoothene_contours(contours):
    smoothened_contours = []
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1.0, per=1, quiet=2)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
    return smoothened_contours


def filter_contours(contours, smallest_contour_len=30):
    filter_contours = []
    for contour in contours:
        x, y = contour.T[:, 0, :]
        # contours are complex and do not touch the boundaries
        if (len(contour) > smallest_contour_len) and \
                ((x.min() != 0) and (y.min() != 0) and (x.max() != 255) and (y.max() != 255)):
            filter_contours.append(contour)
    return filter_contours


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_t = move2cpu(im_t)
    if len(im_t.shape) == 3:
        im_t = np.transpose(im_t, (1, 2, 0))
    im_np = np.clip(np.round(im_t * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def convolve(images_tensor, kernel):
    kernel = np.array(kernel)
    kernel_height, kernel_width = kernel.shape
    kernel = torch.from_numpy(kernel).float().cuda().view(1, 1, kernel_height, kernel_width)

    return conv2d(input=images_tensor, weight=kernel / torch.norm(kernel), padding='same')


def read_images(input_dir):
    images_list = Path(input_dir).glob('*.jpg')
    to_tensor = transforms.ToTensor()

    images, names = [], []
    for image_path in images_list:
        pil_image = Image.open(image_path)
        image = to_tensor(pil_image)[0].unsqueeze(dim=0)
        images.append(image.cuda())
        names.append(image_path.stem)

    return torch.stack(images), names


def save_images(images, names, output_dir):
    to_image = transforms.ToPILImage()
    os.makedirs(output_dir, exist_ok=True)
    for idx, image in enumerate(images):
        to_image(image).save(output_dir + '/' + names[idx] + '.jpg')


def thresh(images, threshold):
    images = np.uint8(cv2.normalize(tensor2im(images), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

    masks = []
    for image in images:
        _, thresh_img = cv2.threshold(src=image.squeeze(), thresh=threshold * 255, maxval=255, type=cv2.THRESH_BINARY)
        masks.append(thresh_img)

    return masks


def sharpen(images):
    to_image = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    new_images = []
    for image in images:
        pil_image = to_image(image)
        img = ImageEnhance.Contrast(pil_image)
        contrast_image = img.enhance(10)
        tensor_image = to_tensor(contrast_image)
        new_images.append(tensor_image)

    return torch.stack(new_images)

if __name__ == '__main__':
    # input_dir = '../../datasets/hadassah/std/DR_00065/17132_158663_529050/volume/images'
    # input_dir = '../../datasets/hadassah/std/DR_01380/19801_143118_474628/volume/images'
    input_dir = '../../datasets/hadassah/std/DR_00385/13673_74702_257508/volume/images'

    output_dir = './output/preprocess/28'

    # kernel = [[1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1],
    #           [1, 1, 1, 1, 1]]

    kernel = np.ones((10, 10)).tolist()

    images, names = read_images(input_dir)

    convolved_images = convolve(images, kernel)

    thresh_images = thresh(convolved_images, threshold=0.9)

    # save_images(thresh_images, [name + '-mask' for name in names], output_dir)
    # save_images(convolved_images, [name + '-convolved' for name in names], output_dir)
    # save_images(images, [name + '-raw' for name in names], output_dir)

    for name, image in zip(names, images):
        image = image.squeeze().detach().cpu()
        # fig = plt.figure(figsize=(8, 8))
        # plt.plot(np.arange(image.shape[0]), image.sum(axis=1))
        # plt.savefig(output_dir + '/' + name + '-histogram.png')

        fig = plt.figure(figsize=(8, 8))
        fft = abs(np.fft.fft(image, axis=1))  # FFT is performed on the LAST axis
        signal = np.max(fft[:, 1:], axis=1)  # Take the spectral breakdown "envelope", aside from the "sum" element

        smooth_signal = np.convolve(signal, np.ones(30))
        gradient = np.gradient(smooth_signal)
        gradient = np.gradient(gradient)  # second gradient...
        gradient = np.convolve(gradient, np.ones(30))

        gradient -= np.mean(gradient)
        gradient = np.abs(gradient)
        gradient /= np.max(gradient)

        signal_boundaries = np.argwhere(gradient[:] > 0.1).squeeze()
        _min, _max = signal_boundaries[0], signal_boundaries[-1]

        outlier = np.diff(signal_boundaries) > 50
        if any(outlier):
            _min = signal_boundaries[np.nonzero(outlier)[0] + 1]

        _min = max(0, _min - 10)
        _max = min(_max + 10, image.shape[0])
        image[_min, :] = 1
        image[_max, :] = 1
        save_images([image], [name + '-bounds'], output_dir)

        plt.plot(gradient)
        plt.savefig(output_dir + '/' + name + 'grad-fft.png')

    with open(output_dir + '/kernel.json', 'w+') as f:
        json.dump(kernel, f)
    with open(output_dir + '/src.json', 'w+') as f:
        json.dump(input_dir, f)


    # # find contours
    # contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #
    # try:
    #     contours = smoothene_contours(contours=contours)
    # except:
    #     print("Failed to smoothen contours :/    continuing...")
    #
    # # draw only filled contours
    # mask = np.zeros(image.shape)
    # cv2.fillPoly(img=mask, pts=contours, color=1)
