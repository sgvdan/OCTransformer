import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from oct_converter.readers import E2E
import PySimpleGUI as sg
from PIL import Image, ImageEnhance
import shutil
import os
from pathlib import Path


def align_path(path1, path2, show=False, save=False):
    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
    return align_images(img1, img2, show, save)


"""
in: 1,2
2->1
ret 2

"""


def align_images(img1_orig, img2_orig, show=False, save=False):
    img1 = ndimage.gaussian_filter(img1_orig, sigma=(10, 10), order=0)
    img2 = ndimage.gaussian_filter(img2_orig, sigma=(10, 10), order=0)

    filter_blurred_f1 = ndimage.gaussian_filter(img1, 1)
    filter_blurred_f2 = ndimage.gaussian_filter(img2, 1)

    alpha = 30
    img1 = img1 + alpha * (img1 - filter_blurred_f1)
    img2 = img2 + alpha * (img2 - filter_blurred_f2)

    _, img1 = cv.threshold(img1, 65, 255, cv.THRESH_BINARY)
    _, img2 = cv.threshold(img2, 65, 255, cv.THRESH_BINARY)

    mask = np.array([j / 5 if j % 5 == 0 else 0 for i in range(img1.shape[0]) for j in range(img1.shape[1])],
                    dtype=np.uint8).reshape(
        img1.shape[0], img1.shape[1])

    mask2 = np.array([i / 5 if i % 5 == 0 else 0 for i in range(img1.shape[0]) for j in range(img1.shape[1])],
                     dtype=np.uint8).reshape(
        img1.shape[0], img1.shape[1])

    img1 = mask * img1 + mask2 * img1
    img2 = mask * img2 + mask2 * img2

    l1 = []
    l2 = []
    for i in range(img2.shape[1]):
        w1 = np.where(img1[:, i] > 0)[0]
        w2 = np.where(img2[:, i] > 0)[0]
        if len(w1) == 0 or len(w2) == 0:
            continue
        l1.append([[i, w1.max()]])
        l2.append([[i, w2.max()]])

        l1.append([[i, w1.min()]])
        l2.append([[i, w2.min()]])
        # print(i)

    src_pts = np.array(l1)
    dst_pts = np.array(l2)

    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = img1.shape
    dst2 = cv.warpPerspective(img1_orig, M, (w, h))
    if show:
        plt.title("img1"), plt.imshow(img1_orig, 'gray'), plt.show()
        plt.title("img2"), plt.imshow(img2_orig, 'gray'), plt.show()
        plt.title("img1_play"), plt.imshow(img1, 'gray'), plt.show()
        plt.title("img2_play"), plt.imshow(img2, 'gray'), plt.show()
        plt.title("img1 transform to img2"), plt.imshow(dst2, 'gray'), plt.show()
    if save:
        cv.imwrite(
            "save.png",
            dst2)
    return dst2


def align_E2E_dir(src_e2e_path, dst_dir_path, save=False):
    src_e2e_path = Path(src_e2e_path)
    dst_dir_path = Path(dst_dir_path)
    # Sanity
    if src_e2e_path.suffix.lower() != '.e2e' or not src_e2e_path.is_file():
        return False

    # Copy original e2e volume & save each b-scan to 'images' dir
    images_path = dst_dir_path / 'images_' + str(src_e2e_path)
    if save:
        os.makedirs(images_path, exist_ok=True)
    res = []
    if save:
        shutil.copy2(src_e2e_path, dst_dir_path)
    last = None
    idx = 0
    for volume in E2E(src_e2e_path).read_oct_volume():
        last = volume.volume[0]
        volume.volume = volume.volume[1:]
        volume.volume.append(last)
        for b_scan1, b_scan2 in zip(volume.volume[:-1], volume.volume[1:]):
            img1 = Image.fromarray(b_scan1).convert('RGB')
            img2 = Image.fromarray(b_scan2).convert('RGB')
            enhancer1 = ImageEnhance.Brightness(img1)
            enhancer2 = ImageEnhance.Brightness(img2)
            img_enhanced1 = enhancer1.enhance(2)
            img_enhanced2 = enhancer2.enhance(2)

            if idx == 0:
                res.append(img_enhanced1)
                if save:
                    img_enhanced1.save(images_path / '{idx}_{pat_id}.tiff'.format(idx=idx, pat_id=volume.patient_id))
                idx += 1
                last = np.array(img_enhanced1)[:, :, 2]
                continue
            tmp1, tmp2 = np.array(img_enhanced1)[:, :, 2], np.array(img_enhanced2)[:, :, 2]
            # in: 2->1
            # ret
            # 2
            try:
                img_enhanced_aligned = align_images(tmp2, last)
                img_enhanced_aligned = Image.fromarray(img_enhanced_aligned).convert('RGB')
                res.append(img_enhanced1)
                if save:
                    img_enhanced_aligned.save(
                        images_path / '{idx}_{pat_id}.png'.format(idx=idx, pat_id=volume.patient_id))
            except Exception:
                pass
            idx += 1
            last = np.array(img_enhanced_aligned)[:, :, 2]

    return res


if __name__ == '__main__':
    pil_vol = align_E2E_dir("files", "aligned_files")

    # path = "C:/Users/Lutsk/OneDrive/Desktop/Guy/Weizmann/OCT/RAFT/demo-frames/"
    # path2 = "C:/Users/Lutsk/OneDrive/Desktop/Guy/Weizmann/OCT/data/"
    #
    # img1_orig = cv.imread(path2 + "im_6.png", cv.IMREAD_GRAYSCALE)
    # img2_orig = cv.imread(path2 + "im_13.png", cv.IMREAD_GRAYSCALE)
    # img1 = ndimage.gaussian_filter(img1_orig, sigma=(10, 10), order=0)
    # img2 = ndimage.gaussian_filter(img2_orig, sigma=(10, 10), order=0)
    #
    # filter_blurred_f1 = ndimage.gaussian_filter(img1, 1)
    # filter_blurred_f2 = ndimage.gaussian_filter(img2, 1)
    #
    # alpha = 30
    # img1 = img1 + alpha * (img1 - filter_blurred_f1)
    # img2 = img2 + alpha * (img2 - filter_blurred_f2)
    #
    # _, img1 = cv.threshold(img1, 65, 255, cv.THRESH_BINARY)
    # _, img2 = cv.threshold(img2, 65, 255, cv.THRESH_BINARY)
    #
    # mask = np.array([j / 5 if j % 5 == 0 else 0 for i in range(img1.shape[0]) for j in range(img1.shape[1])],
    #                 dtype=np.uint8).reshape(
    #     img1.shape[0], img1.shape[1])
    #
    # mask2 = np.array([i / 5 if i % 5 == 0 else 0 for i in range(img1.shape[0]) for j in range(img1.shape[1])],
    #                  dtype=np.uint8).reshape(
    #     img1.shape[0], img1.shape[1])
    #
    # img1 = mask * img1 + mask2 * img1
    # img2 = mask * img2 + mask2 * img2
    #
    # plt.title("img1"), plt.imshow(img1_orig, 'gray'), plt.show()
    # plt.title("img2"), plt.imshow(img2_orig, 'gray'), plt.show()
    #
    # plt.title("img1_play"), plt.imshow(img1, 'gray'), plt.show()
    # plt.title("img2_play"), plt.imshow(img2, 'gray'), plt.show()
    #
    # l1 = []
    # l2 = []
    # for i in range(img2.shape[1]):
    #     w1 = np.where(img1[:, i] > 0)[0]
    #     w2 = np.where(img2[:, i] > 0)[0]
    #     if len(w1) == 0 or len(w2) == 0:
    #         continue
    #     l1.append([[i, w1.max()]])
    #     l2.append([[i, w2.max()]])
    #
    #     l1.append([[i, w1.min()]])
    #     l2.append([[i, w2.min()]])
    #     print(i)
    #
    # src_pts = np.array(l1)
    # dst_pts = np.array(l2)
    #
    # M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # h, w = img1.shape
    # dst2 = cv.warpPerspective(img1_orig, M, (w, h))
    # plt.title("img1 transform to img2"), plt.imshow(dst2, 'gray'), plt.show()
