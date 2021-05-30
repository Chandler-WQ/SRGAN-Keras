from skimage.measure import compare_psnr, compare_ssim
from niqe import calculate_niqe
from numpy import array
import os
import imageio
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext):
    files = []
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                imagetem = imageio.imread(os.path.join(d, f))
                if len(imagetem.shape) == 3:
                    files.append(imagetem)
    return files


def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files


def cal_niqe(img):
    return calculate_niqe(img, 0)[0][0]


def cal_all_metrics(dir1, dir2):
    imgs = array(load_data(dir1, "png"))
    img_hrs = array(load_data(dir2, "png"))
    print(imgs.shape)
    print(img_hrs.shape)
    niqes = []
    for i in range(imgs.shape[0]):
        niqes.append(cal_niqe(imgs[i]))
    return niqes


niqes = cal_all_metrics("./ESRGAN-900", "./HR")

print(niqes)
