import lpips
import torch
import numpy as np
from numpy import array
import os
import imageio
from xl_op import insert_metrics_lpips
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

loss = lpips.LPIPS(net='alex')


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


def cal_lpips(ima, imb):
    ima = torch.from_numpy(np.transpose(ima, (2, 0, 1))).float()
    imb = torch.from_numpy(np.transpose(imb, (2, 0, 1))).float()
    d = loss.forward(imb, ima)
    return d.tolist()[0][0][0][0]






def cal_all_metrics(dir1, dir2):
    imgs = array(load_data(dir1, "png"))
    img_hrs = array(load_data(dir2, "png"))
    print(imgs.shape)
    print(img_hrs.shape)
    lpipss = []
    for i in range(imgs.shape[0]):
        lpipss.append(cal_lpips(imgs[i], img_hrs[i]))
    return lpipss


lpipss = cal_all_metrics("./SRGAN4270-5590", "./HR")
insert_metrics_lpips("./metrics.xlsx","SRGAN-9700",lpipss)