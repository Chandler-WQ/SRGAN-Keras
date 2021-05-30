import matplotlib.pyplot as plt
from numpy import array
import os
import imageio
from xl_op import read_data


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


# 这里是对比不同的重建方法
img_bcs = array(load_data("./bc", "png"))
img_lrs = array(load_data("./LR", "png"))
# img_hrs = array(load_data("./HR", "png"))
img_ESR = array(load_data("./ESRGAN-3900", "png"))
img_SR = array(load_data("./SRGAN-3900", "png"))

niqeBC = read_data("./metrics.xlsx", "B")
ssimBC = read_data("./metrics.xlsx", "C")
psnrBC = read_data("./metrics.xlsx", "D")
lpipsBC = read_data("./metrics.xlsx", "E")

niqeESR = read_data("./metrics.xlsx", "R")
ssimESR = read_data("./metrics.xlsx", "S")
psnrESR = read_data("./metrics.xlsx", "T")
lpipsESR = read_data("./metrics.xlsx", "U")

niqeSR = read_data("./metrics.xlsx", "AL")
ssimSR = read_data("./metrics.xlsx", "AM")
psnrSR = read_data("./metrics.xlsx", "AN")
lpipsSR = read_data("./metrics.xlsx", "AO")

for i in range(img_bcs.shape[0]):
# for i in range(2):
    fig, ax = plt.subplots(2, 2, figsize=(40, 40))
    axes = ax.flatten()
    images = {
        'LR': img_lrs[i],
        'BC': img_bcs[i],
        'SRGAN': img_SR[i],
        "NEWSR": img_ESR[i],
    }
    for n, (title, img) in enumerate(images.items()):
        if n is 0:
            axes[n].set_title(
                "{} - {} - niqe{} - lpips{} - ssim{:} - psnr{:}".format(title, img.shape, "None", "None", "None",
                                                                        "None"), fontsize=30)
        if n is 1:
            axes[n].set_title(
                "{} - {} - niqe{:.6f} - lpips{:.6f} - ssim{:.4f} - psnr{:.4f}".format(title, img.shape, niqeBC[i + 2],
                                                                                      lpipsBC[i + 2], ssimBC[i + 2],
                                                                                      psnrBC[i + 2]), fontsize=30)
        if n is 2:
            axes[n].set_title(
                "{} - {} - niqe{:.6f} - lpips{:.6f} - ssim{:.4f} - psnr{:.4f}".format(title, img.shape, niqeSR[i + 2],
                                                                                      lpipsSR[i + 2], ssimSR[i + 2],
                                                                                      psnrSR[i + 2]), fontsize=28)
        if n is 3:
            axes[n].set_title(
                "{} - {} - niqe{:.6f} - lpips{:.6f} - ssim{:.4f} - psnr{:.4f}".format(title, img.shape, niqeESR[i + 2],
                                                                                      lpipsESR[i + 2], ssimESR[i + 2],
                                                                                      psnrESR[i + 2]), fontsize=28)
        axes[n].imshow(img)
        axes[n].axis('off')
    fig.savefig("./ims/444-{}.png".format(i))
    plt.close()

# img_ESR900 = array(load_data("./ESRGAN-900", "png"))
# img_ESR1900 = array(load_data("./ESRGAN-1900", "png"))
# img_ESR2900 = array(load_data("./ESRGAN-2900", "png"))
# img_ESR3900 = array(load_data("./ESRGAN-3900", "png"))
# for i in range(img_ESR900.shape[0]):
# # for i in range(1):
#     fig, ax = plt.subplots(2, 2, figsize=(40, 40))
#     axes = ax.flatten()
#     images = {
#         'Epoch900': img_ESR900[i],
#         'Epoch1900': img_ESR1900[i],
#         'Epoch2900': img_ESR2900[i],
#         'Epoch3900': img_ESR3900[i],
#     }
#     for n, (title, img) in enumerate(images.items()):
#         axes[n].imshow(img)
#         axes[n].axis('off')
#         axes[n].set_title(title,fontsize=50)
#     fig.savefig("./ims/epoch-{}.png".format(i))
#     plt.close()