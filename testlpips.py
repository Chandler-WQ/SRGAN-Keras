import os

import imageio
import lpips
import numpy as np
import torch
from PIL import Image
from numpy import array
from lpips import lpips

downscale_factor = 4
image_shape = (384, 384, 3)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
# vgg19.trainable = False
# for l in vgg19.layers:
#     l.trainable = False
# loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
# loss_model.trainable = False


# def vgg_loss(y_true, y_pred):
#     return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
#
#
# def mse_loss(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)
#
#
# def first_loss(y_true, y_pred):
#     binary_cross = losses.binary_crossentropy(y_true, y_pred)
#     return mse_loss(y_true, y_pred) + 1e-3 * binary_cross
#
#
# def second_loss(y_true, y_pred):
#     binary_cross = losses.binary_crossentropy(y_true, y_pred)
#     return vgg_loss(y_true, y_pred) + 1e-3 * binary_cross


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
                rand_nums1 = np.random.randint(0, imagetem.shape[0] - 384)
                rand_nums2 = np.random.randint(0, imagetem.shape[1] - 384)
                imagetem = imagetem[rand_nums1:rand_nums1 + 384, rand_nums2:rand_nums2 + 384, :]
                if imagetem.shape[0] != 384 or imagetem.shape[1] != 384 or imagetem.shape[2] != 3:
                    continue
                if len(imagetem.shape) == 3:
                    files.append(imagetem)
    return files


def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files


# x_train = load_data("/root/userfolder/data/DIV2K_train_HR/DIV2K_train_HR/", ".png")
x_test = load_data("D:\\pyProjects\\data\\MINI\\DIV2K_valid_HR\\DIV2K_valid_HR", ".png")


# print("data loaded" + str(len(x_train)) + "\n")


def hr_images(images):
    images_hr = array(images)
    return images_hr


def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        images.append(np.array(Image.fromarray(images_real[img]).resize((images_real[img].shape[0] // downscale,
                                                                         images_real[img].shape[1] // downscale),
                                                                        resample=Image.BICUBIC)))
    images_lr = array(images)
    return images_lr


def normalize(input_data):
    return (input_data - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1.) * 127.5
    return input_data


# x_train_hr = hr_images(x_train)
# x_train_hr = normalize(x_train_hr)
# print(x_train_hr.shape)
#
# x_train_lr = lr_images(x_train, downscale_factor)
# x_train_lr = normalize(x_train_lr)
# torch.set_default_dtype(torch.FloatTensor)
x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)
print(x_test_hr.shape)

x_test_lr = lr_images(x_test, downscale_factor)
x_test_lr = normalize(x_test_lr)
print(x_test_hr.shape)
loss = lpips.LPIPS(net='alex')
print(np.transpose(x_test_lr[0], (2, 0, 1)).shape)

pil_img = denormalize(x_test_lr[0]).astype('uint8')
pil_img = Image.fromarray(pil_img)
# print(pil_img.shape)
hr_shape = (384, 384)

# 插值之后的图像
imbc = normalize(np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC)))

ima = torch.from_numpy(np.transpose(imbc, (2, 0, 1))).float()
imb = torch.from_numpy(np.transpose(x_test_hr[0], (2, 0, 1))).float()
d = loss.forward(imb, ima)

print(d)

d1= lpips(imbc,x_test_hr[0])
print(d1)