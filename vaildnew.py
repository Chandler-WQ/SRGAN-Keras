import matplotlib.pyplot as plt
from Emodel import Generator, Discriminator

# plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import losses
import keras.backend as K
import imageio
import numpy as np
from numpy import array
import os
from util import DataLoader, plot_test_images, plot_test_only, plot_bigger_images, compute_metric, plot_test_image, \
    plot_test_image_with_refer,save_test_image,save_test_image_with_hr_bc_lr
from keras.models import load_model
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vaild_path = "D:\\pyProjects\\data\\DIV2K_valid_HR\\\DIV2K_valid_HR"
down_para = 4
image_shape = (384, 384, 3)

# vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
# vgg19.trainable = False
# for l in vgg19.layers:
#     l.trainable = False
# loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
# loss_model.trainable = False

np.random.seed(10)


# def vgg_loss(y_true, y_pred):
#     return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


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


def lr_images(images_real):
    images = []

    for img in range(len(images_real)):
        a = array(images_real[img])
        images.append(np.array(Image.fromarray(images_real[img]).resize((int(a.shape[0] / down_para),
                                                                         int(a.shape[1] / down_para)),
                                                                        resample=Image.BICUBIC)))
    images_lr = array(images)
    return images_lr


def hr_images(images):
    images_hr = array(images)
    return images_hr


def normalize(input_data):
    return (input_data - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1.) * 127.5
    return input_data


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext, Hr):
    files = []
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                imagetem = imageio.imread(os.path.join(d, f))
                if Hr:
                    if imagetem.shape[0] < 384 or imagetem.shape[1] < 384 or imagetem.shape[2] != 3:
                        continue
                    rand_nums1 = np.random.randint(0, imagetem.shape[0] - 384)
                    rand_nums2 = np.random.randint(0, imagetem.shape[1] - 384)
                    imagetem = imagetem[rand_nums1:rand_nums1 + 384, rand_nums2:rand_nums2 + 384, :]
                    # imagetem = imagetem[0:imagetem.shape[0] // 4 * 4, 0:imagetem.shape[1] // 4 * 4, :]
                    if len(imagetem.shape) == 3:
                        files.append(imagetem)
                else:
                    if imagetem.shape[0] < 96 or imagetem.shape[1] < 96 or imagetem.shape[2] != 3:
                        continue
                    rand_nums1 = np.random.randint(0, imagetem.shape[0] - 96)
                    rand_nums2 = np.random.randint(0, imagetem.shape[1] - 96)
                    imagetem = imagetem[rand_nums1:rand_nums1 + 96, rand_nums2:rand_nums2 + 96, :]
                    if len(imagetem.shape) == 3:
                        files.append(imagetem)
    return files


def load_data(directory, ext, Hr):
    files = load_data_from_dirs(load_path(directory), ext, Hr)
    return files


x_test = load_data(vaild_path, ".png", True)
x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test)
x_test_lr = normalize(x_test_lr)

train_loader = DataLoader(datapath=vaild_path, batch_size=4, height_hr=384, width_hr=384, scale=4,
                          crops_per_image=1)

# model = load_model("./output/gen_model2.h5",
#                    custom_objects={'first_loss': first_loss, 'second_loss': second_loss, 'vgg_loss': vgg_loss,
#                                    'mse_loss': mse_loss})

# 加载ESRGAN模型
gen = Generator().generator()
gen.load_weights("./output1/Egen_model0t.h5")
gen1 = Generator().generator()
gen1.load_weights("./output1/Egen_model1t.h5")
gen2 = Generator().generator()
gen2.load_weights("./output1/Egen_model2t.h5")
gen3 = Generator().generator()
gen3.load_weights("./output1/Egen_model3t.h5")
gen4 = Generator().generator()
gen4.load_weights("./output1/Egen_model4t.h5")
plot_test_image(gen, train_loader, x_test_hr, x_test_lr, 'output1', 3,name="ESRGANGen0")
# plot_test_image_with_refer(gen, train_loader, x_test_hr, x_test_lr, 'output1', 1, name="gen0", refer_model=gen1,
#                            refer_model_name="gen1")
# save_test_image_with_hr_bc_lr(gen,train_loader,x_test_hr,x_test_lr,'output2',epoch=900,name="ESRGAN")
save_test_image(gen,train_loader,x_test_hr,x_test_lr,'ESRGAN-900',epoch=900,name="ESRGAN")
save_test_image(gen1,train_loader,x_test_hr,x_test_lr,'ESRGAN-1900',epoch=1900,name="ESRGAN")
save_test_image(gen2,train_loader,x_test_hr,x_test_lr,'ESRGAN-2900',epoch=2900,name="ESRGAN")
save_test_image(gen3,train_loader,x_test_hr,x_test_lr,'ESRGAN-3900',epoch=3900,name="ESRGAN")
save_test_image(gen4,train_loader,x_test_hr,x_test_lr,'ESRGAN-4000',epoch=4000,name="ESRGAN")


