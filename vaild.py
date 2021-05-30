import matplotlib.pyplot as plt

plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import losses
import keras.backend as K
import imageio
import numpy as np
from numpy import array
import os
from niqe import calculate_niqe
from util import DataLoader, plot_test_images, plot_test_only, plot_bigger_images, compute_metric, plot_test_image
from keras.models import load_model
from PIL import Image

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
image_shape = (384, 384, 3)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

def plot_generated_images(lr, genr, dim=(1, 2)):
    plt.figure(figsize=(10, 5))
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(lr, interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(genr, interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('./demo/hanhangan1.png')
    plt.close()


vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
loss_model.trainable = False


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def first_loss(y_true, y_pred):
    binary_cross = losses.binary_crossentropy(y_true, y_pred)
    return mse_loss(y_true, y_pred) + 1e-3 * binary_cross


def second_loss(y_true, y_pred):
    binary_cross = losses.binary_crossentropy(y_true, y_pred)
    return vgg_loss(y_true, y_pred) + 1e-3 * binary_cross


def lr_images(images_real):
    images = []
    for img in range(len(images_real)):
        images.append(np.array(Image.fromarray(images_real[img]).resize((96,
                                                                         96),
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


x_test = load_data("D:\\pyProjects\\data\\MINI\\DIV2K_valid_HR\\DIV2K_valid_HR", ".png", True)
x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)
print(x_test_hr.shape)

x_test_lr = lr_images(x_test)
x_test_lr = normalize(x_test_lr)
#
# file = load_data("./demo", "png")
# file = normalize(array(file))
#
# print(file[0].shape)
#
# file = denormalize(file)

# for i in range(len(file)):
# res = calculate_niqe(file[0], 0)
# print(res)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model = load_model("./output/gen_model2.h5",
                   custom_objects={'first_loss': first_loss, 'second_loss': second_loss, 'vgg_loss': vgg_loss,
                                   'mse_loss': mse_loss})
# model1 = load_model("./output/gen_model1.h5",
#                     custom_objects={'first_loss': first_loss, 'second_loss': second_loss, 'vgg_loss': vgg_loss,
#                                     'mse_loss': mse_loss})
# # file0 = np.reshape(file[0], (1,) + file[0].shape)
# # file_gan_0 = model.predict(file0)
# # plot_generated_images(denormalize(file0[0]) / 255., denormalize(file_gan_0[0]) / 255.)
train_hr_path = 'D:\\pyProjects\\data\\MINI\\DIV2K_train_HR\\DIV2K_train_HR'
train_loader = DataLoader(datapath=train_hr_path, batch_size=4, height_hr=384, width_hr=384, scale=4,
                          crops_per_image=1)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# plot_test_images(model, train_loader, "./demo", "./outputq", 3, refer_model=model1)
# plot_bigger_images(model, train_loader, "./demo", "./outputq", 1)
# plot_test_only(model, "./demo", "./outputq", 2)

plot_test_image(model, train_loader, x_test_hr, x_test_lr, 'output1', 3)


