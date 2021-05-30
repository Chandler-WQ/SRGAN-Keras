from model import Generator, Discriminator

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam
from keras import losses
import keras.backend as K
import imageio
import numpy as np
from numpy import array
import os
import sys
from PIL import Image
import tensorflow as tf

downscale_factor = 4
# np.random.seed(10)
image_shape = (384, 384, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代表是用cpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

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


x_train = load_data("/root/userfolder/data/DIV2K_train_HR/DIV2K_train_HR/", ".png")
x_test = load_data("/root/userfolder/data/DIV2K_valid_HR/DIV2K_valid_HR/", ".png")
print("data loaded" + str(len(x_train)) + "\n")


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


x_train_hr = hr_images(x_train)
x_train_hr = normalize(x_train_hr)
print(x_train_hr.shape)

x_train_lr = lr_images(x_train, downscale_factor)
x_train_lr = normalize(x_train_lr)

x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)
print(x_test_hr.shape)

x_test_lr = lr_images(x_test, downscale_factor)
x_test_lr = normalize(x_test_lr)

print("data processed")




def plot_generated_images(epoch, generator, examples=3, dim=(1, 3)):
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = denormalize(x_test_hr[rand_nums]) / 255.
    image_batch_lr = x_test_lr[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img) / 255.
    image_batch_lr = denormalize(image_batch_lr) / 255.

    # generated_image = deprocess_HR(generator.predict(image_batch_lr))

    plt.figure(figsize=(15, 5))
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('./output/gan_generated_image_epoch_%d.png' % epoch)
    plt.close()


def train(epoch_init=10, epochs=100, batch_size=4):
    batch_count = int(x_train_hr.shape[0] / batch_size / 2)
    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])

    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    generator.compile(loss=first_loss,
                      optimizer=adam)

    # 先让生成器经过一定的训练，损失函数为mse
    # for e in range(1, epoch_init + 1):
    #     print('-' * 15, 'Epoch %d' % e, '-' * 15)
    #     for i in range(batch_count):
    #         print('-' * 15, 'Epoch %d' % e, 'step %d' % i, '-' * 15)
    #         rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
    #         image_batch_hr = x_train_hr[rand_nums]
    #         image_batch_lr = x_train_lr[rand_nums]
    #         gloss = generator.train_on_batch(image_batch_lr, image_batch_hr)
    #     print("gloss is " + str(gloss) + "\n")
    for e in range(1, epoch_init + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        image_batch_hr = x_train_hr[rand_nums]
        image_batch_lr = x_train_lr[rand_nums]
        gloss = generator.train_on_batch(image_batch_lr, image_batch_hr)
        print("gloss is " + str(gloss) + "\n")

    generator.save('./output/gen_model_init.h5')

    generator.compile(loss=second_loss,
                      optimizer=adam)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for i in range(batch_count):
            print('-' * 15, 'Epoch %d' % e, 'step %d' % i, '-' * 15)
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size)
            fake_data_Y = np.zeros(batch_size)

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)

            gloss = generator.train_on_batch(image_batch_lr, image_batch_hr)

        print("Loss real_data , fake_data, Loss G")
        print(d_loss_real, d_loss_fake, gloss)
        print("\n")
        print("lr is " + str(K.get_value(adam.lr)) + "\n")
        if e == 1 or e % 10 == 0:
            plot_generated_images(e, generator)
        if e % 100 == 0:
            generator.save('./output/gen_model%d.h5' % (e // 1000))
            discriminator.save('./output/dis_model%d.h5' % (e // 1000))
        if e == epochs / 2:
            K.set_value(adam.lr, 1E-5)



train(100, 20000, 4)
