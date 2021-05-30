from Emodel import Generator, Discriminator

import matplotlib.pyplot as plt
import gc

plt.switch_backend('agg')
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras import losses
import keras.backend as K
import numpy as np
import os
import imageio
import tensorflow as tf
from vgg19_noAct import VGG19
from keras.layers import Lambda
from util import DataLoader, plot_test_image
from numpy import array
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 代表是用cpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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


def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def first_loss(y_true, y_pred):
    binary_cross = losses.binary_crossentropy(y_true, y_pred)
    return mse_loss(y_true, y_pred) + 1e-3 * binary_cross


def build_vgg(input):
    # Input image to extract features from
    img = Input(shape=input)

    # Get the vgg network. Extract features from last conv layer
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[20].output]

    # Create model and compile
    model = Model(inputs=img, outputs=vgg(img))
    model.trainable = False
    return model


def preprocess_vgg(x):
    """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
    if isinstance(x, np.ndarray):
        return preprocess_input((x + 1) * 127.5)
    else:
        return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)


def RaGAN(shape_hr, discriminator):
    def comput_loss(x):
        real, fake = x
        fake_logit = K.sigmoid(fake - K.mean(real))
        real_logit = K.sigmoid(real - K.mean(fake))
        return [fake_logit, real_logit]

    imgs_hr = Input(shape_hr)
    generated_hr = Input(shape_hr)
    real_discriminator_logits = discriminator(imgs_hr)
    fake_discriminator_logits = discriminator(generated_hr)
    total_loss = Lambda(comput_loss, name='comput_loss')([real_discriminator_logits, fake_discriminator_logits])
    fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
    real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])
    dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                      K.binary_crossentropy(K.ones_like(real_logit), real_logit))
    model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit])

    model.add_loss(dis_loss)
    model.compile(optimizer=Adam(lr=learn_rate, epsilon=1e-08))

    model.metrics_names.append('dis_loss')
    model.metrics_tensors.append(dis_loss)
    return model


def Srgan(generator, discriminator, RaGAN, vgg, shape_lr, shape_hr, learn_rate,
          ):
    loss_weights = {'percept': 1e-3, 'gen': 5e-3, 'pixel': 1e-2}
    """Create the combined SRGAN network"""

    def comput_loss(x):
        img_hr, generated_hr = x
        # Compute the Perceptual loss
        gen_feature = vgg(preprocess_vgg(generated_hr))
        ori_feature = vgg(preprocess_vgg(img_hr))
        percept_loss = tf.losses.mean_squared_error(gen_feature, ori_feature)
        # Compute the RaGAN loss
        fake_logit, real_logit = RaGAN([img_hr, generated_hr])
        gen_loss = K.mean(
            K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
            K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))
        # Compute the pixel_loss with L1 loss
        # pixel_loss = tf.losses.absolute_difference(generated_hr, img_hr)
        return [percept_loss, gen_loss]

    # Input LR images
    img_lr = Input(shape_lr)
    img_hr = Input(shape_hr)
    # Create a high resolution image from the low resolution one
    generated_hr = generator(img_lr)

    # In the combined model we only train the generator
    discriminator.trainable = False
    RaGAN.trainable = False

    # Output tensors to a Model must be the output of a Keras `Layer`
    total_loss = Lambda(comput_loss, name='comput_loss')([img_hr, generated_hr])
    percept_loss = Lambda(lambda x: loss_weights['percept'] * x, name='percept_loss')(total_loss[0])
    gen_loss = Lambda(lambda x: loss_weights['gen'] * x, name='gen_loss')(total_loss[1])
    # pixel_loss = Lambda(lambda x: self.loss_weights['pixel'] * x, name='pixel_loss')(total_loss[2])
    # loss = Lambda(lambda x: x[0]+x[1]+x[2], name='total_loss')(total_loss)

    # Create model
    model = Model(inputs=[img_lr, img_hr], outputs=[percept_loss, gen_loss])

    # Add the loss of model and compile
    # model.add_loss(loss)
    model.add_loss(percept_loss)
    model.add_loss(gen_loss)
    # model.add_loss(pixel_loss)
    model.compile(optimizer=Adam(learn_rate, epsilon=1e-08))

    # Create metrics of ESRGAN
    model.metrics_names.append('percept_loss')
    model.metrics_tensors.append(percept_loss)
    model.metrics_names.append('gen_loss')
    model.metrics_tensors.append(gen_loss)
    # model.metrics_names.append('pixel_loss')
    # model.metrics_tensors.append(pixel_loss)
    return model


print("start ESRGAN")
x_test = load_data("/root/userfolder/data/DIV2K_valid_HR1/DIV2K_valid_HR/", ".png")
x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)
print(x_test_hr.shape)

x_test_lr = lr_images(x_test, 4)
x_test_lr = normalize(x_test_lr)

train_hr_path = '/root/userfolder/data/DIV2K_train_HR1/DIV2K_train_HR/'
# train_hr_path = 'D:\\pyProjects\\data\\MINI\\DIV2K_train_HR\\DIV2K_train_HR'
learn_rate = 1e-4
bath_size = 2
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
    plt.savefig('./output/Egan_generated_image_epoch_%d.png' % epoch)
    plt.close()
    gc.collect()


def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    The equation is:
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

    Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def train(gen_init_epoch=50, epoch=2000):
    Hr = (384, 384, 3)
    Lr = (96, 96, 3)
    train_loader = DataLoader(train_hr_path, batch_size=bath_size, height_hr=Hr[0], width_hr=Hr[1], scale=4,
                              crops_per_image=1)
    generator = Generator().generator()
    discriminator = Discriminator().discriminator()
    generator.compile(loss=first_loss, optimizer=Adam(learn_rate, epsilon=1e-08))
    # print(len(train_loader))
    # for step in range(1, gen_init_epoch + 1):
    #     for i in range(len(train_loader)):
    #         # for i in range(1):
    #         train_lr, train_hr = train_loader.load_batch(idx=i, bicubic=True)
    #         gloss = generator.train_on_batch(train_lr, train_hr)
    #         print("step is " + str(step) + " i is " + str(i) + " gloss is " + str(gloss) + "\n")
    # print("save Egen_model_init")
    # generator.save_weights('./output1/e11gen_model_init.h5')

    print("load RaGan")
    raGan = RaGAN(Hr, discriminator)
    print("vgg")
    vgg = build_vgg(Hr)
    print("srgan")
    print(len(train_loader))
    srgan = Srgan(generator, discriminator, raGan, vgg, Lr, Hr, learn_rate)
    for step in range(1, epoch + 1):
        for i in range(len(train_loader)):
            train_lr, train_hr = train_loader.load_batch(idx=i, bicubic=True)
            gen_hr = generator.predict(train_lr)
            ra_loss = raGan.train_on_batch([train_hr, gen_hr], None)
            gen_loss = srgan.train_on_batch([train_lr, train_hr], None)
            if i % 10 == 0:
                print("step:%d i:%d ra_loss:%s gen_loss:%s" % (step, i, str(ra_loss), str(gen_loss)))
        if step % 10 == 0:
            print("lr is " + str(K.get_value(raGan.optimizer.lr)))
        if step % 100 == 0 or step == epoch or step == 1:
            print("lr is " + str(K.get_value(raGan.optimizer.lr)))
            plot_test_image(generator, train_loader, x_test_hr, x_test_lr, 'output1', step)
            generator.save_weights('./output1/Egen_model%dt.h5' % (step // 1000))
            discriminator.save_weights('./output1/Ed_model%d.h5' % (step // 1000))
        lr = learn_rate / (step ** (1 / 3))
        K.set_value(raGan.optimizer.lr, lr)
        K.set_value(srgan.optimizer.lr, lr)


train(1, 4000)
