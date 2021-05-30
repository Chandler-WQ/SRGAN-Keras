import os
import gc
import numpy as np
from PIL import Image
from random import choice

from niqe import calculate_niqe

# from cat_lpips import cat_lpips

try:
    from skimage.measure import compare_psnr, compare_ssim
except ModuleNotFoundError:
    print(">> You'd better install scikit-image to support PSNR & SSIM comput")
from keras.models import load_model
from keras.utils import Sequence
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except:
    pass


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DataLoader(Sequence):
    def __init__(self, datapath, batch_size, height_hr, width_hr, scale, crops_per_image):
        """        
        :param string datapath: filepath to training images
        :param int height_hr: Height of high-resolution images
        :param int width_hr: Width of high-resolution images
        :param int height_hr: Height of low-resolution images
        :param int width_hr: Width of low-resolution images
        :param int scale: Upscaling factor
        """

        # Store the datapath
        self.datapath = datapath
        self.batch_size = batch_size
        self.height_hr = height_hr
        self.height_lr = int(height_hr / scale)
        self.width_hr = width_hr
        self.width_lr = int(width_hr / scale)
        self.scale = scale
        self.crops_per_image = crops_per_image
        self.total_imgs = None

        # Options for resizing
        self.options = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

        # Check data source
        self.img_paths = []
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        self.total_imgs = len(self.img_paths)
        print(self.total_imgs)

    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    # @staticmethod
    # def scale_lr_imgs(imgs):
    #     """Scale low-res images prior to passing to SRGAN"""
    #     return imgs / 255.
    #
    # @staticmethod
    # def unscale_lr_imgs(imgs):
    #     """Un-Scale low-res images"""
    #     return imgs * 255

    @staticmethod
    def scale_imgs(imgs):
        """Scale high-res images prior to passing to SRGAN"""
        return imgs / 127.5 - 1

    @staticmethod
    def unscale_imgs(imgs):
        """Un-Scale high-res images"""
        return (imgs + 1.) * 127.5

    @staticmethod
    def load_img(path, training=True):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if training:
            flag = np.random.randint(0, 8)
            if flag > 3:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # img = img.rotate(flag * 90)
        return np.array(img)

    def __len__(self):
        return int(self.total_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        return self.load_batch(idx=idx)

    def load_batch(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of images from datapath folder"""

        # Starting index to look in
        cur_idx = 0
        if not img_paths:
            cur_idx = idx * self.batch_size

            # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:

            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break

            try:
                # Load image
                img_hr = None
                if img_paths:
                    img_hr = self.load_img(img_paths[cur_idx], training)
                else:
                    img_hr = self.load_img(self.img_paths[cur_idx], training)

                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        # print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))
                else:
                    # img_crops = [img_hr]
                    img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))

                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break

                        # For LR, do bicubic downsampling
                    method = Image.BICUBIC if bicubic else choice(self.options)
                    lr_shape = (int(img_hr.shape[1] / self.scale), int(img_hr.shape[0] / self.scale))
                    img_lr = Image.fromarray(img_hr.astype(np.uint8))
                    img_lr = np.array(img_lr.resize(lr_shape, method))
                    # Scale color values
                    img_hr = self.scale_imgs(img_hr)
                    img_lr = self.scale_imgs(img_lr)

                    # Store images
                    imgs_hr.append(img_hr)
                    imgs_lr.append(img_lr)

            except Exception as e:
                # print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr


# 这里是与其他模型进行测试集的比较，并且提供了ssim和psnr维度的比较
def plot_test_images(model, loader, datapath_test, test_output, epoch, name='ESRGAN', refer_model=None,
                     refer_model_name='SRGAN'):
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    """
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    # try:
    # Get the location of test images
    test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if
                   any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]
    print(test_images)
    # Load the images to perform test on images
    # 加载全部的数据
    imgs_lr, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=True)

    for i in range(len(test_images)):
        print(imgs_lr[i].shape)
        print(imgs_hr[i].shape)
    # Create super resolution and bicubic interpolation images
    imgs_res = []
    imgs_sr = []
    imgs_bc = []
    for i in range(len(test_images)):

        # Bicubic interpolation
        pil_img = loader.unscale_imgs(imgs_lr[i]).astype('uint8')
        pil_img = Image.fromarray(pil_img)
        hr_shape = (imgs_hr[i].shape[1], imgs_hr[i].shape[0])

        # 插值之后的图像
        imgs_bc.append(
            loader.scale_imgs(
                np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
            )
        )
        # refer_model prediction
        # 参考模型预测
        if refer_model is not None:
            imgs_res.append(
                np.squeeze(
                    refer_model.predict(
                        np.expand_dims(imgs_lr[i], 0),
                        batch_size=1
                    ),
                    axis=0
                )
            )
        # SRGAN prediction
        imgs_sr.append(
            np.squeeze(
                model.predict(
                    np.expand_dims(imgs_lr[i], 0),
                    batch_size=1
                ),
                axis=0
            )
        )

    # Unscale colors values
    imgs_lr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_lr]
    imgs_bc = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_bc]
    if refer_model is not None:
        imgs_res = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_res]
    imgs_hr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_hr]
    imgs_sr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_sr]

    print("start cal")
    if refer_model is None:
        # Loop through images
        for img_hr, img_lr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_bc, imgs_sr, test_images):
            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
            psnr = []
            ssim = []
            niqe = []
            psnr.append(-1)
            ssim.append(-1)

            niqe.append(calculate_niqe(img_lr, 0))
            niqe.append(calculate_niqe(img_bc, 0))
            niqe.append(calculate_niqe(img_sr, 0))
            niqe.append(calculate_niqe(img_hr, 0))
            psnr.append(compare_psnr(img_hr, img_bc))
            psnr.append(compare_psnr(img_hr, img_sr))
            ssim.append(compare_ssim(img_hr, img_bc, multichannel=True))
            ssim.append(compare_ssim(img_hr, img_sr, multichannel=True))
            psnr.append(-1)
            ssim.append(-1)
            # Images and titles
            images = {
                'Low Resolution': img_lr,
                'Bicubic Interpolation': img_bc,
                # 'SRResNet': img_res,
                name: img_sr,
                'Original': img_hr
            }
            plt.imsave(os.path.join(test_output, "{}_out.png".format(filename)), img_sr)
            # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title(
                    "{} - {} - psnr:{:.4f} - ssim{:.4f} -- niqe{}".format(title, img.shape, psnr[i], ssim[i],
                                                                          str(niqe[i][0][0])[:6]))
                # axes[i].set_title(
                #     "{} - {} - psnr:{:.4f} - ssim{:.4f} ".format(title, img.shape, psnr[i], ssim[i]
                #                                                  ))
                axes[i].axis('off')
            plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

            # Save directory
            savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
            fig.savefig(savefile)
            plt.close()
            gc.collect()

    else:
        # Loop through images
        for img_hr, img_bc, img_res, img_sr, img_path in zip(imgs_hr, imgs_bc, imgs_res, imgs_sr, test_images):
            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
            psnr = []
            ssim = []
            niqe = []

            niqe.append(calculate_niqe(img_bc, 0))
            niqe.append(calculate_niqe(img_res, 0))
            niqe.append(calculate_niqe(img_sr, 0))
            niqe.append(calculate_niqe(img_hr, 0))

            psnr.append(compare_psnr(img_hr, img_bc))
            psnr.append(compare_psnr(img_hr, img_res))
            psnr.append(compare_psnr(img_hr, img_sr))
            ssim.append(compare_ssim(img_hr, img_bc, multichannel=True))
            ssim.append(compare_ssim(img_hr, img_res, multichannel=True))
            ssim.append(compare_ssim(img_hr, img_sr, multichannel=True))
            psnr.append(-1)
            ssim.append(-1)
            # Images and titles
            images = {
                'Bicubic Interpolation': img_bc,
                refer_model_name: img_res,
                name: img_sr,
                'Original': img_hr
            }
            plt.imsave(os.path.join(test_output, "{}_out.png".format(filename)), img_sr)
            # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title("{} - {} - psnr:{:.4f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i]))
                axes[i].set_title(
                    "{} - {} - psnr:{:.4f} - ssim{:.4f} -- niqe{}".format(title, img.shape, psnr[i], ssim[i],
                                                                          str(niqe[i][0][0])[:6]))
                # axes[i].set_title("{} - {} - psnr:{:.4f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i])
                axes[i].axis('off')
                plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
                print('PSNR:', psnr)
                print('SSIM:', ssim)
                # Save directory
                savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
                fig.savefig(savefile)
                plt.close()
                gc.collect()
                # except Exception as e:
                #     print(">> Could not perform printing. Maybe matplotlib is not installed.")


def plot_test_image(model, loader, test_hrs, test_lrs, test_output, epoch, name='ESRGAN'):
    rand_nums = np.random.randint(0, test_hrs.shape[0], size=1)
    test_hr = test_hrs[rand_nums][0]
    test_lr = test_lrs[rand_nums][0]
    pil_img = loader.unscale_imgs(test_lr).astype('uint8')
    pil_img = Image.fromarray(pil_img)
    # print(pil_img.shape)
    hr_shape = (test_hr.shape[1], test_hr.shape[0])

    # 插值之后的图像
    img_bc = loader.scale_imgs(
        np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
    )

    img_sr = np.squeeze(model.predict(
        np.expand_dims(test_lr, 0),
        batch_size=1
    ), axis=0)

    img_lr = loader.unscale_imgs(test_lr).astype(np.uint8)
    img_bc = loader.unscale_imgs(img_bc).astype(np.uint8)
    img_hr = loader.unscale_imgs(test_hr).astype(np.uint8)
    img_sr = loader.unscale_imgs(img_sr).astype(np.uint8)
    psnr = []
    ssim = []
    niqe = []
    lpips = []
    niqe.append(calculate_niqe(img_lr, 0))
    niqe.append(calculate_niqe(img_bc, 0))
    niqe.append(calculate_niqe(img_sr, 0))
    niqe.append(calculate_niqe(img_hr, 0))
    # lpips.append(0.0)
    # lpips.append(cat_lpips(img_bc, img_hr))
    # lpips.append(cat_lpips(img_sr, img_hr))
    # lpips.append(0.0)

    psnr.append(-1)
    ssim.append(-1)
    psnr.append(compare_psnr(img_hr, img_bc))
    psnr.append(compare_psnr(img_hr, img_sr))
    ssim.append(compare_ssim(img_hr, img_bc, multichannel=True))
    ssim.append(compare_ssim(img_hr, img_sr, multichannel=True))
    psnr.append(-1)
    ssim.append(-1)

    images = {
        'LR': img_lr,
        'BC': img_bc,
        name: img_sr,
        'HR': img_hr
    }
    # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        # axes[i].set_title(
        #     "{} - {} - psnr:{:.4f} - ssim{:.4f} -- niqe{} --lpips{}".format(title, img.shape, psnr[i], ssim[i],
        #                                                                     str(niqe[i][0][0])[:6]), str(lpips[i])[:6])
        axes[i].set_title(
            "{} - {} - psnr:{:.4f} - ssim{:.4f} -- niqe{}".format(title, img.shape, psnr[i], ssim[i],
                                                                  str(niqe[i][0][0])[:6]))
        axes[i].axis('off')
    plt.suptitle(name + '- Epoch: {}'.format(epoch))

    # Save directory
    savefile = os.path.join(test_output, name + "-Epoch{}.png".format(epoch))
    fig.savefig(savefile)
    plt.close()
    gc.collect()


def plot_test_image_with_refer(model, loader, test_hrs, test_lrs, test_output, epoch, name='ESRGAN', refer_model=None,
                               refer_model_name='None'):
    rand_nums = np.random.randint(0, test_hrs.shape[0], size=1)
    test_hr = test_hrs[rand_nums][0]
    test_lr = test_lrs[rand_nums][0]
    pil_img = loader.unscale_imgs(test_lr).astype('uint8')
    pil_img = Image.fromarray(pil_img)
    # print(pil_img.shape)
    hr_shape = (test_hr.shape[1], test_hr.shape[0])

    # 插值之后的图像
    # img_bc = loader.scale_imgs(
    #     np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
    # )
    img_refer = np.squeeze(refer_model.predict(
        np.expand_dims(test_lr, 0),
        batch_size=1
    ), axis=0)

    img_sr = np.squeeze(model.predict(
        np.expand_dims(test_lr, 0),
        batch_size=1
    ), axis=0)

    img_lr = loader.unscale_imgs(test_lr).astype(np.uint8)
    img_refer = loader.unscale_imgs(img_refer).astype(np.uint8)
    img_hr = loader.unscale_imgs(test_hr).astype(np.uint8)
    img_sr = loader.unscale_imgs(img_sr).astype(np.uint8)
    psnr = []
    ssim = []
    niqe = []
    niqe.append(calculate_niqe(img_lr, 0))
    niqe.append(calculate_niqe(img_refer, 0))
    niqe.append(calculate_niqe(img_sr, 0))
    niqe.append(calculate_niqe(img_hr, 0))

    psnr.append(-1)
    ssim.append(-1)
    psnr.append(compare_psnr(img_hr, img_refer))
    psnr.append(compare_psnr(img_hr, img_sr))
    ssim.append(compare_ssim(img_hr, img_refer, multichannel=True))
    ssim.append(compare_ssim(img_hr, img_sr, multichannel=True))
    psnr.append(-1)
    ssim.append(-1)

    images = {
        'Low Resolution': img_lr,
        refer_model_name: img_refer,
        name: img_sr,
        'Original': img_hr
    }
    # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title(
            "{} - {} - psnr:{:.4f} - ssim{:.4f} -- niqe{}".format(title, img.shape, psnr[i], ssim[i],
                                                                  str(niqe[i][0][0])[:6]))
        axes[i].axis('off')
    plt.suptitle(name + ' - Epoch: {}'.format(epoch))

    # Save directory
    savefile = os.path.join(test_output, name + "-Epoch{}.png".format(epoch))
    fig.savefig(savefile)
    plt.close()
    gc.collect()


def save_test_image(model, loader, test_hrs, test_lrs, test_output, epoch, name='ESRGAN'):
    for i in range(len(test_lrs)):
        test_hr = test_hrs[i]
        test_lr = test_lrs[i]
        pil_img = loader.unscale_imgs(test_lr).astype('uint8')
        pil_img = Image.fromarray(pil_img)
        # print(pil_img.shape)
        hr_shape = (test_hr.shape[1], test_hr.shape[0])

        # 插值之后的图像
        img_bc = loader.scale_imgs(
            np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
        )

        img_sr = np.squeeze(model.predict(
            np.expand_dims(test_lr, 0),
            batch_size=1
        ), axis=0)
        img_sr = loader.unscale_imgs(img_sr).astype('uint8')
        im = Image.fromarray(img_sr)
        im.save("./" + test_output + "/" + name + str(epoch) + "-"+ str(i) + ".png")
        print(str(i)+"is over")

def save_test_image_with_hr_bc_lr(model, loader, test_hrs, test_lrs, test_output, epoch, name='ESRGAN'):
    for i in range(len(test_lrs)):
        test_hr = test_hrs[i]
        test_lr = test_lrs[i]
        pil_img = loader.unscale_imgs(test_lr).astype('uint8')
        pil_img = Image.fromarray(pil_img)
        # print(pil_img.shape)
        hr_shape = (test_hr.shape[1], test_hr.shape[0])

        # 插值之后的图像
        img_bc = loader.scale_imgs(
            np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
        )

        img_sr = np.squeeze(model.predict(
            np.expand_dims(test_lr, 0),
            batch_size=1
        ), axis=0)
        img_sr = loader.unscale_imgs(img_sr).astype('uint8')
        im = Image.fromarray(img_sr)
        im.save("./" + test_output + "/" + name + str(epoch) + "-" + str(i) + ".png")

        img_hr = loader.unscale_imgs(test_hr).astype('uint8')
        im_hr = Image.fromarray(img_hr)
        im_hr.save("./" + test_output + "/" + "hr" + str(i) + ".png")

        img_lr = loader.unscale_imgs(test_lr).astype('uint8')
        im_lr = Image.fromarray(img_lr)
        im_lr.save("./" + test_output + "/" + "lr" + str(i) + ".png")

        img_bc = loader.unscale_imgs(img_bc).astype('uint8')
        im_bc = Image.fromarray(img_bc)
        im_bc.save("./" + test_output + "/" + "bc" + str(i) + ".png")
        print(str(i)+"is over")


def plot_bigger_images(model, loader, datapath_test, test_output, epoch, name='ESRGAN', refer_model=None):
    """
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    # Get the location of test images
    test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if
                   any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]

    # Load the images to perform test on images
    _, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=False)
    # Create super resolution and bicubic interpolation images
    imgs_res = []
    imgs_sr = []
    imgs_bc = []
    for i in range(len(test_images)):

        # Bicubic interpolation
        pil_img = loader.unscale_imgs(imgs_hr[i]).astype('uint8')
        pil_img = Image.fromarray(pil_img)
        hr_shape = (4 * imgs_hr[i].shape[1], 4 * imgs_hr[i].shape[0])
        tmp_hr = loader.scale_imgs(np.array(pil_img))
        imgs_bc.append(
            loader.scale_imgs(
                np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
            )
        )
        # refer_model prediction
        if refer_model is not None:
            imgs_res.append(
                np.squeeze(
                    refer_model.predict(
                        np.expand_dims(tmp_hr, 0),
                        batch_size=1
                    ),
                    axis=0
                )
            )
        # SRGAN prediction
        imgs_sr.append(
            np.squeeze(
                model.generator.predict(
                    np.expand_dims(tmp_hr, 0),
                    batch_size=1
                ),
                axis=0
            )
        )

    # Unscale colors values
    imgs_bc = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_bc]
    imgs_hr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_hr]
    if refer_model is not None:
        imgs_res = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_res]
    imgs_sr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_sr]

    if refer_model is None:
        # Loop through images
        for img_hr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_bc, imgs_sr, test_images):
            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]

            # Images and titles
            images = {
                'Original': img_hr,
                'Bicubic Interpolation': img_bc,
                # 'SRResNet': img_res,
                name: img_sr,
            }
            plt.imsave(os.path.join(test_output, "{}_{}.png".format(filename, name)), img_sr)
            # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title("{} - {}".format(title, img.shape))
                axes[i].axis('off')
            plt.suptitle('{}'.format(filename))

            # Save directory
            savefile = os.path.join(test_output, "{}.png".format(filename))
            fig.savefig(savefile)
            plt.close()
            gc.collect()

    else:
        # Loop through images
        for img_hr, img_bc, img_res, img_sr, img_path in zip(imgs_hr, imgs_bc, imgs_res, imgs_sr, test_images):
            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]

            # Images and titles
            images = {
                'Original': img_hr,
                'Bicubic Interpolation': img_bc,
                'SR-RRDB': img_res,
                name: img_sr,
            }

            plt.imsave(os.path.join(test_output, "{}_{}.png".format(filename, name)), img_sr)
            # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title("{} - {}".format(title, img.shape))
                axes[i].axis('off')
            plt.suptitle('{}'.format(filename))

            # Save directory
            savefile = os.path.join(test_output, "{}.png".format(filename))
            fig.savefig(savefile)
            plt.close()
            gc.collect()


# 只保存一个大图
def plot_test_only(model, datapath_test, test_output):
    """
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    # Get the location of test images
    test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if
                   any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]
    for test in test_images:
        print(test + "\t" + test[26:-5])
    test_images.sort(key=lambda x: int(x[26:-5]))

    # Load the images to perform test on images
    imgs_lr = []
    pics_num = len(test_images)
    for path in test_images:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        imgs_lr.append(np.array(img))

    # Create super resolution and bicubic interpolation images
    print("Predicting the SR Image......")
    for i in tqdm(range(pics_num)):
        tmp_hr = DataLoader.scale_imgs(imgs_lr[i])

        # SRGAN prediction
        img_sr = np.squeeze(
            model.generator.predict(
                np.expand_dims(tmp_hr, 0),
                batch_size=1
            ),
            axis=0
        )
        img_sr = (img_sr + 1.) * 127.5
        img_sr = img_sr.astype(np.uint8)
        plt.imsave(os.path.join(test_output, "test_original (%d).png" % (i + 1)), img_sr)
        plt.close()


def compute_metric(model, loader, datapath_test, test_output, epoch):
    """
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    try:
        # SRResNet = load_model('./data/weights/DIV2K_generator.h5')
        # Get the location of test images
        test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if
                       any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]

        # Load the images to perform test on images
        imgs_lr, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=True)

        # Create super resolution and bicubic interpolation images
        imgs_sr = []
        for i in range(len(test_images)):
            # SRGAN prediction
            imgs_sr.append(
                np.squeeze(
                    model.generator.predict(
                        np.expand_dims(imgs_lr[i], 0),
                        batch_size=1
                    ),
                    axis=0
                )
            )

        # Unscale colors values
        imgs_hr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_hr]
        imgs_sr = [loader.unscale_imgs(img).astype(np.uint8) for img in imgs_sr]
        psnr = []
        ssim = []
        # Loop through images
        for img_hr, img_sr, img_path in zip(imgs_hr, imgs_sr, test_images):
            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
            plt.imsave(os.path.join(test_output, "{}_epoch{:05d}.png".format(filename, epoch)), img_sr)
            # psnr.append("{:.4f}".format(compare_psnr(img_hr, img_sr)))
            # ssim.append("{:.4f}".format(compare_ssim(img_hr, img_sr, multichannel=True)))

        return psnr, ssim

    except Exception as e:
        print(">> Could not perform printing. Maybe matplotlib is not installed.")
