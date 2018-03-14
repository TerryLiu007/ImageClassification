import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import listdir
import os
from random import shuffle
import glob
import h5py

category = {'BabyBibs' : 0,
            'BabyHat' : 1,
            'BabyPants' : 2,
            'BabyShirt' : 3,
            'PackageFart' : 4,
            'womanshirtsleeve' : 5,
            'womencasualshoes' : 6,
            'womenchiffontop' : 7,
            'womendollshoes' : 8,
            'womenknittedtop' : 9,
            'womenlazyshoes' : 10,
            'womenlongsleevetop' : 11,
            'womenpeashoes' : 12,
            'womenplussizedtop' : 13,
            'womenpointedflatshoes' : 14,
            'womensleevelesstop' : 15,
            'womenstripedtop' : 16,
            'wrapsnslings' : 17}

# returns bordered image with all the color channel - (64,64,3)
def processImage(path, outsize=64):

    img = cv.imread(path, cv.IMREAD_COLOR)
    img_size = img.shape
    assert img_size[2] == 3, "Image channel is not 3"

    # normalize the image to range 0 - 1
    # img_norm = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # concatenate image if needed
    # if img_size[0] > 2 * img_size[1]:
    #     post = np.concatenate((img, img), axis=1)
    # elif img_size[1] > 2 * img_size[0]:
    #     post = np.concatenate((img, img), axis=0)
    # else:
    #     post = img

    post = img
    post_size = post.shape

    # make border based on the initial size of the image
    if post_size[0] > post_size[1]:
        post = cv.copyMakeBorder(post, 0, 0, int((post_size[0]-post_size[1])/2), int((post_size[0]-post_size[1])/2), cv.BORDER_CONSTANT, value=(255,255,255))
    if post_size[0] < post_size[1]:
        post = cv.copyMakeBorder(post, int((post_size[1]-post_size[0])/2), int((post_size[1]-post_size[0])/2), 0, 0, cv.BORDER_CONSTANT, value=(255,255,255))

    # resize the image to desired size for cnn input
    post = cv.resize(post, (outsize,outsize), interpolation=cv.INTER_AREA)
    return post


# returns bordered image with only channel - (64,64)
def processMono(path, outsize=64):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # img_norm = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    post = img
    post_size = post.shape

    if post_size[0] > post_size[1]:
        post = cv.copyMakeBorder(post, 0, 0, int((post_size[0] - post_size[1]) / 2), int((post_size[0] - post_size[1]) / 2), cv.BORDER_CONSTANT, value=255)
    if post_size[0] < post_size[1]:
        post = cv.copyMakeBorder(post, int((post_size[1]-post_size[0])/2), int((post_size[1]-post_size[0])/2), 0, 0, cv.BORDER_CONSTANT, value=255)

    post = cv.resize(post, (outsize, outsize), interpolation=cv.INTER_AREA)
    return post


# returns max cropped image with gray channel - (64,64)
def processCrop(path, outsize=64):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    crop = img
    # crop = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    crop_size = crop.shape

    # crop the image
    if crop_size[0] > crop_size[1]:
        crop = crop[int(crop_size[0]/2-crop_size[1]/2):int(crop_size[0]/2+crop_size[1]/2)]
    elif crop_size[0] < crop_size[1]:
        crop = crop[:,int(crop_size[1]/2-crop_size[0]/2):int(crop_size[1]/2+crop_size[0]/2)]

    crop = cv.resize(crop, (outsize, outsize), interpolation=cv.INTER_AREA)

    return crop


# returns the texture most different from background with gray channel - (64,64)
def processTxt(path, outsize=64):

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_size = img.shape

    # generate mask for border histogram calculation
    mask = np.ones(img_size[:2],np.uint8)*255
    mask[1:img_size[0]-1,1:img_size[1]-1] = 0

    # visualize masked image
    # mask_img = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('masked image',mask_img)
    # cv.waitKey(0)

    hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])


    # decide sample location based on the image size, maximum 9 locations
    segment_v = 4 if int(img_size[0]/outsize) > 3 else int(img_size[0]/outsize) + 1
    segment_h = 4 if int(img_size[1]/outsize) > 3 else int(img_size[1]/outsize) + 1
    segment_v_len = int(img_size[0]/segment_v)
    segment_h_len = int(img_size[1]/segment_h)

    patch = []
    patch_edge = []
    count = 0

    for i in range(1,segment_v):
        for j in range(1,segment_h):
            patch.append(img[int(segment_v_len*i-outsize/2):int(segment_v_len*i+outsize/2), int(segment_h_len*j-outsize/2):int(segment_h_len*j+outsize/2)])
            patch_edge.append(cv.Canny(patch[count], 50, 200))
            count = count + 1

    hist_patch = []
    hist_edge = []
    hist_correlation = []

    # compare histogram of patch to histogram of border for selection
    for i in range(0,count):
        hist_patch.append(cv.calcHist([patch[i]], [0], None, [256], [0, 256]))
        hist_edge.append(cv.calcHist([patch_edge[i]], [0], None, [2], [0, 256]))
        score_border = cv.compareHist(hist_mask, hist_patch[i], cv.HISTCMP_CORREL)
        score_edge = hist_edge[i][0,0]/(outsize**2)
        # print('{0} border + edge = {1:.3f} + {2:.3f} = {3:.3f}'.format(i,score_border,score_edge,score_edge+score_border))
        hist_correlation.append(score_border+score_edge)

        # visualize all the patch image
        # cv.imshow('patches', patch[i])
        # cv.waitKey(0)

    # plt.subplot(231), plt.plot(hist_mask)
    # plt.subplot(232), plt.plot(hist_patch[0])
    # plt.subplot(233), plt.plot(hist_patch[1])
    # plt.subplot(234), plt.plot(hist_patch[2])
    # plt.subplot(235), plt.plot(hist_patch[3])
    # plt.subplot(236), plt.plot(hist_patch[4])
    # plt.xlim([0, 256])
    # plt.show()

    chosen_patch = patch[np.argmin(hist_correlation)]

    # patch_norm = chosen_patch/255

    return chosen_patch


# returns the texture most different from background with all channels - (64,64)
def processColoredTxt(path, outsize=64):

    img = cv.imread(path, cv.IMREAD_COLOR)
    img_size = img.shape

    # generate mask for border histogram calculation
    mask = np.ones(img_size[:2], np.uint8) * 255
    mask[2:img_size[0] - 2, 2:img_size[1] - 2] = 0

    # visualize masked image
    # mask_img = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('masked image',mask_img)
    # cv.waitKey(0)

    hist_mask = cv.calcHist([img], [0,1,2], mask, [64,64,64], [0,256,0,256,0,256])

    # decide sample location based on the image size, maximum 9 locations
    segment_v = 4 if int(img_size[0] / outsize) > 3 else int(img_size[0] / outsize) + 1
    segment_h = 4 if int(img_size[1] / outsize) > 3 else int(img_size[1] / outsize) + 1
    segment_v_len = int(img_size[0] / segment_v)
    segment_h_len = int(img_size[1] / segment_h)

    if segment_v == 1 or segment_h == 1:
        return processCrop(path)

    patch = []
    patch_edge = []
    count = 0

    for i in range(1, segment_v):
        for j in range(1, segment_h):
            patch.append(img[int(segment_v_len * i - outsize / 2):int(segment_v_len * i + outsize / 2),
                         int(segment_h_len * j - outsize / 2):int(segment_h_len * j + outsize / 2)])
            patch_edge.append(cv.Canny(patch[count], 50, 200))
            count = count + 1
    # print(count)

    hist_patch = []
    hist_edge = []
    hist_correlation = []

    # compare histogram of patch to histogram of border for selection
    for i in range(0, count):
        hist_patch.append(cv.calcHist([patch[i]], [0,1,2], None, [64,64,64], [0,256,0,256,0,256]))
        hist_edge.append(cv.calcHist([patch_edge[i]], [0], None, [2], [0, 256]))

        score_border = cv.compareHist(hist_mask, hist_patch[i], cv.HISTCMP_CORREL)
        score_edge = hist_edge[i][0, 0] / (outsize ** 2)
        # print('{0} border + edge = {1:.3f} + {2:.3f} = {3:.3f}'.format(i, score_border, score_edge, score_edge + score_border))
        hist_correlation.append(score_border + score_edge)

        # visualize all the patch image
        # cv.imshow('patches', patch[i])
        # cv.waitKey(0)

    chosen_patch = cv.cvtColor(patch[np.argmin(hist_correlation)], cv.COLOR_BGR2GRAY)
    # patch_norm = chosen_patch/255
    return chosen_patch


# test unit for image pre-processing
def test(outsize=64):

    for path in sorted(glob.glob('Test/Test_*.jpg'), key=lambda f: int(''.join(filter(str.isdigit, f)))):
        monoImg = processMono(path, outsize)
        croppedImg = processCrop(path, outsize)
        Txt = processTxt(path, outsize)
        coloredTxt = processColoredTxt(path, outsize)

        cv.imshow('compare', np.hstack((monoImg, croppedImg, Txt, coloredTxt)))
        cv.waitKey(0)

    # imagePath = 'Test/Test_28.jpg'
    # generate(imagePath, 64)
    # cv.waitKey(0)


# returns the augmented image - (64,64,3)
def processAugmented(path, outsize=64):

    mono = processMono(path).reshape(outsize,outsize,-1)
    crop = processCrop(path).reshape(outsize,outsize,-1)
    text = processColoredTxt(path).reshape(outsize,outsize,-1)

    # cv.imshow('augmented',np.hstack((mono,crop,text)))
    # cv.waitKey(0)

    augmented = np.concatenate((mono,crop,text),axis=2)
    return augmented


# main dataset generator class
class DataSetGenerator:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    # get folder names as labels under data_dir
    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    # get all the image paths under data_dir
    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            data_paths.append(img_lists)
        return data_paths

    # generate the processed image
    def generate_image(self, out_dir):
        counter = 1

        for i in range(len(self.data_labels)):
            label = self.data_labels[i]

            print('{} just been processed'.format(counter-1))
            print('processing {}, {} needs to be processed'.format(label, len(self.data_info[i])))

            counter = 1
            for path in self.data_info[i]:
                image = processAugmented(path, 64)
                if not os.path.exists(join(out_dir,label)):
                    os.makedirs(join(out_dir,label))
                cv.imwrite('{}_{}.jpg'.format(join(out_dir,label,label),counter), image)
                counter += 1

    # generate the training and development dataset
    def generate_dataset(self, dev_ratio = 0.15):
        images_train = []
        labels_train = []
        images_dev = []
        labels_dev = []

        for i in range(len(self.data_labels)):
            label = self.data_labels[i]
            counter = 1

            for path in self.data_info[i]:
                image = cv.imread(path)
                if counter < dev_ratio*len(self.data_info[i]):
                    images_dev.append(image)
                    labels_dev.append(category[label])
                else:
                    images_train.append(image)
                    labels_train.append(category[label])
                counter += 1

        images_train = np.array(images_train)
        images_dev = np.array(images_dev)
        labels_train = np.array(labels_train).reshape(-1,1)
        labels_dev = np.array(labels_dev).reshape(-1, 1)

        print('images_train shape : {}\nlabel_train shape: {}'.format(images_train.shape,labels_train.shape))
        print('images_dev shape : {}\nlabel_dev shape: {}'.format(images_dev.shape, labels_dev.shape))

        f1 = h5py.File("train_aug.h5", "a")
        train_x = f1.create_dataset("train_set_x", data=images_train)
        train_y = f1.create_dataset("train_set_y", data=labels_train)
        print(train_x.shape, train_y.shape)
        f1.close()

        f2 = h5py.File("dev_aug.h5", "a")
        dev_x = f2.create_dataset("dev_set_x", data=images_dev)
        dev_y = f2.create_dataset("dev_set_y", data=labels_dev)
        print(dev_x.shape, dev_y.shape)
        f2.close()


def main():
    # generator = DataSetGenerator('Training_Images')
    # generator.generate_image('Training_paug')

    generator = DataSetGenerator('Training_paug')
    generator.generate_dataset(dev_ratio=0.15)


if __name__ == '__main__':
    main()