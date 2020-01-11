import os
import glob
import h5py
import random
#import matplotlib.pyplot as plt

from PIL import Image 
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

### Read h5 format data file
def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

### Prepare a series of train data names
def prepare_data(sess, dataset):
    data_path = os.path.join(os.getcwd(), dataset)
    data_name = glob.glob(os.path.join(data_path, "*.png"))
    return data_name

### Make input data as h5 file format
def make_data(sess, data, label, config, h5_name):
    savepath = os.path.join(os.getcwd(), config.dataset_dir, h5_name)

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

### Save image to h5 file
def input_setup(sess, config, stage):
    # Load data path
    if stage == 'Train':
        data_input = prepare_data(sess, dataset=os.path.join(os.getcwd(), config.train_dir, 'input'))
        data_label = prepare_data(sess, dataset=os.path.join(os.getcwd(), config.train_dir, 'label'))
        h5_name = config.train_h5_name
    elif stage == "Val": 
        data_input = prepare_data(sess, dataset=os.path.join(os.getcwd(), config.val_dir, 'input'))
        data_label = prepare_data(sess, dataset=os.path.join(os.getcwd(), config.val_dir, 'label'))
        h5_name = config.val_h5_name

    sub_input_sequence = []
    sub_label_sequence = []

    for i in range(len(data_input)):
        im_input_ = Image.open(data_input[i])
        im_label_ = Image.open(data_label[i])

        input_ = np.array(im_input_, dtype="float32") / 255.0
        label_ = np.array(im_label_, dtype="float32") / 255.0

        h, w = input_.shape
        for x in range(0, h - config.image_height + 1, config.stride):
            for y in range(0, w - config.image_width + 1, config.stride):
                sub_input = input_[x:x + config.image_height, y:y + config.image_width]
                sub_label = label_[x:x + config.image_height, y:y + config.image_width]
                # Add channel dim
                sub_input = sub_input.reshape([config.image_height, config.image_width, 1])
                sub_label = sub_label.reshape([config.image_height, config.image_width, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    make_data(sess, arrdata, arrlabel, config, h5_name)


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

def save_images(filepath, data, label, pred):
    data = np.squeeze(data)
    label = np.squeeze(label)
    pred = np.clip(np.squeeze(pred), 0, 1)

    cat_image = np.concatenate([data, label, pred], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
