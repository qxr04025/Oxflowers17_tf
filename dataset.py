#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Load Oxford's 17 Category Flower Dataset, which has been split to train and val sets.

Author:qinxiaoran

"""

from __future__ import division, print_function, absolute_import

import os
import os.path
import numpy as np
from PIL import Image
import pickle

def load_data(dirname="17flowers", resize_pics=(224, 224), shuffle=True, one_hot=False):
    dataset_dir = os.path.join(dirname, 'jpg')
    trainset_dir = os.path.join(dirname, 'train')
    valset_dir = os.path.join(dirname, 'val')
    trainset_file = os.path.join(dirname, '17flowers_train.pkl')
    valset_file = os.path.join(dirname, '17flowers_val.pkl')
    if not os.path.exists(trainset_file) or not os.path.exists(valset_file):
        assert os.path.exists(dataset_dir), 'Original dataset not exists!'
        build_train_val_directories(dataset_dir, trainset_dir, valset_dir)

    X_train, Y_train = build_image_dataset_from_dir(os.path.join(trainset_dir),
                                        dataset_file=trainset_file,
                                        resize=resize_pics,
                                        filetypes=['.jpg', '.jpeg'],
                                        convert_gray=False,
                                        shuffle_data=shuffle,
                                        categorical_Y=one_hot)
    X_val, Y_val = build_image_dataset_from_dir(os.path.join(valset_dir),
                                                    dataset_file=valset_file,
                                                    resize=resize_pics,
                                                    filetypes=['.jpg', '.jpeg'],
                                                    convert_gray=False,
                                                    shuffle_data=shuffle,
                                                    categorical_Y=one_hot)
    return X_train, Y_train, X_val, Y_val


def build_train_val_directories(dataset_dir, trainset_dir, valset_dir):
    if not os.path.exists(trainset_dir):
        os.mkdir(trainset_dir)
    if not os.path.exists(valset_dir):
        os.mkdir(valset_dir)
    dir_id = 0
    tclass_dir = os.path.join(trainset_dir, str(dir_id))
    vclass_dir = os.path.join(valset_dir, str(dir_id))
    if not os.path.exists(tclass_dir):
        os.mkdir(tclass_dir)
    if not os.path.exists(vclass_dir):
        os.mkdir(vclass_dir)
    pid = 1
    for i in range(1, 1361):
        fname = "image_" + ("%.4i" % i) + ".jpg"
        if pid % 10 != 0:
            os.rename(os.path.join(dataset_dir, fname), os.path.join(tclass_dir, fname))
        else:
            os.rename(os.path.join(dataset_dir, fname), os.path.join(vclass_dir, fname))
        pid += 1
        if i % 80 == 0 and dir_id < 16:
            dir_id += 1
            pid = 1
            tclass_dir = os.path.join(trainset_dir, str(dir_id))
            os.mkdir(tclass_dir)
            vclass_dir = os.path.join(valset_dir, str(dir_id))
            os.mkdir(vclass_dir)

def build_image_dataset_from_dir(directory,
                                 dataset_file,
                                 resize=None, convert_gray=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_gray, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1) # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        pickle.dump((X, Y), open(dataset_file, 'wb'))
    return X, Y

#========================image_dirs_to_samples======================
def image_dirs_to_samples(directory, resize=None, convert_gray=None,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        if resize:
            samples[i] = resize_image(samples[i], resize[0], resize[1])
        if convert_gray:
            samples[i] = convert_color(samples[i], 'L')
        samples[i] = pil_to_nparray(samples[i])
        samples[i] /= 255.
    print("Parsing Done!")
    return samples, targets

def directory_to_samples(directory, flags=None):
    """ Read a directory, and list all subdirectories files as class sample """
    samples = []
    targets = []
    label = 0
    try: # Python 2
        classes = sorted(os.walk(directory).next()[1])
    except Exception: # Python 3
        classes = sorted(os.walk(directory).__next__()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        try: # Python 2
            walk = os.walk(c_dir).next()
        except Exception: # Python 3
            walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            if not flags or any(flag in sample for flag in flags):
                samples.append(os.path.join(c_dir, sample))
                targets.append(label)
        label += 1
    return samples, targets

def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """ Resize an image.
    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        out_image: `str`. If specified, save the image to the given path.
        resize_mode: `PIL.Image.mode`. The resizing mode.
    Returns:
        `PIL.Image`. The resize image.
    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def convert_color(in_image, mode):
    """ Convert image color with provided `mode`. """
    return in_image.convert(mode)


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

#========================one_hot_encoding======================
def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y

#========================shuffle======================
def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)
