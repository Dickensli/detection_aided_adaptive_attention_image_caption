"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys

import base64
import csv
import h5py
import numpy as np


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
basedir = '/data/scratch/public/ImageCaptioning/'
infiles = [os.path.join(basedir, 'flickr30k_train.tsv.0'), os.path.join(basedir, 'flickr30k_train.tsv.1')]
outfile = os.path.join(basedir, 'train.hdf5')

num_imgs = 29000
feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    out = h5py.File(outfile, "w")

    indices = {}

    img_features = out.create_dataset(
        'image_features', (num_imgs, num_fixed_boxes, feature_length), 'f')
    img_bb = out.create_dataset(
        'image_bb', (num_imgs, num_fixed_boxes, 4), 'f')
    spatial_img_features = out.create_dataset(
        'spatial_features', (num_imgs, num_fixed_boxes, 6), 'f')

    counter = 0

    print("reading tsv...")
    for infile in infiles:
        with open(infile, "r+b") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = int(item['image_id'])
                image_w = float(item['image_w'])
                image_h = float(item['image_h'])
                bboxes = np.frombuffer(
                    base64.decodestring(item['boxes']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))

                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h

                box_width = box_width[..., np.newaxis]
                box_height = box_height[..., np.newaxis]
                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                    scaled_y,
                    scaled_x + scaled_width,
                    scaled_y + scaled_height,
                    scaled_width,
                    scaled_height),
                    axis=1)

                indices[image_id] = counter
                img_bb[counter, :, :] = bboxes
                img_features[counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                spatial_img_features[counter, :, :] = spatial_features
                counter += 1
                if counter % 100 == 0:
                    print('{}/{} done'.format(counter, num_imgs))

    assert counter == num_imgs
    out.close()
    print("done!")
