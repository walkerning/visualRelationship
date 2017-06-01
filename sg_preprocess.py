from __future__ import print_function
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import argparse

def read_file(data_dir = ".", train_fname = "annotations_train.json"):
    train = json.load(open(os.path.join(data_dir, train_fname), 'r'))
    #val = json.load(open(os.path.join(data_dir, test_fname), 'r'))
    return train

def transform_bbox(bbox):
    return [bbox[0], bbox[2], bbox[1], bbox[3]]

def _get_box(bbox1, bbox2):
    return (min(bbox1[0], bbox2[0]),\
            min(bbox1[1], bbox2[1]),\
            max(bbox1[2], bbox2[2]),\
            max(bbox1[3], bbox2[3]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _get_output_filename(output_dir, basename, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, basename, idx)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", help = "the json_data dir", default = "/home/mxy/json_data/")
    parser.add_argument("--train_fname", help = "train json", default = "annotations_train.json")
    #parser.add_argument("--test_fname", help = "test json", default = "annotations_test.json")
    
    parser.add_argument("--tfrecord_dir", help = "tfrecord directory", default = ".")
    parser.add_argument("--tfrecord_fname", help = "tfrecord basename", default = "tf_record")
   
    parser.add_argument("--dataset_dir", help = "sp_data dir", default = "/home/mxy/json_data/sg_dataset/sg_train_images")
   
    parser.add_argument("--bbox_height", help = "bbox height", default = 224)
    parser.add_argument("--bbox_width", help = "bbox width", default = 224)

    args = parser.parse_args()
    
    print("Reading from {}".format(os.path.join(args.data_dir, args.train_fname)))
    #print("Reading from {}".format(os.path.join(args.data_dir, args.test_fname)))
    train = read_file(args.data_dir, args.train_fname)
    #record_fname = os.path.join(args.tfrecord_dir, args.tfrecord_fname)
    #record_fname.append(".tfrecord")
    
    idx = 0
    record_fname = _get_output_filename(args.tfrecord_dir, args.tfrecord_fname, idx)
    tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
    print("Saving to {}".format(record_fname))
 
    cnt = 0
    tmp = 0
    for ii in train:
        
        for jj in range(len(train[ii])):
            # bbox = [Ymin, Ymax, Xmin, Xmax]
            bbox1 = train[ii][jj]['object']['bbox']
            bbox2 = train[ii][jj]['subject']['bbox']
            bbox1 = transform_bbox(bbox1)
            bbox2 = transform_bbox(bbox2)
            predicate = train[ii][jj]['predicate']
            fname = os.path.join(args.dataset_dir, ii)
            if not os.path.isfile(fname):
                print("file {} does not exist".format(fname))
                continue
            img = Image.open(fname)
            #img = img.load()
            bbox = _get_box(bbox1, bbox2)

            b = np.array(img.crop(bbox))
            b = np.resize(b, [args.bbox_height, args.bbox_width, 3])

            example = tf.train.Example(features = tf.train.Features(feature = { \
                    'bbox': _bytes_feature(b.tostring()),\
                    'predicate' : _int64_feature(predicate)}))
            tfrecord_writer.write(example.SerializeToString())
            tmp += 1
            if tmp % 100 == 0:
                tfrecord_writer.close()
                idx += 1
                record_fname = _get_output_filename(args.tfrecord_dir, args.tfrecord_fname, idx)
                print("Saving to {}".format(record_fname))
                tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
        cnt += 1
        print("finish {}".format(cnt / float(len(train))))
    tfrecord_writer.close()

        
if __name__ == "__main__":
    main()

