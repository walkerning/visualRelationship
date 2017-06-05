"""
The script for preprocessing data for visual module training.
"""

from __future__ import print_function
import os
import numpy as np
import PIL
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
    # bbox is (y_min, y_max, x_min, x_max)
    # crop is (x_min, y_min, x_max, y_max)
    crop = (min(bbox1[2], bbox2[2]),\
            min(bbox1[0], bbox2[0]),\
            max(bbox1[3], bbox2[3]),\
            max(bbox1[1], bbox2[1]))
    height = crop[3] - crop[1]
    width = crop[2] - crop[0]
    mask = np.zeros((height, width), dtype=np.uint8)
    # mask1 = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.uint8)
    # mask2 = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.uint8)
    # mask1[bbox1[0]-crop[1]:bbox1[1]-crop[1], bbox1[2]-crop[0]:bbox1[3]-crop[0]] = 1
    # mask2[bbox2[0]-crop[1]:bbox2[1]-crop[1], bbox2[2]-crop[0]:bbox2[3]-crop[0]] = 1
    new_bbox1 = [bbox1[0]-crop[1], bbox1[1]-crop[1], bbox1[2]-crop[0], bbox1[3]-crop[0]]
    new_bbox2 = [bbox2[0]-crop[1], bbox2[1]-crop[1], bbox2[2]-crop[0], bbox2[3]-crop[0]]
    mask[new_bbox1[0]:new_bbox1[1], new_bbox1[2]:new_bbox2[3]] = 1
    mask[new_bbox2[0]:new_bbox2[1], new_bbox2[2]:new_bbox2[3]] = 1
    return crop, mask, new_bbox1, new_bbox2#, mask1, mask2


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

    parser.add_argument("--process_bbox", help = "store union of two bboxs", default = "true")
    parser.add_argument("--tfrecord_num", help = "store number of images per tfrecord", default = 100)
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
    if args.process_bbox == "true":
        for ii in train:
            
            for jj in range(len(train[ii])):
                # bbox = [Ymin, Ymax, Xmin, Xmax]
                bbox1 = train[ii][jj]['object']['bbox']
                bbox2 = train[ii][jj]['subject']['bbox']
                #bbox1 = transform_bbox(bbox1)
                #bbox2 = transform_bbox(bbox2)
                predicate = train[ii][jj]['predicate']
                fname = os.path.join(args.dataset_dir, ii)
                if not os.path.isfile(fname):
                    print("file {} does not exist".format(fname))
                    continue
                img = Image.open(fname)
                #img = img.load()
                bbox, mask, new_bbox1, new_bbox2 = _get_box(bbox1, bbox2)
                new_bbox1[:2] = (np.array(new_bbox1[:2], dtype=np.float) / mask.shape[0] * 224).astype(int)
                new_bbox1[2:] = (np.array(new_bbox1[2:], dtype=np.float) / mask.shape[1] * 224).astype(int)
                new_bbox2[:2] = (np.array(new_bbox2[:2], dtype=np.float) / mask.shape[0] * 224).astype(int)
                new_bbox2[2:] = (np.array(new_bbox2[2:], dtype=np.float) / mask.shape[1] * 224).astype(int)

                b = np.array(Image.fromarray(np.array(img.crop(bbox)) * mask[:, :, np.newaxis]).resize([args.bbox_height, args.bbox_width], PIL.Image.BILINEAR))
                # 直接在开始把object, subject按正负分肯定不行... 都不会激活了...
                # 要怎么样在某层卷积层提完大概区域的feature后, 加入object和subject的信息... 用receptive field对应实在是不准...
                # 一般的detection都是从feature到bbox regression坐标. 现在想从bbox映射回feature......有点难
                # 那不用正负分的话. 即使用attention方式从object的mask出发, 然后不同predicate分别attend到subject要关心的位置...
                # 也需要知道从一张224x224的大图上的一个box的1卷完之后怎么得到和pool5后一样大的7x7的map...
                # 是不是pool5也太高层了... receptive field是不是有点太大? 对于要attend到小物体检测是不是不好?
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'bbox': _bytes_feature(b.tostring()),
                    #"object_bbox": 
                    'predicate' : _int64_feature(predicate)}))

                tfrecord_writer.write(example.SerializeToString())
                tmp += 1
                if tmp % args.tfrecord_num == 0:
                    tfrecord_writer.close()
                    idx += 1
                    record_fname = _get_output_filename(args.tfrecord_dir, args.tfrecord_fname, idx)
                    print("Saving to {}".format(record_fname))
                    tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)
            cnt += 1
            print("finish {}".format(cnt / float(len(train))))

    else :
        for ii in train:
            fname = os.path.join(args.dataset_dir, ii)
            if not os.path.isfile(fname):
                print("file {} does not exist".format(fname))
                continue
            img = Image.open(fname)
            # size(height, width)
            size = img.size
            height = size[0]
            width = size[1]
            img = np.array(img)
            bbox1 = []
            bbox2 = []
            predicate = []
            for jj in range(len(train[ii])):
                bbox1.extend(train[ii][jj]['object']['bbox'])
                bbox2.extend(train[ii][jj]['subject']['bbox'])
                predicate.append(train[ii][jj]['predicate'])

            bbox1 = np.array(bbox1)
            bbox2 = np.array(bbox2)
            predicate = np.array(predicate)
            example = tf.train.Example(features = tf.train.Features(feature = {\
                'bbox1': _bytes_feature(bbox1.tostring()),\
                'bbox2': _bytes_feature(bbox2.tostring()),\
                'image': _bytes_feature(img.tostring()),\
                'height': _int64_feature(height),\
                'width': _int64_feature(width),\
                'predicate': _bytes_feature(predicate.tostring())\
                }))
            tfrecord_writer.write(example.SerializeToString())
            cnt += 1
            print("finish {}".format(cnt / float(len(train))))
            if cnt % args.tfrecord_num == 0:
                tfrecord_writer.close()
                idx += 1
                record_fname = _get_output_filename(args.tfrecord_dir, args.tfrecord_fname, idx)
                print("Saving to {}".format(record_fname))
                tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)

    tfrecord_writer.close()

        
if __name__ == "__main__":
    main()

