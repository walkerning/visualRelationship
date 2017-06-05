from __future__ import print_function
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import argparse

import math
import scipy
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from visual_relation_module import VisualModule
from configuration import ModelConfig

def read_file(filename):
	if not os.path.isfile(filename):
		print("file {} does not exist\n".format(filename))
	return json.load(open(filename, 'r'))

def _get_box(bbox1, bbox2):
    return (min(bbox1[2], bbox2[2]),\
            min(bbox1[0], bbox2[0]),\
            max(bbox1[3], bbox2[3]),\
            max(bbox1[1], bbox2[1]))

# def zero_shot(objects, train, key, triples, dataset_dir):
# 	union = []
# 	for obj in objects:
# 		key1 = (key[0], key[1], obj)
# 		key2 = (key[0], obj, key[2])
# 		if key1 in triples:
# 			v = triples[key1]
# 			b1 = train[v[0]][v[1]]['object']['bbox']
# 			b2 = train[v[0]][v[1]]['subject']['bbox']
# 			u = _get_box(b1, b2)
# 			img = Image.open(os.path.join(dataset_dir, v[0]))
			
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_file", "","Path to a pretrained visual model.")
tf.flags.DEFINE_string("data_dir","/home/mxy/json_data/","the json_data dir")
tf.flags.DEFINE_string("train_fname","annotations_train.json","train json")
tf.flags.DEFINE_string("test_fname", "annotations_test.json", "test json")
tf.flags.DEFINE_string("object_fname", "objects.json", "object list")
tf.flags.DEFINE_string("dataset_dir","/home/mxy/json_data/sg_dataset/sg_train_images", "sp_data dir")
tf.flags.DEFINE_string("output_fname","/home/mxy/json_data/unique.txt","list of unique triples")

def _get_logits(sess, data, model):
    return np.squeeze(
            sess.run(model.prediction, 
            feed_dict = {
                model.image_feed: data.tobytes()}))

def main():
    train = read_file(os.path.join(FLAGS.data_dir, FLAGS.train_fname))
    test = read_file(os.path.join(FLAGS.data_dir, FLAGS.test_fname))
    objects = read_file(os.path.join(FLAGS.data_dir, FLAGS.object_fname))
    model_config = ModelConfig()
    model = VisualModule(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    unique = {}
    triples = {}
    pred_rel_lst = {}
    output = open(FLAGS.output_fname, 'wb')
    for ii in train:
    	for jj in range(len(train[ii])):
    	    triples[(train[ii][jj]['predicate'], train[ii][jj]['object']['category'], train[ii][jj]['subject']['category'])] = [ii, jj]
    with tf.Session() as sess:
    	sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)
	for ii in test:
	    pred_rel_lst[ii] = []
	    for jj in range(len(test[ii])):
	    	key = (test[ii][jj]['predicate'], test[ii][jj]['object']['category'], test[ii][jj]['subject']['category'])
	    	if key not in triples:
	    	    unique[key] = [ii, jj]
	    	    output.write("{} {} : {} {} {}\n".format(ii, jj, key[0], key[1], key[2]))
	    	pred_rel_lst[ii].append(key[0])
	top_1_acc = 0
	top_5_acc = 0
	for key in unique:
	    ii, jj = unique[key]
	    pred = key[0]
	    b1 = test[ii][jj]['object']['bbox']
	    b2 = test[ii][jj]['subject']['bbox']
	    u = _get_box(b1, b2)
	    img = Image.open(os.path.join(FLAGS.dataset_dir, ii))
	    union = np.array(img.crop(u))
	    union = np.resize(union, [224, 224, 3])
	    pred_lst = _get_logits(sess, union, model)
	    sort_inds = np.FLAGSort(pred_lst)[::-1]
	    if pred == pred_lst[sort_inds[0]]:
		top_1_acc += 1
	    if pred in pred_lst[sort_inds[0:5]]:
		top_5_acc += 1
	print("\n\ttop 1 acc: {} \n\ttop 5 acc: {}\n".format(top_1_acc / float(len(unique)), top_5_acc / float(len(unique))))


    output.close()

if __name__ == "__main__":
    main()
