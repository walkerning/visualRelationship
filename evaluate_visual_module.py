# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from collections import namedtuple, OrderedDict

import scipy
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from visual_relation_module import VisualModule
from configuration import ModelConfig

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_file", "",
                       "Path to a pretrained visual model.")
tf.flags.DEFINE_string("annotation_file", "./annotations_train.json",
                       "Annotation file name.")
tf.flags.DEFINE_string("dataset_path", "./sg_dataset/sg_train_images",
                       "The path where the images are stored")
tf.flags.DEFINE_integer("same_obj_neg_samples", 3, "")
tf.flags.DEFINE_integer("diff_obj_neg_samples", 10, "")
tf.flags.DEFINE_integer("diff_obj_neg_num", 2, "")


def _get_box(bbox1, bbox2):
    return (min(bbox1[0], bbox2[0]),\
            min(bbox1[1], bbox2[1]),\
            max(bbox1[2], bbox2[2]),\
            max(bbox1[3], bbox2[3]))

def _add_to_dct(obj, dct):
    if obj not in dct:
        dct[obj] = len(dct)
    return dct[obj]

def main(_):
    assert FLAGS.checkpoint_file, "--checkpoint_file is required"
    annotations = json.load(open(FLAGS.annotation_file, "r"))

    model_config = ModelConfig()
    model = VisualModule(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    samples_dct = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)
        
        for img_fname, ann in annotations.iteritems():
            samples_dct[img_fname] = {"positive": [], "negative": []}
            img = Image.open(os.path.join(FLAGS.dataset_path, img_fname))
            Object = namedtuple("Object", ["bbox", "category"])
            objects = OrderedDict()
            rel_set = set()
            for rel in ann:
                obj = _add_to_dct(Object(tuple(rel["object"]["bbox"]), rel["object"]["category"]), objects)
                sub = _add_to_dct(Object(tuple(rel["subject"]["bbox"]), rel["subject"]["category"]), objects)
                rel_set.add((obj, sub))
            print("Handle pic {}: #objects {}; #relations {}".format(img_fname, len(objects), len(ann)))
            objects = list(objects)
            # Evaluate all the positive relationships
            for rel in ann:
                data = np.array(img.crop(_get_box(rel["object"]["bbox"], rel["subject"]["bbox"])).resize([224, 224], PIL.Image.BILINEAR))
                predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                samples_dct[img_fname]["positive"].append((rel["predicate"], rel["object"]["category"], rel["subject"]["category"], predictions[rel["predicate"]]))
                sort_inds = np.argsort(predictions)[::-1][:FLAGS.same_obj_neg_samples+1]
                find = np.where(sort_inds == rel["predicate"])[0]
                if find:
                    sort_inds = np.delete(sort_inds, find)
                else:
                    sort_inds = sort_inds[:FLAGS.same_obj_neg_samples]
                samples_dct[img_fname]["negative"].extend([(i, rel["object"]["category"], rel["subject"]["category"], predictions[i]) for i in sort_inds])
            print("\tadded {} same-obj negative relationships".format(len(samples_dct[img_fname]["negative"])))
            # Evaluate diff-obj negative relationship samples
            get_diff_neg_num = 0
            while get_diff_neg_num < FLAGS.diff_obj_neg_samples:
                obj = int(np.random.uniform(0, len(objects)))
                sub = int(np.random.uniform(0, len(objects)))
                if (obj, sub) in rel_set:
                    continue
                data = np.array(img.crop(_get_box(objects[obj].bbox, objects[sub].bbox)).resize([224, 224], PIL.Image.BILINEAR))
                predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                sort_inds = np.argsort(predictions)[::-1]
                samples_dct[img_fname]["negative"].extend([(i, objects[obj].category, objects[sub].category, predictions[i]) for i in sort_inds[:FLAGS.diff_obj_neg_num]])
                get_diff_neg_num += 1
            print("\tadded {} diff-obj negative relationships".format(FLAGS.diff_obj_neg_samples * FLAGS.diff_obj_neg_num))
            import pdb
            pdb.set_trace()

if __name__ == "__main__":
    tf.app.run()
