# -*- coding: utf-8 -*-
"""
Script for evaluating visual module. Will generate a text file containing pos/neg visual scores or reporting recalls.

@TODO: move the recall calculating utility out, keep it as a function, as we will need to evalaute visual module and language module together soon.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cPickle
from collections import namedtuple, OrderedDict
from itertools import product

import scipy
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from visual_relation_module import VisualModule
from configuration import ModelConfig

NEG_MARGIN = 0.7

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_file", "",
                       "Path to a pretrained visual model.")
tf.flags.DEFINE_string("annotation_file", "./annotations_train.json",
                       "Annotation file name.")
tf.flags.DEFINE_string("dataset_path", "./sg_dataset/sg_train_images",
                       "The path where the images are stored.")
tf.flags.DEFINE_integer("same_obj_neg_samples", 3, "")
tf.flags.DEFINE_integer("diff_obj_neg_samples", 10, "")
tf.flags.DEFINE_integer("diff_obj_neg_num", 2, "")
tf.flags.DEFINE_boolean("cal_recall", True, 
                        "Whether or not to calucate recall.")
tf.flags.DEFINE_boolean("cal_vscore", True,
                        "Whether or not to calucate visual score for language model training.")
tf.flags.DEFINE_boolean("verbose", False,
                        "Whether or not to print more verbose information.")
tf.flags.DEFINE_string("save", "./visual_scores.txt",
                       "The pkl file name to save the pos/neg visual scores.")

def _get_box(bbox1, bbox2):
    return (min(bbox1[0], bbox2[0]),\
            min(bbox1[2], bbox2[2]),\
            max(bbox1[1], bbox2[1]),\
            max(bbox1[3], bbox2[3]))

def _add_to_dct(obj, dct):
    if obj not in dct:
        dct[obj] = len(dct)
    return dct[obj]

def main(_):
    assert FLAGS.checkpoint_file, "--checkpoint_file is required"
    assert FLAGS.cal_vscore or FLAGS.cal_recall, "What are you doing if you are not calculating vscore neither recall???"
    annotations = json.load(open(FLAGS.annotation_file, "r"))

    model_config = ModelConfig()
    model = VisualModule(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    #samples_dct = {}
    recalls_50_dct = {}
    recalls_100_dct = {}

    # FIXME: 是不是其实可以cross-image, 直接把所有positive拼在一起, 所有negative拼在一起就行了
    print("Saving pos/neg visual scores to {}...".format(FLAGS.save))
    samples_wf = open(FLAGS.save, "w")
    ind_fname_wf = open(FLAGS.save + ".ind_fnamemap.txt", "w")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)

        for ind, (img_fname, ann) in enumerate(annotations.iteritems()):
            if len(ann) == 0:
                continue
            #samples_dct[img_fname] = {"positive": [], "negative": []}
            samples_dct = {"positive": [], "negative": []}

            img = Image.open(os.path.join(FLAGS.dataset_path, img_fname))
            Object = namedtuple("Object", ["bbox", "category"])
            objects = OrderedDict()
            rel_set = set() # positive relation set without predicate, for negative triple sample generation
            rel_pred_set = set() # positive relation set with predicate, for recall calculation
            for rel in ann:
                obj = _add_to_dct(Object(tuple(rel["object"]["bbox"]), rel["object"]["category"]), objects)
                sub = _add_to_dct(Object(tuple(rel["subject"]["bbox"]), rel["subject"]["category"]), objects)
                rel_set.add((obj, sub))
                rel_pred_set.add((rel["predicate"], obj, sub))
            print("{}: Handle pic {}; #objects {}; #relations {}".format(ind, img_fname, len(objects), len(ann)))
            # `objects` holds unique objects in this image
            objects = list(objects)

            if FLAGS.cal_recall:
                # calcualte recall@100, recall@50
                predict_rel_lst = []
                predict_lst = []

                # iterate all possible object pairs
                pairs = list(product(range(len(objects)), range(len(objects))))
                for pair in pairs:
                    if pair[0] == pair[1]:
                        # relation must be between two different objects
                        continue
                    predict_rel_lst.extend([(i, pair[0], pair[1]) for i in range(70)])
                    data = np.array(img.crop(_get_box(objects[pair[0]].bbox, objects[pair[1]].bbox)).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    predict_lst.extend(predictions)

                sort_inds = np.argsort(predict_lst)[::-1]
                # recall 100
                recalls_100_dct[img_fname] = float(len(set([tuple(r) for r in np.array(predict_rel_lst)[sort_inds[:100]].tolist()]).intersection(rel_pred_set))) / len(ann)
                # recall 50
                recalls_50_dct[img_fname] = float(len(set([tuple(r) for r in np.array(predict_rel_lst)[sort_inds[:50]].tolist()]).intersection(rel_pred_set))) / len(ann)
                if FLAGS.verbose:
                    print("\trecall@50: {}\n\trecall@100: {}".format(recalls_50_dct[img_fname], recalls_100_dct[img_fname]))

            if FLAGS.cal_vscore:
                # Evaluate all the positive relationships
                for rel in ann:
                    data = np.array(img.crop(_get_box(rel["object"]["bbox"], rel["subject"]["bbox"])).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    samples_dct["positive"].append((rel["predicate"], rel["object"]["category"], rel["subject"]["category"], predictions[rel["predicate"]]))
                    # Add several negative relations that have the same object pair as this positive relation
                    sort_inds = np.argsort(predictions)[::-1][:FLAGS.same_obj_neg_samples+1]
                    find = np.where(sort_inds == rel["predicate"])[0]
                    if find:
                        sort_inds = np.delete(sort_inds, find)
                    else:
                        sort_inds = sort_inds[:FLAGS.same_obj_neg_samples]
                    samples_dct["negative"].extend([(i, rel["object"]["category"], rel["subject"]["category"], predictions[i]) for i in sort_inds])
                print("\tadded {} same-obj negative relationships".format(len(samples_dct["negative"])))
    
                # Evaluate diff-obj negative relationship samples
                get_diff_neg_num = 0
                # There are not enough negative samples in some pictures
                # multiply by `NEG_MARGIN` to avoid sampling too slow
                num_diff_neg_samples = min(FLAGS.diff_obj_neg_samples,
                                           int((len(objects) * (len(objects) - 1) - len(rel_set)) * NEG_MARGIN))
                while get_diff_neg_num < num_diff_neg_samples:
                    obj = int(np.random.uniform(0, len(objects)))
                    sub = int(np.random.uniform(0, len(objects)))
                    if (obj, sub) in rel_set or obj == sub:
                        continue
                    data = np.array(img.crop(_get_box(objects[obj].bbox, objects[sub].bbox)).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    sort_inds = np.argsort(predictions)[::-1]
                    samples_dct["negative"].extend([(i, objects[obj].category, objects[sub].category, predictions[i]) for i in sort_inds[:FLAGS.diff_obj_neg_num]])
                    get_diff_neg_num += 1
                print("\tadded {} diff-obj negative relationships".format(num_diff_neg_samples * FLAGS.diff_obj_neg_num))
                for s in samples_dct["positive"]:
                    samples_wf.write("{} 1 {} {} {} {}\n".format(ind, *s))
                for s in samples_dct["negative"]:
                    samples_wf.write("{} 0 {} {} {} {}\n".format(ind, *s))
                ind_fname_wf.write("{} {}\n".format(ind, img_fname))

    if FLAGS.cal_recall:
        mean_recall50 = np.mean(recalls_50_dct.values())
        mean_recall100 = np.mean(recalls_100_dct.values())
        print("mean recall@50: {}\nmean recall@100: {}".format(mean_recall50, mean_recall100))
        cPickle.dump((recalls_50_dct, recalls_100_dct), open("mean_recalls.pkl", "r"))

if __name__ == "__main__":
    tf.app.run()
