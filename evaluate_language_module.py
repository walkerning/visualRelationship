# -*- coding: utf-8 -*-
"""
Script for evaluating language module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cPickle

import scipy
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from evaluate_utils import parse_annotation, get_union_box, calculate_recall
from visual_relation_module import VisualModule
from language_module import LanguageModule
from configuration import ModelConfig, LanguageModelConfig


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("visual_checkpoint_file", "",
                       "Path to a pretrained visual model.")
tf.flags.DEFINE_string("language_checkpoint_file", "",
                       "Path to a pretrained language model.")
tf.flags.DEFINE_string("annotation_file", "./annotations_train.json",
                       "Annotation file name.")
tf.flags.DEFINE_string("dataset_path", "./sg_dataset/sg_train_images",
                       "The path where the images are stored.")
tf.flags.DEFINE_boolean("verbose", True,
                        "Print verbose information.")

def main(_):
    assert FLAGS.visual_checkpoint_file, "--visual_checkpoint_file is required"
    assert FLAGS.language_checkpoint_file, "--language_checkpoint_file is required"
    annotations = json.load(open(FLAGS.annotation_file, "r"))
    num_examples = len(annotations)
    num_actual_examples = 0
    print("Readed {} examples from {}".format(num_examples, FLAGS.annotation_file))

    model_config = ModelConfig()
    l_model_config = LanguageModelConfig()
    with tf.variable_scope("visual"):
        model = VisualModule(model_config, mode="inference")
        model.build()
    with tf.variable_scope("language"):
        l_model = LanguageModule(l_model_config, mode="inference")
        l_model.build()
    v_saver = tf.train.Saver({v.op.name[v.op.name.find("/", 1) + 1:]:v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="visual")})
    l_saver = tf.train.Saver({v.op.name[v.op.name.find("/", 1) + 1:]:v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="language")})
    recalls_50_dct = {}
    recalls_100_dct = {}

    top1_correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v_saver.restore(sess, FLAGS.visual_checkpoint_file)
        l_saver.restore(sess, FLAGS.language_checkpoint_file)

        for ind, (img_fname, ann) in enumerate(annotations.iteritems()):
            if len(ann) == 0:
                continue
            try:
                img = Image.open(os.path.join(FLAGS.dataset_path, img_fname))
            except IOError:
                continue
            samples_dct = {"positive": [], "negative": []}
            objects, rel_set, rel_pred_set = parse_annotation(ann)
            if len(objects) == 1:
                # Some pictures's bbox annotation is wrong... eg. "366773716_1d06242e38_o.jpg" in the train set
                continue
            num_actual_examples += 1
            print("{}: Handle pic {}; #objects {}; #relations {}".format(ind, img_fname, len(objects), len(ann)))

            def get_pred(obj1, obj2):
                """
                obj1, obj2 are both of type `evluation_utils.Object`
                """
                data = np.array(img.crop(get_union_box(obj1.bbox, obj2.bbox)).resize([224, 224], PIL.Image.BILINEAR))
                
                v_predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                l_predictions = np.squeeze(sess.run(l_model.prediction, feed_dict={l_model.obj1_feed: [obj1.category], l_model.obj2_feed: [obj2.category]}))
                return v_predictions * l_predictions

            recalls_50_dct[img_fname], recalls_100_dct[img_fname], top_1_predictions = calculate_recall(rel_pred_set, objects, get_pred)
            cor = top_1_predictions[0][0] in rel_pred_set
            top1_correct += cor
            if FLAGS.verbose:
                print("\trecall@50: {}\n\trecall@100: {}".format(recalls_50_dct[img_fname], recalls_100_dct[img_fname]))
                obj, sub = objects[top_1_predictions[0][0][1]].category, objects[top_1_predictions[0][0][2]].category
                print("\t{}: the top-1 prediction is ({} {} {}), score {}".format("CORRECT" if cor else "FALSE", top_1_predictions[0][0][0],
                                                                                  obj, sub, top_1_predictions[0][1]))


    mean_recall50 = np.mean(recalls_50_dct.values())
    mean_recall100 = np.mean(recalls_100_dct.values())
    print("number actual valid examples: {}".format(num_actual_examples))
    top1_correct = float(top1_correct) / num_actual_examples
    print("mean recall@50: {}\nmean recall@100: {}".format(mean_recall50, mean_recall100))
    print("top1 accuracy: {}".format(top1_correct))
    recall_fname = "vl_mean_recalls_{}.pkl".format(int(time.time()))
    print("Writing recall information into {}.".format(recall_fname))
    cPickle.dump((recalls_50_dct, recalls_100_dct), open(recall_fname, "r"))

if __name__ == "__main__":
    tf.app.run()