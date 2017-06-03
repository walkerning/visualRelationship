# -*- coding: utf-8 -*-
"""
Script for evaluating visual module. Will generate a text file containing pos/neg visual scores or reporting recalls.

@TODO: batch the inference...
@TODO: this script share many codes with evaluate_language_module.py.. If have time, refactor the code...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import cPickle

import scipy
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from evaluate_utils import parse_annotation, get_union_box, calculate_recall, get_post_process_func
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
tf.flags.DEFINE_string("post_process", "none",
                      "Post process the logits to get v score. Default `none`, can choose `softmax`, `relu`.")
tf.flags.DEFINE_boolean("cal_recall", True, 
                        "Whether or not to calucate recall.")
tf.flags.DEFINE_boolean("cal_vscore", True,
                        "Whether or not to calucate visual score for language model training.")
tf.flags.DEFINE_integer("same_obj_neg_samples", 3, "")
tf.flags.DEFINE_integer("diff_obj_neg_samples", 10, "")
tf.flags.DEFINE_integer("diff_obj_neg_num", 2, "")
tf.flags.DEFINE_boolean("verbose", False,
                        "Whether or not to print more verbose information.")
tf.flags.DEFINE_string("save", "",
                       "The pkl file name to save the pos/neg visual scores.")

def main(_):
    assert FLAGS.checkpoint_file, "--checkpoint_file is required"
    assert FLAGS.cal_vscore or FLAGS.cal_recall, "What are you doing if you are not calculating vscore neither recall???"
    post_process_vscore = get_post_process_func[FLAGS.post_process]

    if FLAGS.cal_vscore:
        assert FLAGS.save, "--save is reuiqred when calculating vscores "

    annotations = json.load(open(FLAGS.annotation_file, "r"))
    num_examples = len(annotations)
    num_actual_examples = 0
    print("Readed {} examples from {}".format(num_examples, FLAGS.annotation_file))

    model_config = ModelConfig()
    model = VisualModule(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()
    #samples_dct = {}
    recalls_50_dct = {}
    recalls_100_dct = {}

    # FIXME: 是不是其实可以cross-image, 直接把所有positive拼在一起, 所有negative拼在一起就行了
    if FLAGS.cal_vscore:
        print("Saving pos/neg visual scores to {}...".format(FLAGS.save))
        samples_wf = open(FLAGS.save, "w")
        ind_fname_wf = open(FLAGS.save + ".ind_fnamemap.txt", "w")

    top1_correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)

        for ind, (img_fname, ann) in enumerate(annotations.iteritems()):
            if len(ann) == 0:
                continue
            try:
                img = Image.open(os.path.join(FLAGS.dataset_path, img_fname))
            except IOError:
                # Some pictures do not exists...
                continue
            samples_dct = {"positive": [], "negative": []}
            objects, rel_set, rel_pred_set = parse_annotation(ann)
            if len(objects) == 1:
                # Some pictures's bbox annotation is wrong... eg. "366773716_1d06242e38_o.jpg" in the train set
                continue
            num_actual_examples += 1
            print("{}: Handle pic {}; #objects {}; #relations {}".format(ind, img_fname, len(objects), len(ann)))

            if FLAGS.cal_recall:
                def get_pred(obj1, obj2):
                    """
                    obj1, obj2 are both of type `evluation_utils.Object`
                    """
                    data = np.array(img.crop(get_union_box(obj1.bbox, obj2.bbox)).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    return post_process_vscore(predictions)
                recalls_50_dct[img_fname], recalls_100_dct[img_fname], top_1_predictions = calculate_recall(rel_pred_set, objects, get_pred)
                cor = top_1_predictions[0][0] in rel_pred_set
                top1_correct += cor
                if FLAGS.verbose:
                    print("\trecall@50: {}\n\trecall@100: {}".format(recalls_50_dct[img_fname], recalls_100_dct[img_fname]))
                    obj, sub = objects[top_1_predictions[0][0][1]].category, objects[top_1_predictions[0][0][2]].category
                    print("\t{}: the top-1 prediction is ({} {} {}), score {}".format("CORRECT" if cor else "FALSE", top_1_predictions[0][0][0],
                                                                                      obj, sub, top_1_predictions[0][1]))

            if FLAGS.cal_vscore:
                # Evaluate all the positive relationships
                for rel in ann:
                    data = np.array(img.crop(get_union_box(rel["object"]["bbox"], rel["subject"]["bbox"])).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    predictions = post_process_vscore(predictions)
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
                    data = np.array(img.crop(get_union_box(objects[obj].bbox, objects[sub].bbox)).resize([224, 224], PIL.Image.BILINEAR))
                    predictions = np.squeeze(sess.run(model.prediction, feed_dict={model.image_feed: data.tostring()}))
                    predictions = post_process_vscore(predictions)
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
        print("number actual valid examples: {}".format(num_actual_examples))
        top1_correct = float(top1_correct) / num_actual_examples
        print("mean recall@50: {}\nmean recall@100: {}".format(mean_recall50, mean_recall100))
        print("top1 accuracy: {}".format(top1_correct))
        recall_fname = "v_mean_recalls_{}.pkl".format(int(time.time()))
        print("Writing recall information into {}.".format(recall_fname))
        cPickle.dump((recalls_50_dct, recalls_100_dct), open(recall_fname, "w"))

if __name__ == "__main__":
    tf.app.run()
