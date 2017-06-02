import tensorflow as tf
import os
import json
import numpy as np
import PIL
from PIL import Image
import scipy
import argparse

from visual_relation_module import VisualModule
from configuration import ModelConfig


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_file", "",
                        "Path to a pretrained visual model.")
tf.flags.DEFINE_string("annotation_file", 
                        "/home/mxy/json_data/annotations_train.json",
                        "Annotation file name.")
tf.flags.DEFINE_string("dataset_path", 
                       "/home/mxy/json_data/sg_dataset/sg_train_images",
                       "The path where the images are stored.")
def _get_box(bbox1, bbox2):
    return (min(bbox1[0], bbox2[0]),\
            min(bbox1[2], bbox2[2]),\
            max(bbox1[1], bbox2[1]),\
            max(bbox1[3], bbox1[3]))

def _get_logits(sess, data, model):
    return np.squeeze(
            sess.run(model.prediction, 
            feed_dict = {
                model.image_feed : model.image_feed : data.tostring()}))

def _softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def _entropy(logits, pred):
    scores = _softmax(logits)
    return -math.log(scores[pred])

def _remove_square(image, size = 30, Xstart, Ystart):
    # black out image[Ystart : Ystart + size, Xstart : Xstart + 30]
    img = np.array(image)
    img_size = img.shape
    for ii in range(Ystart, Ystart + size):
        for jj in range(Xstart, Xstart + size):
            img[ii][jj] = (0, 0, 0)
    return img

def _roll_image(sess, img, size, pred, model):
    img_size = img.size
    logits_init = _get_logits(sess, img, model)
    loss_init = _entropy(logits_init, pred)
    print("init loss is {}".format(loss_init))
    
    loss_metric = np.zeros(img_size[0] - size, img_size[1] - size)

    for ii in range(0, img_size[0] - size):
        for jj in range(0, img_size[1] - size):
            black_image = _remove_square(img, size, ii, jj)
            logits = _get_logits(sess, data, model)
            loss_metric[ii][jj] = _entropy(logits, pred)

    return loss_metric

def main():

    model_config = ModelConfig()
    model = VisualModule(model_config, mode = "inference")
    model.build()
    if not os.path.isfile(FLAGS.annotation_file):
        print("Failed reading from file {}".format(FLAGS.annotation_file))
    annotations = json.load(open(FLAGS.annotation_file, 'r'))
    saver = tf.train.Saver()

    image_name = []
    prediction = []

    with tf.Session() as sess:
        sess.run(tf.global_variales_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)
        for ii in images_name:
            image_fname = os.path.join(FLAGS.dataset_path, ii)
            if not os.path.isfile(image_fname):
                print("file {} does not exist".format(image_fname))
                continue
            img = Image.open(image_fname)

        
