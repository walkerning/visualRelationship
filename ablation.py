# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import os
import json
import numpy as np
import PIL
from PIL import Image
import scipy
import argparse
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from visual_relation_module import VisualModule
from configuration import ModelConfig


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_file", "",
                        "Path to a pretrained visual model.")
tf.flags.DEFINE_string("dataset_path", 
                       "./ablation_pics",
                       "The path where the images are stored.")

def _get_box(bbox1, bbox2):
    return (min(bbox1[2], bbox2[2]),\
            min(bbox1[0], bbox2[0]),\
            max(bbox1[3], bbox2[3]),\
            max(bbox1[1], bbox1[1]))

def _get_logits(sess, data, model):
    return np.squeeze(
            sess.run(model.prediction, 
            feed_dict = {
                model.image_feed: data.tobytes()}))

def _softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def _entropy(logits, pred):
    scores = _softmax(logits)
    return -math.log(scores[pred])

def _remove_square(image, size, Ystart, Xstart, number):
    # black out image[Ystart : Ystart + size, Xstart : Xstart + 30]
    img = np.array(image)
    for ii in range(Ystart, Ystart + size):
        for jj in range(Xstart, Xstart + size):
            try:
                img[ii, jj] = (0, 0, 0)
            except:
                import pdb
                pdb.set_trace()

    number[Ystart:Ystart+size, Xstart:Xstart+size] = number[Ystart:Ystart + size, Xstart:Xstart + size] + 1
    return img

def _roll_image(sess, img, size, pred, model, stride=15):
    img_size = img.size
    logits_init = _get_logits(sess, img, model)
    loss_init = _entropy(logits_init, pred)
    print("init loss is {}".format(loss_init))
    
    #loss_metric = np.zeros(((img_size[0] - size) / stride, (img_size[1] - size) / stride))
    losses = np.zeros((img_size[1], img_size[0]))
    number = np.zeros((img_size[1], img_size[0]), dtype=int)
    
    for ii in range(0, img_size[1] - size+stride, stride):
        for jj in range(0, img_size[0] - size+stride, stride):
            if ii >= img_size[1] - size:
                ii = img_size[1] - size
            if jj >= img_size[0] - size:
                jj = img_size[0] - size
            black_image = _remove_square(img, size, ii, jj, number)
            logits = _get_logits(sess, black_image, model)
            losses[ii:ii+size, jj:jj+size] = losses[ii:ii+size, jj:jj+size] + _entropy(logits, pred) - loss_init

    return losses, number

def _metric_mean(loss_metric, size = 30):
    img_size = loss_metric.shape
    new_metric = np.zeros((img_size[0], img_size[1]))

    for ii in range(img_size[0]):
        for jj in range(img_size[1]):
            cut = loss_metric[max(ii - size + 1, 0):min(ii+1, img_size[0]-size), 
                              max(jj - size + 1, 0):min(jj+1, img_size[1]-size)]
            new_metric[ii][jj] = np.mean(cut)
    return new_metric
    

def main():
    assert FLAGS.checkpoint_file, "--checkpint_file is required"
    model_config = ModelConfig()
    model = VisualModule(model_config, mode="inference")
    model.build()
    saver = tf.train.Saver()

    images_name = ["1807338675_5e13fe07f9_o_0_5_20.jpg"]#"9399147028_3927b000f1_b_1_33_0.jpg"]
    prediction = [0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_file)
        for ii in range(len(images_name)):
            image_fname = os.path.join(FLAGS.dataset_path, images_name[ii])
            if not os.path.isfile(image_fname):
                print("file {} does not exist".format(image_fname))
                continue
            img = Image.open(image_fname)
            assert (img.height, img.width) == (224, 224)
            loss, number = _roll_image(sess, img, 50, prediction[ii], model)
            loss_mean = loss / number
            loss_mean.tofile("./ablation_results/ablation_test_output_{}_{}.npz".format(images_name[ii], ii))
            show_attend = np.maximum(loss_mean, 0)
            show_attend = show_attend / np.max(show_attend)
            fig, axes = plt.subplots(2, 1)
            fig.suptitle("ablation: " + images_name[ii])
            axes[0].imshow(np.array(img).astype(np.float) / 255)
            axes[1].imshow(show_attend[:, :, np.newaxis] * np.array(img)/255)

    plt.show()
            

if __name__ == "__main__":
    main()
