# -*- coding: utf-8 -*-

import os
import json
import argparse

from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_file", help="The tensorflow checkpoint file.")
    parser.add_argument("--save-dir", default="./attention_ims/",
                        help="The directory to save result pics.")
    parser.add_argument("--pred-file", default="./predicates.json",
                        help="The json file of predicates")
    args = parser.parse_args()

    reader = pywrap_tensorflow.NewCheckpointReader(args.checkpoint_file)
    predicate_embedding = reader.get_tensor('predicate_embedding')
    weights = reader.get_tensor('embedding_to_attention/fully_connected/weights')
    biases = reader.get_tensor('embedding_to_attention/fully_connected/biases')
    pred_names = json.load(open(args.pred_file, "r"))
    
    attention = np.maximum(predicate_embedding.dot(weights) + biases, 0)
    # reception_size = 212 # 这个太大了.. 算有效reception size还要折腾... 先按stride 32算
    stride = 32
    fig, axes = plt.subplots(10, 7, figsize=(10, 10))
    axes = axes.flatten()
    for ind, att in enumerate(attention):
        att = att / np.max(att)
        att_map = np.zeros((224, 224), dtype=float)
        for ii in range(7):
            for jj in range(7): 
                att_map[ii*stride:(ii+1)*stride, jj*stride:(jj+1)*stride] = att[ii * 7 + jj]
        att_map = ndimage.gaussian_filter(att_map, sigma=(stride, stride))
        Image.fromarray((att_map * 255).astype(np.uint8)).save(os.path.join(args.save_dir, "{}.jpg".format(ind)))
        axes[ind].imshow(att_map, cmap="gray")
        axes[ind].set_title(pred_names[ind])

    plt.show()

if __name__ == "__main__":
    main()
