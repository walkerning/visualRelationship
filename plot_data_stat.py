# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json

import pandas as pd
from matplotlib import pyplot as plt

def plot_stat(anns, objs, preds, phase):
    split = None
    #plot_n = int(math.ceil(float(len(anns)) / split))
    plot_n = 1
    ann_num_lst = [[0 for _ in range(70)] for _ in range(plot_n)]
    obj_num_lst = [[0 for _ in range(100)] for _ in range(plot_n)]
    i = 0
    j = 0
    for img_fname, ann in anns.iteritems():
        for an in ann:
            ann_num_lst[i][an["predicate"]] += 1
            # 这里不管重复的
            obj_num_lst[i][an["object"]["category"]] += 1
            obj_num_lst[i][an["subject"]["category"]] += 1
        j += 1
        if split and j == split:
            i += 1
            j = 0
    
    #pd.DataFrame(ann_num_lst, columns=preds).plot.bar()
    #pd.DataFrame(obj_num_lst, columns=objs).plot.bar()
    ann_num_lst = np.transpose(ann_num_lst)
    obj_num_lst = np.transpose(obj_num_lst)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("{}-predicates".format(phase))
    pd.DataFrame(ann_num_lst[:35], index=preds[:35]).plot.bar(ax=axes[0])
    pd.DataFrame(ann_num_lst[35:], index=preds[35:]).plot.bar(ax=axes[1])
    plt.subplots_adjust(hspace=0.8)
    fig2, axes2 = plt.subplots(2, 2, figsize=(10,10))
    fig2.suptitle("{}-objects".format(phase))
    axes2 = axes2.flatten()
    pd.DataFrame(obj_num_lst[:25], index=objs[:25]).plot.bar(ax=axes2[0])
    pd.DataFrame(obj_num_lst[25:50], index=objs[25:50]).plot.bar(ax=axes2[1])
    pd.DataFrame(obj_num_lst[50:75], index=objs[50:75]).plot.bar(ax=axes2[2])
    pd.DataFrame(obj_num_lst[75:100], index=objs[75:100]).plot.bar(ax=axes2[3])
    plt.subplots_adjust(hspace=0.8)


if __name__ == "__main__":
    # plot train
    anns_train = json.load(open("./annotations_train.json"))
    # plot test
    anns_test = json.load(open("./annotations_test.json"))
    objs = ["{}.{}".format(x[1], x[0]) for x in enumerate(json.load(open("./objects.json")))]
    preds = ["{}.{}".format(x[1], x[0]) for x in enumerate(json.load(open("./predicates.json")))]
    plot_stat(anns_train, objs, preds, "train")
    plot_stat(anns_test, objs, preds, "test")
    plt.show()
