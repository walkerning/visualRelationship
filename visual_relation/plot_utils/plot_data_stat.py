# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json

import pandas as pd
from matplotlib import pyplot as plt
from visual_relation.evaluate_utils import parse_annotation

# random guess recall@100 100/(100*100*7): 0.0001428571429
# frequency guess recall@100. avergage 18 objects....
def cal_guess_recall(anns, top_preds):
    recall50_dct = {}
    recall100_dct = {}
    time50_dct = {}
    time100_dct = {}
    for img_fname, ann in anns.iteritems():
        objects, _, rel_pred_set =  parse_annotation(ann)
        rel_pred_lst = np.array([r[0] for r in rel_pred_set], dtype=int)
        if len(objects) <= 1:
            continue
        num_every = len(objects) * (len(objects) - 1)
        num_whole = int(50 / num_every)
        num_residual = 50 - num_whole * num_every
        recall50 = 0
        for pred in top_preds[:num_whole]:
            recall50 += len(np.where(rel_pred_lst == pred)[0])
        recall50 += len(np.where(rel_pred_lst == top_preds[num_whole])[0]) * float(num_residual) / num_every
        recall50_dct[img_fname] = float(recall50) / len(rel_pred_set)
        time50_dct[img_fname] = recall50

        num_whole = int(100 / num_every)
        num_residual = 100 - num_whole * num_every
        recall100 = 0
        for pred in top_preds[:num_whole]:
            recall100 += len(np.where(rel_pred_lst == pred)[0])
        recall100 += len(np.where(rel_pred_lst == top_preds[num_whole])[0]) * float(num_residual) / num_every
        recall100_dct[img_fname] = float(recall100) / len(rel_pred_set)
        time100_dct[img_fname] = recall100
    return recall50_dct, recall100_dct, time50_dct, time100_dct

def plot_stat(anns, objs, preds, phase):
    ann_num_lst = np.zeros((70,), dtype=int)
    has_ann_num_lst = np.zeros((70,), dtype=int)
    obj_num_lst = np.zeros((100,), dtype=int)
    legal_num = 0
    for img_fname, ann in anns.iteritems():
        if len(ann) == 0:
            continue
        oc_inds = np.zeros((70,), dtype=int)
        for an in ann:
            ann_num_lst[an["predicate"]] += 1
            oc_inds[an["predicate"]] = 1
            # 这里不管重复的
            obj_num_lst[an["object"]["category"]] += 1
            obj_num_lst[an["subject"]["category"]] += 1
        has_ann_num_lst = has_ann_num_lst + oc_inds

    ann_num_lst = np.transpose([ann_num_lst])
    has_ann_num_lst = np.transpose([has_ann_num_lst])
    obj_num_lst = np.transpose([obj_num_lst])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("{}-#predicates".format(phase))
    pd.DataFrame(ann_num_lst[:35], index=preds[:35]).plot.bar(ax=axes[0])
    pd.DataFrame(ann_num_lst[35:], index=preds[35:]).plot.bar(ax=axes[1])
    plt.subplots_adjust(hspace=0.8)

    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 10))
    fig1.suptitle("{}-#pic that has predicates".format(phase))
    pd.DataFrame(has_ann_num_lst[:35], index=preds[:35]).plot.bar(ax=axes1[0])
    pd.DataFrame(has_ann_num_lst[35:], index=preds[35:]).plot.bar(ax=axes1[1])
    plt.subplots_adjust(hspace=0.8)

    fig2, axes2 = plt.subplots(2, 2, figsize=(10,10))
    fig2.suptitle("{}-objects".format(phase))
    axes2 = axes2.flatten()
    pd.DataFrame(obj_num_lst[:25], index=objs[:25]).plot.bar(ax=axes2[0])
    pd.DataFrame(obj_num_lst[25:50], index=objs[25:50]).plot.bar(ax=axes2[1])
    pd.DataFrame(obj_num_lst[50:75], index=objs[50:75]).plot.bar(ax=axes2[2])
    pd.DataFrame(obj_num_lst[75:100], index=objs[75:100]).plot.bar(ax=axes2[3])
    plt.subplots_adjust(hspace=0.8)

    return np.squeeze(obj_num_lst), np.squeeze(ann_num_lst), np.squeeze(has_ann_num_lst)

if __name__ == "__main__":
    # plot train
    anns_train = json.load(open("./annotations_train.json"))
    # plot test
    anns_test = json.load(open("./annotations_test.json"))
    objs = ["{}.{}".format(x[1], x[0]) for x in enumerate(json.load(open("./objects.json")))]
    preds = ["{}.{}".format(x[1], x[0]) for x in enumerate(json.load(open("./predicates.json")))]
    obj_num_lst, ann_num_lst, has_ann_num_lst = plot_stat(anns_train, objs, preds, "train")
    train_total_ann_num = np.sum(ann_num_lst)
    print("predicates\n-------")
    print("total train predicates: ", train_total_ann_num, "\n")
    print("\n".join(["{:20s} {:<8d} {:.6f}".format(name, oc, ratio) for name, oc, ratio in zip(preds, ann_num_lst, ann_num_lst/np.sum(ann_num_lst))]))
    print("#pic that have predciates\n-----------")
    print("\n".join(["{:20s} {:<8d}".format(name, oc) for name, oc in zip(preds, has_ann_num_lst)]))
    print("objects\n-------")
    print("\n".join(["{:20s} {:<8d} {:.6f}".format(name, oc, ratio) for name, oc, ratio in zip(objs, obj_num_lst, obj_num_lst/np.sum(obj_num_lst))]))
    test_obj_num_lst, test_ann_num_lst, test_has_ann_num_lst = plot_stat(anns_test, objs, preds, "test")
    test_total_ann_num = np.sum(test_ann_num_lst)
    print("total test predicates: ", test_total_ann_num, "\n")

    # calculate if we're just guess according to the occuring frequency, what recall will we get
    recall50_dct, recall100_dct, time50_dct, time100_dct = cal_guess_recall(anns_train, np.arange(70)[np.argsort(ann_num_lst)[::-1]])
    print("\nGuess train recall@50:{}\nguess train recall@100: {}".format(np.mean(recall50_dct.values()), np.mean(recall100_dct.values())))
    print("\nGuess train recall_time@50:{}\nguess train recall_time@100: {}".format(np.sum(time50_dct.values()) / train_total_ann_num, np.sum(time100_dct.values()) / train_total_ann_num))
    print("----")
    recall50_dct, recall100_dct, time50_dct, time100_dct = cal_guess_recall(anns_test, np.arange(70)[np.argsort(ann_num_lst)[::-1]])
    print("\nGuess test recall@50:{}\nguess test recall@100: {}".format(np.mean(recall50_dct.values()), np.mean(recall100_dct.values())))
    print("\nGuess test recall_time@50:{}\nguess test recall_time@100: {}".format(np.sum(time50_dct.values()) / test_total_ann_num, np.sum(time100_dct.values()) / test_total_ann_num))

    plt.show()
