# -*- coding: utf-8 -*-

from itertools import product
from collections import namedtuple, OrderedDict

import numpy as np

Object = namedtuple("Object", ["bbox", "category"])

def _add_to_dct(obj, dct):
    if obj not in dct:
        dct[obj] = len(dct)
    return dct[obj]

def get_union_box(bbox1, bbox2):
    return (min(bbox1[2], bbox2[2]),\
            min(bbox1[0], bbox2[0]),\
            max(bbox1[3], bbox2[3]),\
            max(bbox1[1], bbox2[1]))

def parse_annotation(annotation):
    """
    Parse annotation of a single image

    Returns:
        objects: list of `evalute_utils.Object`, which is all the unique objects in this image
        rel_set: positive relation set without predicate, for negative triple sample generation
        rel_pred_set: positive relation set with predicate, for recall calculation
    """
    objects = OrderedDict()
    rel_set = set()
    rel_pred_set = set()
    for rel in annotation:
        obj = _add_to_dct(Object(tuple(rel["object"]["bbox"]), rel["object"]["category"]), objects)
        sub = _add_to_dct(Object(tuple(rel["subject"]["bbox"]), rel["subject"]["category"]), objects)
        rel_set.add((obj, sub))
        rel_pred_set.add((rel["predicate"], obj, sub))
    # `objects` holds unique objects in this image
    objects = list(objects)
    return objects, rel_set, rel_pred_set

def calculate_recall(true_rel_pred_set, objects, get_pred, return_top_k=1):
    # calcualte recall@100, recall@50
    predict_rel_lst = [] # relation triple list
    predict_lst = [] # score list, one-to-one correspondance with `predict_rel_lst`

    # iterate all possible object pairs
    pairs = list(product(range(len(objects)), range(len(objects))))
    for pair in pairs:
        if pair[0] == pair[1]:
            # relation must be between two different objects
            continue
        predict_rel_lst.extend([(i, pair[0], pair[1]) for i in range(70)])
        predictions = get_pred(objects[pair[0]], objects[pair[1]])
        predict_lst.extend(predictions)

    sort_inds = np.argsort(predict_lst)[::-1]
    # recall 50
    match_50 = len(set([tuple(r) for r in np.array(predict_rel_lst)[sort_inds[:50]].tolist()]).intersection(true_rel_pred_set))
    recall_50 = float(match_50) / len(true_rel_pred_set)
    # recall 100
    match_100 = len(set([tuple(r) for r in np.array(predict_rel_lst)[sort_inds[:100]].tolist()]).intersection(true_rel_pred_set))
    recall_100 = float(match_100) / len(true_rel_pred_set)
    # the top-k prediction
    top_k_predictions = zip([tuple(r) for r in np.array(predict_rel_lst)[sort_inds[:return_top_k]]], np.array(predict_lst)[sort_inds[:return_top_k]].tolist())
    return recall_50, recall_100, match_50, match_100, top_k_predictions


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

_post_process_funcs = {
    "none": lambda x: x,
    "relu": lambda x: np.maximum(x, 0),
    "softmax": _softmax
}

def get_post_process_func(name):
    assert name in _post_process_funcs, "Legal post process funcs include: _post_process_funcs.keys()"
    return _post_process_funcs[name]
