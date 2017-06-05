# -*- coding: utf-8 -*-
"""
Plot the score histogram for positive/negative samples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

pred_names = json.load(open("./predicates.json", "r"))
print("Loading score file.")
vscores = pd.read_csv(sys.argv[1], sep=" ", header=None, index_col=False)
vscores.columns = ["p_ind", "posneg", "pred", "obj", "subj", "score"]
pos_vscores = vscores[vscores["posneg"]==1]
neg_vscores = vscores[vscores["posneg"]==0]
pred_score_dct = OrderedDict()
neg_score_dct = OrderedDict()
for pred, pred_name in enumerate(pred_names):
    pred_score_dct[(pred, pred_name)] = pd.Series(pos_vscores[pos_vscores["pred"] == pred]["score"].values)
    neg_score_dct[(pred, pred_name)] = pd.Series(neg_vscores[neg_vscores["pred"] == pred]["score"].values)

pos_data = pd.DataFrame(pred_score_dct)
neg_data = pd.DataFrame(neg_score_dct)
#plt.figure(figsize=(10, 10))
# will create it own axes... will call _try_sort on columns...
# so use tuple rather than strings as columns to preserve histogram order
pos_data.hist(figsize=(15, 10))
plt.gcf().canvas.set_window_title("score histogram of positive samples")
#suptitle("score histogram of positive samples")
plt.tight_layout()

neg_data.hist(figsize=(15, 10))
plt.gcf().canvas.set_window_title("score histogram of high-score negative samples")
#suptitle("score histogram of positive samples")
plt.tight_layout()

plt.show()
