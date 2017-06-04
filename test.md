Visual only test
-------
Use model `trainlog_visual_module3/model.ckpt-4009`
V-only train:
mean recall@50: 0.299654843031
mean recall@100: 0.442361625419

V-only test:
number actual valid examples: 954
mean recall@50: 0.30213374183
mean recall@100: 0.451477242124
top1 accuracy: 0.275681341719

现在每张图检测的top-1的predicate都是on...
V_score在单个predicate上太自信了不行......没有真正用到信息...
妈呀... dead unit了肯定....看出来了...所有图连分数都一样...把激活打印出来看看好了

v score加不加softmax没有太大区别
number actual valid examples: 3778
mean recall@50: 0.303348101647
mean recall@100: 0.448069926288
top1 accuracy: 0.236103758602
Writing recall information into v_mean_recalls_1496522322.pkl.

Langauge model + visual model
-------------
还没有调visual model的输出...用relu的话跟现在的结果应该不会有太大差别...top100可能有点吧...
效果比单用visual model差很多...感觉有点崩... 要调整一下language model训练了

不过已经不都只预测on了...这个从某种意义上好点......
感觉不同triple的f_score是不是差距有点偏大...

train: 这特么也太坑了...
number actual valid examples: 3778
mean recall@50: 0.10051915784
mean recall@100: 0.171169408576
top1 accuracy: 0.00582318687136


predicates
-------
total train predicates:  30355

on.0                 5113     0.168440
wear.1               4345     0.143140
has.2                2314     0.076231
next to.3            2366     0.077944
sleep next to.4      3        0.000099
sit next to.5        75       0.002471
stand next to.6      109      0.003591
park next.7          37       0.001219
walk next to.8       5        0.000165
above.9              2417     0.079624
behind.10            2237     0.073695
stand behind.11      8        0.000264
sit behind.12        7        0.000231
park behind.13       1        0.000033
in the front of.14   1539     0.050700
under.15             1481     0.048789
stand under.16       5        0.000165
sit under.17         3        0.000099
near.18              1372     0.045198
walk to.19           20       0.000659
walk.20              5        0.000165
walk past.21         3        0.000099
in.22                734      0.024181
below.23             762      0.025103
beside.24            754      0.024839
walk beside.25       3        0.000099
over.26              569      0.018745
hold.27              610      0.020096
by.28                417      0.013737
beneath.29           366      0.012057
with.30              222      0.007313
on the top of.31     189      0.006226
on the left of.32    381      0.012551
on the right of.33   364      0.011991
sit on.34            205      0.006753
ride.35              159      0.005238
carry.36             156      0.005139
look.37              120      0.003953
stand on.38          94       0.003097
use.39               72       0.002372
at.40                64       0.002108
attach to.41         51       0.001680
cover.42             60       0.001977
touch.43             57       0.001878
watch.44             48       0.001581
against.45           38       0.001252
inside.46            29       0.000955
adjacent to.47       27       0.000889
across.48            45       0.001482
contain.49           21       0.000692
drive.50             21       0.000692
drive on.51          9        0.000296
taller than.52       32       0.001054
eat.53               15       0.000494
park on.54           21       0.000692
lying on.55          13       0.000428
pull.56              21       0.000692
talk.57              23       0.000758
lean on.58           17       0.000560
fly.59               23       0.000758
face.60              15       0.000494
play with.61         9        0.000296
sleep on.62          9        0.000296
outside of.63        6        0.000198
rest on.64           7        0.000231
follow.65            7        0.000231
hit.66               10       0.000329
feed.67              6        0.000198
kick.68              5        0.000165
skate on.69          4        0.000132
#pic that have predciates
-----------
on.0                 2394
wear.1               1694
has.2                1334
next to.3            1349
sleep next to.4      3
sit next to.5        52
stand next to.6      85
park next.7          27
walk next to.8       4
above.9              1209
behind.10            1362
stand behind.11      8
sit behind.12        6
park behind.13       1
in the front of.14   1038
under.15             939
stand under.16       5
sit under.17         3
near.18              719
walk to.19           16
walk.20              5
walk past.21         3
in.22                524
below.23             526
beside.24            504
walk beside.25       2
over.26              372
hold.27              450
by.28                256
beneath.29           277
with.30              142
on the top of.31     167
on the left of.32    280
on the right of.33   269
sit on.34            161
ride.35              139
carry.36             132
look.37              107
stand on.38          74
use.39               63
at.40                55
attach to.41         46
cover.42             47
touch.43             51
watch.44             33
against.45           34
inside.46            27
adjacent to.47       19
across.48            39
contain.49           21
drive.50             19
drive on.51          8
taller than.52       23
eat.53               13
park on.54           20
lying on.55          13
pull.56              19
talk.57              20
lean on.58           17
fly.59               20
face.60              14
play with.61         8
sleep on.62          6
outside of.63        5
rest on.64           7
follow.65            6
hit.66               9
feed.67              5
kick.68              4
skate on.69          4
objects
-------
person.0             17616    0.290166
sky.1                3032     0.049942
building.2           2296     0.037819
truck.3              853      0.014050
bus.4                1332     0.021940
table.5              1466     0.024148
shirt.6              2156     0.035513
chair.7              731      0.012041
car.8                1584     0.026091
train.9              456      0.007511
glasses.10           933      0.015368
tree.11              997      0.016422
boat.12              520      0.008565
hat.13               1065     0.017542
trees.14             700      0.011530
grass.15             659      0.010855
pants.16             1063     0.017509
road.17              708      0.011662
motorcycle.18        639      0.010525
jacket.19            853      0.014050
monitor.20           515      0.008483
wheel.21             705      0.011613
umbrella.22          672      0.011069
plate.23             463      0.007626
bike.24              609      0.010031
clock.25             411      0.006770
bag.26               544      0.008961
shoe.27              113      0.001861
laptop.28            450      0.007412
desk.29              413      0.006803
cabinet.30           120      0.001977
counter.31           243      0.004003
bench.32             396      0.006523
shoes.33             553      0.009109
tower.34             387      0.006375
bottle.35            362      0.005963
helmet.36            426      0.007017
stove.37             97       0.001598
lamp.38              292      0.004810
coat.39              264      0.004349
bed.40               220      0.003624
dog.41               238      0.003920
mountain.42          356      0.005864
horse.43             272      0.004480
plane.44             239      0.003937
roof.45              309      0.005090
skateboard.46        389      0.006408
traffic light.47     138      0.002273
bush.48              172      0.002833
phone.49             363      0.005979
airplane.50          179      0.002948
sofa.51              263      0.004332
cup.52               266      0.004381
sink.53              171      0.002817
shelf.54             141      0.002323
box.55               246      0.004052
van.56               279      0.004596
hand.57              162      0.002668
shorts.58            366      0.006029
post.59              223      0.003673
jeans.60             392      0.006457
cat.61               194      0.003196
sunglasses.62        309      0.005090
bowl.63              142      0.002339
computer.64          121      0.001993
pillow.65            204      0.003360
pizza.66             209      0.003443
basket.67            163      0.002685
elephant.68          161      0.002652
kite.69              222      0.003657
sand.70              163      0.002685
keyboard.71          221      0.003640
plant.72             160      0.002635
can.73               90       0.001482
vase.74              113      0.001861
refrigerator.75      72       0.001186
cart.76              148      0.002438
skis.77              168      0.002767
pot.78               140      0.002306
surfboard.79         143      0.002355
paper.80             92       0.001515
mouse.81             146      0.002405
trash can.82         128      0.002108
cone.83              162      0.002668
camera.84            147      0.002421
ball.85              126      0.002075
bear.86              68       0.001120
giraffe.87           93       0.001532
tie.88               176      0.002899
luggage.89           111      0.001828
faucet.90            91       0.001499
hydrant.91           101      0.001664
snowboard.92         130      0.002141
oven.93              51       0.000840
engine.94            74       0.001219
watch.95             139      0.002290
face.96              83       0.001367
street.97            2382     0.039236
ramp.98              97       0.001598
suitcase.99          92       0.001515


只按频率猜, 相当于是在只按频率猜嘛..visual model的训练肯定也进了dead zone了

Guess train recall@50:0.307578813579
guess train recall@100: 0.445223970735

Guess train recall_time@50:0.187039138924
guess train recall_time@100: 0.307722744255
----

Guess test recall@50:0.308403207698
guess test recall@100: 0.445828507478

Guess test recall_time@50:0.19573694902
guess test recall_time@100: 0.313647699044

total test predicates:  7638


trainlog_visual_module6/model.ckpt-654 把learning rate改小了... 确实有0, 1, 9三种predicates了...不过还是由于各类数量差太多...top1也只有这三类.. 然后虽然recall@50/100增高了...但是top one accuracy降低很多诶...

```
foxfi@foxfi-eva5:~/visualRelationship$ python evaluate_visual_module.py --annotation_file annotations_train.json --checkpoint_file trainlog_visual_module6/model.ckpt-654 --cal_recall --cal_vscore --verbose --save v_scores_6.txt | tee train_recall_log_6.txt
number actual valid examples: 3778
mean recall@50: 0.402183932239
mean recall@100: 0.535688309905
recall_time@100: 0.467806542541
top1 accuracy: 0.0979354155638
Writing recall information into v_mean_recalls_1496542759.pkl.
```

GBC: 结果比visual 好看多了...就算visual那里loss到80吧...平均的ce loss也是2.5, 比这里大...
```
Parsing the annotation files...
train accuracy: 0.576544226651
train cross entropy loss: 44126.1326805; mean: 1.45366933555
test accuracy: 0.486383870123
test cross entropy loss: 14866.4305369; mean: 1.9463773942

foxfi@foxfi-eva5:~/visualRelationship$ python evaluate_gbc.py --verbose --cal_recall --cal_vscore --save gbc_v_score.txt --checkpoint_file gbc.pkl  | tee logs/gbc_train_recall_log.txt
number actual valid examples: 3778
mean recall@50: 0.682864639337
mean recall@100: 0.802319964521
recall_time@50: 0.648431125287
recall_time@100: 0.805207328833
top1 accuracy: 0.233986236104
Writing recall information into gbc_mean_recalls_1496586277.pkl.
```