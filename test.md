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

Langauge model + visual model
-------------
还没有调visual model的输出...用relu的话跟现在的结果应该不会有太大差别...top100可能有点吧...
效果比单用visual model差很多...感觉有点崩... 要调整一下language model训练了

不过已经不都只预测on了...这个从某种意义上好点......
感觉不同triple的f_score是不是差距有点偏大...
