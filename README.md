# AdaBelief-Matlab

[![DOI](https://zenodo.org/badge/308234256.svg)](https://zenodo.org/badge/latestdoi/308234256)

The original implementation of AdaBelief is in:

https://juntang-zhuang.github.io/adabelief/

This is my matlab implementation of the AdaBelief which is much simpler and clear than the nasty python version. Meanwhile, a toy DNN along with some other optimizers are includeded.

We compare the performance of AdaBelief and SGD with momentum using real-world signal recognition dataset from:
https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot

We set learning rate of AdaBelief to 0.001 with other configuration identical to their paper; We set SGD with momentum with learning rate 0.001, initial momentum 0.9 velocity 0. And discover that:
a) The AdaBelief is not guaranteed to converge to as good solutions as what SGD with momentum can reach. Indeed AdaBelief converges slightly faster than SGD at early training stages.

b) Our Zero-bias dense layer enabled DNN seems to have even better compatibility with the AdaBelief. 
More details of the zero-bias dense layer can be found at:
https://arxiv.org/abs/2009.02267

![alt text](https://github.com/pcwhy/AdaBelief-Matlab/blob/main/TrainingSGDADAMZb.png?raw=true)

![alt text](https://github.com/pcwhy/AdaBelief-Matlab/blob/main/TrainingSGDADAMFC.png?raw=true)


