# AdaBelief-Matlab

[![DOI](https://zenodo.org/badge/308234256.svg)](https://zenodo.org/badge/latestdoi/308234256)

The original implementation of AdaBelief is in:

https://juntang-zhuang.github.io/adabelief/

This is my matlab implementation of the AdaBelief which is much simpler than the python version. Meanwhile, a toy DNN model along with real-world example are includeded. Noted that makeDataTensor.m, extractNoise.m, countmember.m are data processing modules solely for the real-world example.

Remember to download dataset from the following urls if you want to run the AdaBeliefOnRealData.m

Direct Link:
https://drive.google.com/uc?export=download&id=1N-eImoAA3QFPu3cBJd-1WIUH0cqw2RoT

Or:
https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot

We compare the performance of AdaBelief and SGD with momentum using real-world signal recognition dataset from:
https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot

We set learning rate of AdaBelief to 0.001 with other configuration identical to their paper; We set SGD with momentum with learning rate 0.001, initial momentum 0.9 velocity 0. And discover that:
a) Using default hyperparams, AdaBelief is not guaranteed to converge to as good solutions as what SGD with momentum can reach. Indeed AdaBelief converges slightly faster than SGD at early training stages. If hyperparms are tuned, AdaBelief could have a high probability to converge faster and approach the optimal convergence where SGDM can do.

b) Our Zero-bias dense layer enabled DNN seems to have even better compatibility with the AdaBelief. 
More details of the zero-bias dense layer can be found at:
https://arxiv.org/abs/2009.02267


Update on Oct-31-2020

After a discussion with Juntang-Zhang at https://github.com/juntang-zhuang/Adabelief-Optimizer/issues/22 We realize that the Hyperparams of AdaBelief should be different from SGDM. Therefore, we posted an implementation, AdaBeliefOnRealDataOptimizedParam.m using optimized parameters specifically for AdaBelief.

Updates on Oct-30-2020:

I implemented the decoupled weight decay, but it doesn't play the Black Magic. Comparably, in SGD, I also disabled the L2 regularization, I think it will be fair to let the two models run freely.

I implemented the bias correction, indeed it helps in the regular CNN and DNN, but it does not work in my own DNN with a zero-bias (Zb) dense layer. Specifically, Zb is a differentiable template matching layer using cosine similarity to increase model explainability.

In these figures, dense represents regular DNN with dense layer, z.b. represents the DNN with the zero-bias dense layer, B.C. represents bias correction, and, H.P. denotes Hyperparams.

![alt text](https://github.com/pcwhy/AdaBelief-Matlab/blob/main/TrainingSGDADAMFC.png)
![alt text](https://github.com/pcwhy/AdaBelief-Matlab/blob/main/TrainingSGDADAMZb.png)

