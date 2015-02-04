# RNTN.

This repository is based on research paper `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank` and Associated Stanford website `http://nlp.stanford.edu/sentiment/`.

## Numpy 

An RNTN using only Numpy objects already works on the PTB Tree Format dataset.
With no special parameter tuning (e.g. grid search optimization), with a classical Stochastic Gradient Descent,
one can obtain a classification accuracy around 64%. All files are in the Numpy folder.

## Theano

I then tried to add GPU methods in the code via Theano.
My graphics card does not support well Cuda 6.5 but you will surely have better luck running the code on your computer.

I tried the following :

* **A.** Allocating data into symbolic variables just before computing matrix operations on the GPU.

This works but very slowly - as one can expect, the cost of going there and back from the CPU to the GPU
is too high to beat a pure Numpy implementation.  It was a rather ineffective GPU implementation, but at least it runs and offer the same results as the Numpy implementation. All files are in the Theano - Old folder.

* **B.** Creating a 'real' Theano implementation, with Shared Variables for the weights/biases within the RNTN class.

I couldn't make it work properly - obviously there is a cost problem at the moment.
All files are in the Theano - Latest folder.

All these Theano codes have an argument parser to run things directly from the terminal.
I did not used the AdaGrad algorithm mentioned in the paper.
Also another path of exploration for this project, that could be done afterwards.
