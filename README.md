# CNN Classifier
This model is a PyTorch implemented CNN classifier, fitted to Fashion MNIST dataset. The default parameters get above 90% accuracy, but there is a probability to get much better. e.g. More layers, Output dropout, Hyperparameter tuning, ...

## Quick start
```python train.py [--parameters]```

## Parameters
* batch_size: the size of mini-batches (default 256)
* hidden_size: the size of hidden state of generator, one layer before fianl layer (default 128)
* n_kernels: the number of kernels in convolution layers (default 16)
* kernel_size: the size of kernel in convolution layers (default 5)
* pool_size: the size of pooling kernel (default 5)
* dropout: the ratio of node that will be dropped randomly during training (default 0.5)
* epochs: the number of epochs (defualt 10)
* optimizer: adam / rmsprop / sgd, capital doesn't matter
* lr: the learning rate when gradient descent steps.
* verbose: print result logs every n-epochs.
