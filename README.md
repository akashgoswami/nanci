# NeurAl Network C Implementation

Inspired by Karpathy's micrograd and other excellent tutorials, this is my own attempt to understand deep learning without getting lost in mathematic implementation details. 

# Audience: 
This is primarily intended for people who are getting started with Neural network but have strong background in System programming. 


# Getting started

There are two examples: 

## Simple
simple.c explains a basic NN implementation where the network learns the model weight. 

`gcc -o simple simple.c && ./simple`



## Advanced

In this tutorial we will attempt to recognize MNIST database of handwritten digits by training a local NN in C. 

[Overview](https://raw.githubusercontent.com/akashgoswami/nanci/refs/heads/main/nn-flow-diagram-clear.svg)

```
pip3 install tensorflow numpy
#downnload dataset
python3 mnist.py

gcc -o advance advance.c
./advance

```

advanced.c is a more complex implementation where the network trains on MNIST hand written number images and tries to predict the right number by looking at the image. After sufficient training, results are more than 95% accurate. 

