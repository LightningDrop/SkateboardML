# SkateboardML
Classifying skateboarding tricks


16 June TODO:

- Gather video training data of tricks by searching for Creative Commons license in YouTube.
    We'll also need to maintain the links to all videos so we can properly credit them.
- Edit videos into 1 to 2 second clips of just the trick.
    Our goal is to have 100 clips of each trick.
    We should start by picking just two tricks- ollie and kickflip?
- (Maybe) start processing videos into homogeneous form: 30 frames / second at a specific length and resolution.
- Start reading up on background for video machine learning- is 100 clips of training data enough?

23 JUNE TODO:
- (Maybe) start processing videos into homogeneous form: 30 frames / second at a specific length and resolution.
- Finish gathering video training data
- ~Email Justin~
- Reading about background for video machine learning 


## General approach

Our goal is to come up with a reasonable classifier of tricks for 1-2 second video clips.
We plan to use whatever the most convenient and capable approaches to make that successful.

Our plan for preprocessing the data is to sample the videos down to a consistent number of frames and resolution.
For example, each video can consist of 30 frames, each of which is a 480 x 360 (360p) image, so that every video becomes a 3d array (or tensor) with dimension 480 x 360 x 30.
These are the inputs to the model.
Color is not important for trick classification, so we can also transform to black and white.

The general idea is to build a convolutional neural network (CNN), roughly following the approach of image classification.


## Questions

A CNN for images combines spatially local information from nearby pixels using convolution filters.
Should we be doing the same with pixels that are close together in time?
Google's tutorial uses a [2d convolution](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).
Is there any reason we can't use a 3d convolution?
Sure, tensorflow has one.
And there are [many popular academic papers](https://scholar.google.com/scholar?q=3d+convolution+neural+network+video&hl=en&as_sdt=0&as_vis=1&oi=scholart) doing exactly this.
Hopefully we can build on these ideas.


## Background reading

- [Convolutional Neural Network Tutorial](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks) from Google.
    Explains how a convolutional filter works, and how to design the network, in practical and explicit terms.
- [Google paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) describing training on large video dataset.
    We need to figure out how to do more with less- using only 100 or so videos of each class.
- [3D Convolutional Neural Networks for Human Action Recognition](https://icml.cc/Conferences/2010/papers/100.pdf)
    I wonder how state of the art this is.
