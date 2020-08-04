# Machine Learning Skateboarding Tricks

Skateboarders can easily recognize tricks performed by other skateboarders.
Our goal in this project is to teach the computer to recognize skateboarding tricks.
Given a video of a skateboard trick, can the computer classify the trick with high probability?
We developed a dataset and a machine learning model that can distinguish between two of the most common skateboarding tricks, ollies and kickflips.

We started by gathering over 200 short (1 to 2 second) videos of kickflips and ollies.
We adapted the approach described in [Hands on Computer Vision with Tensorflow](https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter08/ch8_nb1_action_recognition.ipynb) to our new data set.
The data starts as a video, then passes through a convolutional neural network (CNN), then through a long short term memory (LSTM) model, and finally to an output of probabilities that the video depicts each class of tricks.

For example, here Stephen performs a kickflip:

 ![Markdown Monster](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/GoodFlip.gif)

The model predicts this is a kickflip with probability 0.9, so it works well.
We found that clear videos where the skateboarder's full body is in view did better than videos of only the skateboarder's legs.
We also found that the model was able to correctly predict a kickflip down a stair set, even though all the training data was of tricks on flat ground.
This shows that the model was able to generalize to a new and different situation.

To take this work further, we can add more tricks beyond kickflips and ollies, as well as combinations of tricks, for example, a backside 180 kickflip.
With more training data and more model tweaking this model could get good enough to rival a human skateboarder.


## Details

Classifying skateboarding tricks

This project in the the simplest words classifies skateboard tricks. We are trying to use machine learning 
to classify each skateboarding trick. There are a lot of skateboarding tricks we chose two to classify inbetween.
We chose between Ollie and Kickflip.

 
So that algorithm would take an input like above and spit out a probability of how sure it is that it is a ollie or a kickflip.



16 June TODO:

- ~Gather video training data of tricks by searching for Creative Commons license in YouTube.~
    We'll also need to maintain the links to all videos so we can properly credit them.
- ~Edit videos into 1 to 2 second clips of just the trick.~
    ~Our goal is to have 100 clips of each trick.~
    ~We should start by picking just two tricks- ollie and kickflip?~
- (Maybe) start processing videos into homogeneous form: 30 frames / second at a specific length and resolution.
- ~Start reading up on background for video machine learning- is 100 clips of training data enough?~

23 JUNE TODO:
- (Maybe) start processing videos into homogeneous form: 30 frames / second at a specific length and resolution.
- ~Finish gathering video training data~
- ~Email Justin~
- ~Reading about background for video machine learning~

30 JUNE TODO:
- ~Summarize UCF101 approach~
- ~Try to run UCF101 code~
- FUN: make a project introduction video

7 July:
- ~Adapt UCF101 code to our data, and train LSTM~

15 July:

- ~Written summary describing exactly what our code does and what our results are. Google "Literate programming"~
- ~Run some individual videos through the trained model and interpret the results. For example, P(kickflip) = 0.8~
- ~Check if running fewer training epochs will improve validation accuracy.~
- Construct CNN and compare performance to LSTM. (this will take longer)

21 July:
- ~Randomize test train split and check accuracy~
- ~Think about how you want to present this work~
- Find out why some of the videos are misclassified

28 July:
- Write high level goals and do a 30 second video
- Package and release this dataset on an open repository (Clark)
- ~Try using kickflip down stairs to see if the trick classifies~

4 August:
- Rewrite read me with a little bit more detail 3 to 5 paragraphs

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
