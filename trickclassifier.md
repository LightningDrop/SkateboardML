# Skateboarding Trick Classifier

```python
import tensorflow as tf;
import os;
import cv2;
import numpy as np;
import tqdm;
from sklearn.preprocessing import LabelBinarizer;

BASE_PATH = 'D:/SkateboardML/Tricks'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mov')
SEQUENCE_LENGTH = 40
```
    
To start with this program we import the following modules from python. We will use `tensorflow`'s `keras` to build the model. We will use the `os` module to find some of the video paths. We use `cv2` to read in the images for processing. Finally, we will use `tqdm` to count the number of iterations the loop makes it through. We will also use `LabelBinarizer` to encode the trick.

```python
def frame_generator():
    video_paths = tf.io.gfile.glob(VIDEOS_PATH)
    np.random.shuffle(video_paths)
    for video_path in video_paths:
        
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
        current_frame = 0

        max_images = SEQUENCE_LENGTH
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame % sample_every_frame == 0:
                # OPENCV reads in BGR, tensorflow expects RGB so we invert the order
            frame = frame[:, :, ::-1]
            img = tf.image.resize(frame, (299, 299))
            img = tf.keras.applications.inception_v3.preprocess_input(
                img)
            max_images -= 1
            yield img, video_path

        if max_images == 0:
            break
        current_frame += 1

dataset = tf.data.Dataset.from_generator(frame_generator,
             output_types=(tf.float32, tf.string),
             output_shapes=((299, 299, 3), ()))

dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
```
Starting from the `dataset` variable, we call the `tf.data.Dataset.from_generator` function to generate the frames which are the input to the CNN.
If we examine the `frame_generator` function, we start by listing all the training videos in random order.
Next, we convert the data from bgr to rgb because `opencv` reads in bgr.
We resize the image to (299,299,3) because we are going to input the images into a model that accepts images of that size.
We preprocess the image weights so that they are changed into the range of (-1,1) to fit into the model.
Each video gets converted to at most `SEQUENCE_LENGTH = 40` frames.
For example, If there is a video of 160 frames this generator will yield every 4th frame.
Then we prepare the data in the dataset variables with expected output size and a string naming the trick.

```python
inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

x = inception_v3.output

# We add Average Pooling to transform the feature map from
# 8 * 8 * 2048 to 1 x 2048, as we don't need spatial information
pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)
```

Next, we setup the feature extraction model.
In the first line we call the pretrained model object from tensorflow.
We then get the output size and assign it to an arbitrary value to be flattened because we do not need spatial information.
So we flatten the array using the Global average pooling function.
We then finish setting up the models input and expected output in the final line.

```python
current_path = None
all_features = []

for img, batch_paths in tqdm.tqdm(dataset):
    batch_features = feature_extraction_model(img)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1))
    
    for features, path in zip(batch_features.numpy(), batch_paths.numpy()):
        if path != current_path and current_path is not None:
            output_path = current_path.decode().replace('.mov', '.npy')
            np.save(output_path, all_features)
            all_features = []
            
        current_path = path
        all_features.append(features)
```

Next, we iterate through all the generated frames and paths, extracting key features from the paths and images, and then storing the extracted features from each image in a 1x2048 array.
There are 40 images per video, which leaves us with a 40x2048 numpy array.

```python
LABELS = ['Ollie','Kickflip','Shuvit'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])
```

We create our labels so that the Neural net can choose between the 3 tricks. Then we create a new variable called `encoder` that maps back and forth between the text class labels that humans understand and the numeric encoding that the model understands.
We then move on to defining our model.

The model takes input and produces output, as with any machine learning model.
Each layer does something different for our model.
The `Masking` layer ignores padding so that it makes the learning process more efficient because all elements of our array will not always be filled with unique values.
The next layer we define is our LSTM layer which is a derivative of a Recurrent Neural Network (RNN).
An RNN can be complex, but the ones that we use for our LSTM application are the units, the dropout, and the recurrent dropout.
We set the units of neurons to 512 because it is smaller than 2048 which is the length of the array.
If we increased the number of neurons, then we would make the network more powerful, but would also increase training time.
The dropout argument we use is probabilistically excluded from activation and weight updates while training a network.
This has the effect of reducing overfitting and improving model performance.
Recurrent dropout masks (or "drops") the connections between the recurrent units.
We then go on to define our dense layer.
The dense layer defines how many neurons we have for our application so we have 256 neurons available.
We also provide an activation argument with it.
We then move on to our own independent Dropout layer which we just use as to not overfit the data.
Finally, we use a dense layer that outputs the probability that each input belongs to one of the originally specified classes.

```python
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
```

We then compile our model, configuring the loss function, optimizer, and other metrics.
The loss function calculates a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem.
We choose this loss function specifically because we are doing multiclass classification.
~~Then we set our optimizer which is basically our learning rate.~~
~~Our learning rate will determine how long it takes our machine learning algorithm to converge in accuracy between the test dataset and the training set.~~
Finally, we have our metrics which tell us how often the prediction matches its label and how often the predictions are in the top k of categories.
 

```python
with open('testlist02.txt') as f:
    test_list = [row.strip() for row in list(f)]

with open('trainlist02.txt') as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]
```
        
Here we are just populating our arrays with data from our test train list.

```python
def make_generator(file_list):
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            full_path = os.path.join(BASE_PATH + '/', path).replace('.mov', '.npy')

            label = os.path.basename(os.path.dirname(path))
            features = np.load(full_path)

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            transformed_label = encoder.transform([label])
            yield padded_sequence, transformed_label[0]
    return generator

train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
```

In these lines of code we have a lot going on.
We have a generator defined inside a function.
We also have two variables being declared and assigned, however, this is basically just like the same code above except slightly different.
In this instance we are just padding the 40,2048 array with zeros where needed, as well as transforming the label.
From then on everything else is the same as the feature extraction phase.

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
model.fit(train_dataset, epochs=17, callbacks=[tensorboard_callback], validation_data=valid_dataset)
```
    
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow.
It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
We are just left with the last model fit function, which trains the model.

# Interpreting the data

After we run all the code above we can examine our model.
We can use `model.predict(valid_dataset)` to get a view of the predicted outcomes of the `test_list`, which is a 2d numpy array containing all of our predicted values.
We can plot the predictions.

```python
import matplotlib.pyplot as plt

predict = model.predict(valid_dataset)
Kickflip = predict[:,0]
x = np.arange(len(Kickflip))
plt.scatter(x, Kickflip)
```

Now we can plot the predicted probabilities that each trick is a kickflip:

![kickflip graph](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/kickflipplot.png)

The predicted probabilities that each trick is an ollie, is just 1 minus the probability that the trick is a kickflip.

![Ollie graph](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/ollieplot.png)

If we look at the data points we can see that the data is not cleanly separated between ollie and kickflip.
Examining the ollie plot, only 13 plot points are above 0.8 probability, so let's find out why the model is so certain on these points and so uncertain on other points.

The clarity of the video and location of the skateboarder in the frame seem to influence how certain the model is of a prediction.
The model had trouble saying with certainty that this trick was a kickflip:

![Bad Kickflip](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/ezgif.com-video-to-gif.gif)

This gif had a 65% probability of being a kickflip. 
If we look at the video we can see why.
This video just shows the skateboarders feet and the camera is actually out of focus.
In contrast, if we look at a better example like this one which has a way higher probability of classifying as a kickflip:

![Good Kickflip](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/GoodFlip.gif)

This gif is more clear.
You see the whole view of the skater, and you see the board clearly, which allows the model to perform better.
The more data we get, and the more time we spend training the model, the more accurate we can expect our predictions to be.

# Sources

https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/#:~:text=Dropout%20is%20a%20regularization%20method,overfitting%20and%20improving%20model%20performance. 

https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout

https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

https://www.tensorflow.org/tensorboard/get_started
