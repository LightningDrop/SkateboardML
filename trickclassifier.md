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
    
To start with this program you need to import the following modules from python. You will use tensorflow to build the model with its keras wrapper class. You will use the os module to find some of the video paths. You use cv2 to read in the images for processing. You will finally use tqdm to make a count the number of iterations the loop makes it through. We will also use LabelBinarizer to let us select which trick it is.

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
    
Starting from the dataset variable we call the tf.data.Dataset.from_generator function to generate the frames that are to be featured by the CNN. If we exam the frame_generator function, when we first arrive in the function we first start by assigning the VIDEOS_PATH variable to a different variable using tf.io.gfile.glob() function then shuffle that list so that we are always choosing a different video to feature extract from. This is basically just preparing the data by converting it from bgr to rgb because opencv reads in bgr. Then we resize the image to (299,299,3) because we are going to input the images into a pretrained model that accepts images of that size. Then we preprocess the image weights so that they are changed into the range of (-1,1) to fit into the pretrained model.we do this as a max of 40 times per image. If there is a video of 160 frames it'll read every 4th frame. Then we prepare the data in the dataset variables with expected output size and the string it is supposed to come with.

```python
inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

x = inception_v3.output

# We add Average Pooling to transform the feature map from
# 8 * 8 * 2048 to 1 x 2048, as we don't need spatial information
pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)
```

Then we setup the feature extraction model. In the first line we call the pretrained model object from tensorflow. We then get the output size and assign it to an arbitrary value to be flattened because we do not need spatial information. So we flatten the array using the Global average pooling function. We then finish setting up the models input and expected output in the final line.

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

Then we run this loop. What this loop does is that it iterates through all the generated frames and paths and so this loop extract key features from these list of paths of 40 and images then it stores each images features are stored in a 1x2048 array there are 40 images per video so your basically left with a 40x2048 npy file.

```python
LABELS = ['Ollie','Kickflip'] 
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

We create our labels so that the Neural net can choose between the 2 tricks. Then we create a new variable called encoder so that we can binarize the 2 options we have and we then use the fit function to fit the encoder to the tricks or classes we want to be classified. We then move on to defining our model. The model that we define basically takes our input and gives us our output. We are building our model with the mind that we have one input and one output so we use the Sequential object in keras to reflect that. We then define the layers of our sequential model. Each layer does something different for our model. The Masking layer ignores padding so that it makes the learning process more efficient because all elements of our array will not always be filled with unique values. The next layer we define is our LSTM layer which is a derivative of a Recurrent Neural Network. A RNN is… The arguments that you can provide are numerous but the ones that we use for our LSTM application are the units, the dropout, and the recurrent dropout. We set the units of neurons to 512. I choose this number because it is smaller than 2048 which is the length of the array. If we increased the number of neurons then we would make the network more powerful but then we would also increase train time to some degree. The dropout argument we use is probabilistically excluded from activation and weight updates while training a network. This has the effect of reducing overfitting and improving model performance. Recurrent dropout masks (or "drops") the connections between the recurrent units. We then go on to define our dense layer. The dense layer defines how many neurons we have for our application so we have 256 neurons available. We also provide an activation argument with it. We then move on to our own independent Dropout layer which we just use as to not overfit the data. We then move on to our last dense layer Which is just set to the length of our labels variable so 2 because we have 2 different options for probabilistic output.

```python
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
```

We then go on to use the compile function. The compile function is used to configure your model with losses and other metrics. The loss function calculates a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem. We choose this loss function specifically because we are doing multiclass classification. Then we set our optimizer which is basically our learning rate. Our learning rate will determine how long it takes our machine learning algorithm to converge in accuracy between the test dataset and the training set. Then finally we have our metrics which just gives us how often the prediction matches its label and how often the predictions are in the top k of categories.
    

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

In these lines of code we have seemingly a lot going on. We have a generator inside a function. We also have two variables being declared and assigned however, this is basically just like the same code above except slightly different. In this instance we are just padding the 40,2048 array with zeros where needed it also transforms the label. From then on everything else is the same from the feature extraction phase.

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
model.fit(train_dataset, epochs=17, callbacks=[tensorboard_callback], validation_data=valid_dataset)
```
    
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more. Then we are just left with the last model fit function. Which just starts the training phase.

# Interpreting the data

After we run all the code above there is a lot that we can do and show after compiling all the data we can use model.predict(valid_dataset) to get a view of the predicted outcomes of the test_list. When run we are met with a 2d numpy array with all of our predicted values which can be stored to be then plotted. For example we can store the model.predict(valid_dataset) outcome in some variable then import matplotlib.pyplot as plt. With plt we can plot the predictions. An example would be:

```python
import matplotlib.pyplot as plt

predict = model.predict(valid_dataset)
Kickflip = predict[:,0]
x = np.arange(len(Kickflip))
plt.scatter(x, Kickflip)
```

Which would yield a thing like this scatter plot that looks like below for kickflips

![kickflip graph](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/kickflipplot.png)

and for ollies the same code but moving the y value over one would yield this graph for ollies

![Ollie graph](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/ollieplot.png)

If we look at the data points we can see that the data is really sporadic. If we count the ollie plot only 13 plot points are above 0.8 probability so let's find out why the model is so certain on these points and so uncertain on other points.

If we just look at the kickflip plot points we can see how clear the video is that is being observed. A good example of low quality data leading to low outcomes is something like this:

![Bad Kickflip](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/ezgif.com-video-to-gif.gif)

This gif had a 65% probability of being a kickflip. If we look at the video we can sort of see why this video just shows the skateboarders feet and the camera is actually out of focus but, if we look at a better example like this one which has a way higher probability of classifying as a kickflip:

![Good Kickflip](https://raw.githubusercontent.com/LightningDrop/SkateboardML/master/images/GoodFlip.gif)

this gif is more clear you see the whole view of the skater and you see the board clearly so the probability of it placing higher is way better in theory. Although if you improve the data more and more the better the outcome will be.


# Sources

https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/#:~:text=Dropout%20is%20a%20regularization%20method,overfitting%20and%20improving%20model%20performance. 

https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout

https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

https://www.tensorflow.org/tensorboard/get_started
