# ResNets On Birds Classification

A trained ResNet-32 model on birds classification task. It is able to take images in different sizes and output the name
of the bird. However, only 555 types of birds are in the train sets, therefore it can only provide valid prediction if the 
bird is among them.

The project is mainly built from the work done by gahaalt. [Here](https://github.com/gahaalt/resnets-in-tensorflow2) is the original repo.

## Brief Overview
[](https://youtu.be/y3QdRKH1h6U)

## Problem Setup:
This project is mainly created to participate in a [Kaggle competition](https://www.kaggle.com/c/birds-22wi/overview) on birds classifications.
My goal is not only create a working neural network, but also I would like to tune the model as much as possible to acheive
better performance by various techniques.
## Dataset:
The dataset is given by the organizer of the competition. There are totally 38,562 labeled images as training set and 10,000 
unlabeled images as testing set. Furthermore, there are 555 kinds of birds are included in the dataset, and each class has 
different number of samples. One of the challenge is that all images has various sizes, therefore some preprocessing is
required.
## Techniques:
### Network and Hyperparameters
Actually, there are several networks that have been implemented. The largest one is ResNet-110, which should out-performed than others.
Due to the limitation on hardware, I selected ResNet-32 as the model I trained on since this is the largest model I can load into the memory. 

Below, I have listed several hyperparameters regarding the training.
```
batch_size = 32
number_of_batch_per_epoch = 1200
number_of_epoch = 200

learning_rate = [0.1, 0.01, 0.001, 0.0001]
learning_rate_boundaries = [60, 120, 160, 200]

schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay()
optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9, nesterov=nesterov)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) + L2_regularization_loss
```
One thing worth to mention is that the learning rate varies dynamically throughout the training process. During the first 60 epochs,
the learning rate is 0.1, and it decays by the factor of 10 in the next 60 epochs. Then, it decays by the factor of 10 in every 40 epochs util the end of training.

### Challenges rasied by the limition of hardware.
Here is the info of the device I used to train the model.
```
CPU: Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz   2.30 GHz
RAM: 16 GB
GPU: RTX 2060
```
16 GB memory space is not enough to hold a large network and a large detaset at the same time. Therefore, few modifications
have been done to encounter this problem.
- All images are resized to (224, 224, 3).
- Decrease the batch_size from 128 to 32.
- Smaller network (ResNet-32) is selected. One should definitely try ResNet-110 if hardware is available. 

Lastly, to speed up the training, I decided to get the first version in 100 epochs training so that I can see whether I am on the right track. 
The second version is still under training while I am writing this.

### Image Processing
Another reason to resize the image into the same size is that ResNet requires all the inputs are in the same size. Besides,
several modifications like random crops and flips are also implemented to improve the generalization of the model.
```
image = tf.cast(image, tf.float32) / 255
image = tf.image.random_flip_left_right(image) 
image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], 'REFLECT')
image = tf.image.random_crop(image, (224, 224, 3))
```

### Splitting the dataset
Since only one dataset is labelled, after batched them into size of 32 samples, I selected the first 1100 batches as the training set, and the rest 106 batches became the test set. By doing this, 
I can make sure that the performance on testing set is valid since the model hasn't seen any of the samples in the test set.

Moreover, before the separation, the dataset has been shuffled.
## Model evaluation
Here are some figures generated from logs.
### Training
![Training figures](figs/train.PNG)
We can see that learning rate decay is an effective techniques to improve the performance. The error rate drops significantly as 
the model moves along with the gradient in a smaller step. The final error rate on training set is 0.012.

### Test
![Test figures](figs/test.PNG)
The final error rate on the test set is 0.00045. Since the data is shuffled, it is unlikely to have sample distribution bias.

### Validation
The model has scored 0.561 on the real test set provided by the organizer of the competition. In other words, 56.1% of the predictions on the real test set are correct.


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