#**Behavioral Cloning** 

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report
Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

### Model Architecture and Training Strategy

My model architecture is as follows

![Architecture](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/layers.png)

The model includes 
* Three convolution layers (64, 32 and 16 filters and 3x3 kernels) with RELU activation function.
* One max pooling layer with 2,2 pool size.
* Three fully connected layers with RELU activation function.
* Three Dropout layers - after several iterations I settled with 0.2, 0.4 and 0.4 to reduce the overfitting.
* Epochs used = 8.
* The model used an Adam optimizer, so the learning rate was not tuned manually .
* Total params used in training 329207.


#### Training the data

Training data was chosen to keep the vehicle driving on the road. 
* I used only center lane driving in my training. 
* I resized the images to 30x60 pixels to speed up the training (iterations as well) and to reduce the size of resulting model.h5 file.
* I generated the recovering from the left and right sides of the road wherever the car started to drift from the center and crashed. Below are some snapshots of the areas on the track where the car crashed. I did not add any extra recovery data. 
* I ended up using images in total for training and validation. 
* I found that the data with 0, 0, 0 angles did not make any difference in training. Hence to speed up my training, I removed all the images with 0, 0, 0 from the training set. 
![recovery1](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/recovery6.png)
![recovery2](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/recovery2.png)
![recovery3](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/recovery5.png)
![recovery4](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/recovery4.png)


