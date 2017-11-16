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

![Architecture](https://github.com/pssdc/CarND-Behavioral-Cloning-P3/blob/master/model_layers.png)

The model includes 



Total params	252219
####2. Attempts to reduce overfitting in the model

The model contains three dropout layers in order to reduce overfitting (model.py lines 64, 84, 91).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
* I used only center lane driving in my training. 
* I generated the recovering from the left and right sides of the road wherever the car started to drift from the center and crashed. 
* I found that the data with 0, 0, 0 angles did not make any difference in training. Hence to speed up my training, I removed all the images with 0, 0, 0 from the training set. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started the project with the code provided in the lesson. 
The Keras network I used was also similar to the one that is provided in Keras lesson.  
Removed all the data with 0,0,0 angles in driving_log.csv file.
Imported and split the data into training (80%) and validation set (20%).
Randomly shuffled the data to generalize the training.
Used generator (as explained in the lesson) to efficiently used the RAM. 
Used only the center image for training. Resized it to 80, 160 pixels.
Used BatchNormalization to normalize the images. 
Added the recovery images wherever the car drifted from the center and crashed. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

