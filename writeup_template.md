#**Behavioral Cloning** 

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report
Rubric Points

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md summarizing the results

####2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model architecture is as follows

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
batchnormalization_1 (BatchNorma (None, 30, 60, 3)     12          batchnormalization_input_1[0][0] 
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 28, 58, 64)    1792        batchnormalization_1[0][0]       
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 28, 58, 64)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 28, 58, 64)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 26, 56, 32)    18464       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 26, 56, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 24, 54, 16)    4624        activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 24, 54, 16)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 22, 52, 8)     1160        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 22, 52, 8)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 11, 26, 8)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2288)          0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 2288)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           292992      dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 128)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128)           0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 64)            8256        dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 64)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 32)            2080        activation_6[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 32)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dendense_4 (Dense)                  (None, 1)             33          activation_7[0][0]               
====================================================================================================
Total params: 329,413
Trainable params: 329,407
Non-trainable params: 6
Layer (type)                     Output Shape          Param #     Connected to                     






Total params	252219
####2. Attempts to reduce overfitting in the model

The model contains three dropout layers in order to reduce overfitting (model.py lines 64, 84, 91).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used only center lane driving in my training. 
I generated the recovering from the left and right sides of the road wherever the car started to drift from the center and crashed. 
I found that the data with 0, 0, 0 angles did not make any difference in training. Hence to speed up my training, I removed all the images with 0, 0, 0 from the training set. 

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

