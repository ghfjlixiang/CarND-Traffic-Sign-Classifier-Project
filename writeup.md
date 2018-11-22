# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/Vehicles over 3.5 metric tons prohibited.png "Traffic Sign 1"
[image5]: ./test_images/No vehicles.png "Traffic Sign 2"
[image6]: ./test_images/Speed limit (120kph).png "Traffic Sign 3"
[image7]: ./test_images/Road work.png "Traffic Sign 4"
[image8]: ./test_images/Priority road.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ghfjlixiang/udacity-CarND-Traffic-Sign-Classifier-Project/)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of images distributing among different categories
![Classes vs Quantities Distribution Of Training Set][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is not the main characteristic that determines the type of traffic signs,so it would be easier for my classifier to learn.

Here is an example of a traffic sign image before and after grayscaling.

![color vs gray][image2]

As a last step, I normalized the image data because it helps improve the consistency of the model in image processing.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 graycale and normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout     	        | 0.5                                       	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Dropout     	        | 0.5                                       	|
| Fully connected		| outputs 120       							|
| Fully connected		| outputs 84       						        |
| Fully connected		| outputs 43       						        |
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used optimizer that implements the Adam algorithm  to minimize the loss function similarly to what stochastic gradient descent does.The Adam algorithm is a little more sophisticated than stochastic gradient descent,then we run the minimize function on the optimizer which uses backpropagation to update the network and minimize our training loss.First, I tune the the learning rate which is a hyperparameter that tells TensorFlow how quickly to update the network's weights.The experience of my predecessors tells me 0.001 is a good default value,Later we will use this EPOCHS variable,to tell TensorFlow how many times to run our training data through the network.In general, the more EPOCHS,the better our model will train,but also the longer training will take.Later we'll also use the batch size variable,to tell TensorFlow, how many training images to run through the network at a time.The larger the batch size ,the faster our model will train,so it's worth noting that our processor may have a memory limit on how large a batch it can run.here I chose 100 training epochs and 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.938 
* test set accuracy of 0.929

Here are some iterative approaches and strategies that I choose to Improve my classification model:
* experiment with different network architectures like changing the filter size from 3x3x16 to 5x5x32
* add regularization features like drop out to make sure the network doesn't overfit the training data
* improve the data pre-processing with steps like normalization and setting a zero mean

As you know, All my work are base on a well known architecture called LeNet, why I believe it would be relevant to the traffic sign application because character recognition has similar data features and classification ideas

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Vehicles over 3.5 metric tons prohibited][image4]
![No vehicles][image5]
![Speed limit (120kph)][image6]
![Road work][image7]
![Priority road][image8]

The first image might be difficult to classify because it's features make up parts of other images, so it's easy to misjudge.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited | 16 Vehicles over 3.5 metric tons prohibited |
| No vehicles      		| 15 No vehicles    							| 
| Speed limit (120kph)  | 8 Speed limit (120kph)					 	|
| Road work			    | 25 Road work   								|
| Priority road     	| 12 Priority road 								|
 
The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is pretty sure that this is a "Vehicles over 3.5 metric tons prohibited" sign (probability of 0.958), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9585         			| Vehicles over 3.5 metric tons prohibited  | 
| .0343     				| Yield 									|
| .0025  					| Priority road								|
| .0018 	      			| No passing					 			|
| .0005 				    | Bumpy road      							|

The prediction of the other 4 images is more certain,The code for making predictions on my final model is located in the "Output Top 5 Softmax Probabilities For Each Image Found on the Web" section of the Ipython notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


