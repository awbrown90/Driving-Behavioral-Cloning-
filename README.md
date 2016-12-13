# Driving-Behavioral-Cloning-

To watch the car drive its self in action check out the video!
https://youtu.be/Wp2iw8U2pZU

The files for this project include.
model.py
drive.py
data_prep.py
saved weights.zip (includes)
  model.json
  model.h5
  
Udacity driving sim was not included because unsure about licensing details.


Teaching a simulated car how to drive around a track by using deep regression learning. The training data was captured from an actual user driving the virutal car around the track using a DualShock 4 controller, note a keyboard controller could have been used as well.

In this project, a car was taught how to drive its self on two virutal tracks by recording driving data in the form of images with steering wheel angle values. The input images were fed into a convolutional network that would output a single floating value repressenting its predicted steering wheel value. The network was pretty stright forward and consisted of 3 convolutional layers, the first layer 8x8 filter size and 1 stride, the second and third with 5x5 filters and stride 2. The next layer was a single fully contected layer of size 512 that finally connected to a single neuron representing the streeing wheel output. Drop out was also used intermidently between the fully connected layers input and output to avoid overfitting. One point of interest was the networks activation function for non-linearity which was ELU instead of the fimilar RELU. ELU is very similar to RELU but always negative values and is derivationable unlike the RELU which is to help with the vanishing gradient problem during backpropagation. 

The simulator captured 160x320 size 3 channel images which were first scaled down by a factor of 4 to 40x80 so the input size of the network was 40x80x3. The color channel ended up being very helpful during certain sections of the track, noteable the first track which has a split between staying on the road or going down a dirt path. With the color information its very easy for network to see the gray vs brown difference in the color and output a correct result. There were 3 sections of the first track that were quite difficult to get good test results which included the bridge, the previous mentioned split, and the very sharp right turn infront of the water soon after the split. To get good test results the driving training data consisted of both regular center road driving and clips that had the car starting to go off the center and showing how to correct back to the center. Capturing good training data proved to be very important in order to get good testing results. Another practice which was used was to use the cars caputred left and right camera images and augment the steering wheel values a small amount with an offset, generally right camera's angle was always decreased and left's camera angle was always increased, and this provided even more training data to use. Using an actual game controller also made it alot easier to drive and record steering corrections as well. 

To avoid overfitting 10% of the training data was reserved for validation to see how the model would perform on unseen data. This was actually kind of difficult because too large of a validation set meant maybe the network was not seeing key important clips most importantly on the previous three difficult sections on track 1. An Adam Optimizer was used with SGD as the loss function during training, the final values for the loss function were around .024 for both the training and validation set with about 31,500 training examples and 350 validation. To test the model accuracy even further the second track was tested next. At first the results from track 1 did not generalize over to track 2 at all, this made since because they are completley different environments. However it was actually very easy to train the car to drive all the way through track 2 after minimal training with its already configured weights. Although track 2 had some interesting bends and hills for the most part its roads all looked the same so thats why it was probably easier for the network to train on. With track 2's steep hills though, the drive.py file needed to be edited to have a stronger throttle when the speed fell below 20mph or else the car wasnt able to make it up some of the hills. 

Instead of using a python generator it was possible to store training data in pickle files. the data_prep.py file was used to take trained images and labels and combine them into a pickle file that model.py could use. The training pickle files did become quite large, over 1 GB so resizing images down by a factor of 4 was very helpful for succesful pickle generation. 

The end result was sucessfully having the car drive in automous mode safely all the way through both track 1 and track 2. The car never ventured off the road although on track 1 sometimes it was kind of swervy and in one case made some tight turns nearowily escaping driving off the road. 
