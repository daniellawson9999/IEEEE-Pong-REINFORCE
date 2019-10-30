# Purdue IEEE Computer Society Fall 2019 Pong Machine Learning Project
## OVERVIEW:
These are some guidelines for Purdue IEEE Computer Society Fall 2019 Project. Our project revolves around building a machine learning algorithm to play pong. This README should walk you through our programs and how you can write your own algorithm to play pong.

## Prerequisites:
This program uses Python 3.7. Also uses modules such as numpy for math functions and array processing and pygame to set up the pong environment. Before your program runs, be sure to install these modules using pip or importing them from the python shell. 

## Machine Learning:
Before we begin talking about the specifics of this program, it would probably be smart to brush up on some general machine learning theory, and some basics that happen in most algorithms. Some of the theory can be explained much better if you watch a couple quick videos rather than me trying to explain it in this README. I would recommend the following videos by 3 Blue 1 Brown on YouTube:
- [Nueral Networks] (https://youtu.be/aircAruvnKk)
- [How Nueral Networks Learn] https://youtu.be/IHZwWFHWa-w 
I think he does a good job of explaining the topics for people with little experience and providing good visuals to help grasp what is actually happening in most of these algorithms. If nothing else, it can at least help you get familiar with some of the vernacular or verbiage used in machine learning. He has videos on many other interesting topics too outside of the scope of this project, but I’d definitely recommend checking out some more of his channel too. 

## Project:
The goal of this project was to have teams create their own machine learning algorithms and compete against each other one by one in a tournament style to determine an overall winner. To do this we needed each of these teams to submit a trained algorithm to us. What this entails is a python file with the function to be called, and the set of trained weights. The function will be called each iteration of the while loop that runs the game to determine if the paddle should move up or down. The parameter to these functions is an array called info. Info contains the following:
- The RGB array of the entire screen
- The left paddle
- The right paddle
- The ball
- The reward
- A Done variable
The RGB array of the screen is pretty self-explanatory. The paddles and the ball are given so players can use their positions for potential inputs. The reward is a variable for if any points were scored or not, -1 if the left paddle scores, and 1 if the right paddle scores. The Done variable exists as a Boolean to determine if the game is still happening or not. Using these parameters, each function should return a value between 0 and 1 which will determine the up or down action of the paddle. Between 0 and 0.5 is d down and between 0.5 and 1 is up. Using this your algorithm should make a decision to move up or down each time the game runs.

## Writing your own Network:
### Deciding the input layer:
To begin writing your algorithm, you should first decide what the inputs will be for your algorithm. Whether you want to take the entire screen, use the positions of the paddle or ball, or some combination of the too, you need to start your network with some sort of input. Now although a larger input layer will contain more information, it will also take much longer to train and process. So, keep that in mind when deciding up what to make your input layer, but ultimately the decision is yours.
If you watched the YouTube videos posted above, you should know that much of machine learning is done through linear algebra. Multiplying and combining matrices are some of the key components to delivering a desired output. So, whatever your input layer is, it is advised to convert that into an Nx1 vector, in order to make this output happen. 
### Deciding the hidden layer(s):
After you have decided on your input layer and transformed it into an Nx1 matrix, the next decision is to decide how many hidden layers your network will have. For this case, we recommend using one hidden layer, as the game of pong is pretty trivial, so not too much computation should be required. Also, after deciding upon the number of hidden layers in the network, one must also decide on the number of nodes in each hidden layer. It can be more difficult to decide on this number, but it should be smaller than the layer prior. The hidden layer must also be a one-dimensional vector, with a size of Mx1. Ultimately, the goal is to produce one value, the output, a value between 0 and one.
### Initializing Weight Matrices:
Now to get the Nx1 input layer to a 1x1 output will require a lot of linear algebra and initialization of weight matrices. Your weight matrix, when multiplied by the input layer, will allow you to reach the next layer in your network. So, if the input layer is Nx1 and the hidden layer is Mx1, your weigh matrix should have a size of MxN, and consist of random values between 0 and 1. For more information on how each weight corresponds to each of the input and output layers, please check out the above YouTube videos. But essentially, you have created the first part of an untrained model. I say untrained because these weights are random, the model is completely guessing whether to go up or down and will likely be very bad at pong. The “learning” that happens will occur when you update these weights to be real, meaningful values that will allow your program to make good decisions based on the input. We will talk about updating those weights later. But after creating a weight matrix for the first layer, you must also make one for the hidden layer. With the hidden layer being an Mx1 Vector, this mean the dimensions of the second weight matrix will be 1xM, in order to reach the desired 1x1 output, when multiplied together. Like the prior matrix, this will also be initially random, and then trained later. For now, this is a good start for traversing your way through your network.
### Activating the outputs:
Activation functions are also very important. Each value of nodes in the hidden layer and the overall output must be between 0 and 1. Activation functions allow us to very easily computer these values. There are many different types of activation functions and they will produce different results in your program. We will not go into each one and the pros and cons of them, but the ones we used are the sigmoid and relu functions. I would recommend doing some research and finding the best activation function for your model.
### Compute the results:
After activating the values at each layer, multiplying them by the weight matrices, and repeating the process all the way down to the output, you should have a value between 0 and 1. Values closer to 0 mean that your program is very confident that it should move down. Values close to 1 mean that the program is very confident that it should move up. Values in the middle, around 0.5, mean that your program isn’t entirely sure what it should do. Either way, this is the output to be returned telling your program what to do.
### Training the model:
The most important part of this program is training the model. Otherwise you would just be stuck with a random weight matrix and your model guessing what to do every time. This is also the part of the program that allows for a lot of customization and individuality. There are lots of different ways to do this, whether using back propagation, a genetic algorithm, or something else, the point is the program should determine whether the decision it made was a good or bad one and update the weights accordingly, working its way back through the network. Due to the vastness of different models, we aren’t going to go into each one, but many can be made to train the network well. 
### For Competition:
For the competition you aren’t going to want to initialize a random matrix, that’d ruin all the training your model did leading up to it. You’ll want to load a trained matrix into your function to use. This also implies having a way to save weights after your model has gone through some training. The training function itself is not necessary to be called during the competition, since your model should be down training at that point.

## Final Thoughts:
There are a lot of different ways to write this code to train an algorithm for Machine Learning filled with many different strategies to employ and try. We thought pong was a good game for an introduction to machine learning as it is a very simple game to understand, while many of the important concepts of machine learning can still be applied to it. We hope this was a good way for people to get their feet wet into the machine learning world, as it will likely be a very prevalent concept in the future.
## Credits:
This README was written by Erik Wilson. The code for the files was written mostly by Jerome Schweitzer and Erik Wilson, with help from resources found online. A big shout out is due for 3Blue1Brown on YouTube for posting great tutorial videos as well. If you have any questions, find any bugs in the code, or errors in the README, please reach out. We are always looking for ways to improve. Thank you for reading this far :)
