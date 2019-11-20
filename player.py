import numpy as np
import os.path
import ai
from comp_exec import Game
import pygame
from enum import Enum, auto

class Mode(Enum):
    image = auto()
    coords = auto()

class Player():
    def __init__(self, training_name = None, mode = None):
        if training_name is None:
            training_name = "competition"
            
        if mode is None:
            self.mode = Mode.image
        else:
            self.mode = Mode.coords
            
        self.w1_name = training_name + "_w1.txt"
        self.w2_name = training_name + "_w2.txt"
        self.training_name = training_name
        
        if self.mode == Mode.image:
            self.input_size = 4725
            self.hidden_layer_size = 200
        else:
        # box input:
        # PaddleAx, PaddleAy, PaddleBx, PaddleBy, BallX, BallY
            self.input_size = 6
            self.hidden_layer_size = 4
        
                

        self.weights_one = []
        self.weights_two = []
        self.save_interval = 1
        # hyperparameters
        # how fast training happens. Higher rate = faster convergence, less accurate
        self.learning_rate = .01
        # how much to discount future rewards
        self.discount_rate = .99

        if os.path.exists(self.w1_name) and os.path.exists(self.w2_name):
            self.loadWeights()
        else:
            # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            # weight intialization
            self.weights_one = np.random.randn(self.hidden_layer_size, self.input_size) * pow(1 / self.input_size, 1/2)
            self.weights_two = np.random.randn(self.hidden_layer_size) * pow(2 / self.hidden_layer_size, 1/2)
        
        
    def loadWeights(self):            
        self.weights_one = np.loadtxt(self.w1_name)
        self.weights_two = np.loadtxt(self.w2_name)
        
    def saveWeights(self):
        np.savetxt(self.w1_name, self.weights_one)
        np.savetxt(self.w2_name, self.weights_two)
        
    def saveRewards(self, rewards):
        np.savetxt("./rewards/" + self.training_name + "_rewards.txt", rewards)
        
    def graphRewards(self, rewards):
        from matplotlib import pyplot as plt
        plt.plot(rewards)

    #plt.imshow(img, interpolation='nearest')
    # Creates a 50*60 long array
    def screenProcess(self, np_array):
        # crop
        # 63 * 75 = 4725
        np_array = np_array[::8,106::8,::]
        # get red
        np_array = np_array[::,::,0]
        #flatten
        np_array = np_array.flatten()
        #normalize
        np_array[np_array != 144] = 1
        np_array[np_array != 1] = 0
        return np_array
    
    #info process
    def infoProcses(self, info):
        rectA = info[1]
        rectB = info[2]
        rectBall = info[3]
        return np.asarray([rectA.x, rectA.y, rectB.x, rectB.y, rectBall.x, rectBall.y])
    
    # computes a forward pass through the network
    # returns the hidden layer (these values are used for backprop)
    # and returns the output value
    def forward_pass(self, input_layer):
        hidden_unactivated = np.dot(self.weights_one, input_layer)
        hidden = self.relu(hidden_unactivated)
        output_unactivated = np.dot(self.weights_two, hidden)
        output = self.sigmoid(output_unactivated)
        return hidden, output
    # updates the network by performing backpropgation via gradient descent
    # at the end of every episode
    # hidden array: contains array of hidden layer values for every step
    # output array: contains array of output for every step
    # reward_array: contains awards accumulated at every step
    def backwards_pass(self, input_array, hidden_array, error_array, reward_array):
        #import pdb; pdb.set_trace()
        length = len(input_array)
        # convert to numpy arrays
        input_array = np.asarray(input_array)
        hidden_array = np.asarray(hidden_array)
        error_array = np.asarray(error_array)
        reward_array = np.asarray(reward_array)
        
        # compute discounted rewards
        # this will cause the reward from transitioning into a state
        # to take into the account the future rewards 
        discounted_reward_array = np.zeros(length)
        previous_reward = 0
        for i in range(length):
            # new point
            if reward_array[i] != 0:
                previous_reward = 0
            discounted_reward_array[i] = reward_array[i] + self.discount_rate * previous_reward
            previous_reward = discounted_reward_array[i]
        #import pdb; pdb.set_trace()
        # standardize rewards (similar to getting a z score in statistics)
        discounted_reward_array -= np.mean(discounted_reward_array)
        discounted_reward_array /= np.std(discounted_reward_array)
        # multiply the discounted rewards with our probabilities,
        # this gives a final error, or "gradient log probability" which we can use for backpropagation
        loss_delta = discounted_reward_array * error_array
        # referred to other code to check backprop math
        # get weight two adjustment from loss layer times the hiddne layer
        w2_delta = np.dot(hidden_array.T, loss_delta).ravel()
        # get hidden layer error
        hidden_delta = np.outer(loss_delta, self.weights_two)
        hidden_delta = self.relu(hidden_delta)
        # get the weight one adjustment from the hidden layer error times the input layer
        w1_delta = np.dot(hidden_delta.T, input_array)
        
        # we can now adjust our weights with the gradient arrays given by w1_delta and w2_delta
        print("w1", w1_delta)
        print("w2", w2_delta)
        self.weights_one += self.learning_rate * w1_delta
        self.weights_two += self.learning_rate * w2_delta
        
        
    # takes up action with probability p
    # for example, if the network outputs .8
    # there will be a .8 probabilit ythat we will actually go up
    # previous implementation rounded each number and then took
    # the coressponding action
    def action_from_probability(self, p):
        value = np.random.uniform()
        if value < p:
            return 10
        else:
            return -10
        
    def getAction(self, info, train = False):
        ########################
        # Input Paramter "info":
        # See environment file - info is an array containing: [the rgb array of a frame
        # of the screen, a two-element array representing the coordinates of paddleA (left paddle)
        # (so info[1][0] is the x coordinate of paddleA and info[1][1] is the y coordinate),
        # the two-element array representing the coordinates of paddleB, a two-element array
        # representing the coordinates of the ball, the reward obtained from the previous action
        # from the last screen, boolean indicating whether game is done]
        #print(info[0])
        if self.mode == Mode.image:
            input_layer = self.screenProcess(np.array(info[0]))
        else:
            input_layer = self.infoProcses(info)
        hidden, output = self.forward_pass(input_layer)
        if train:
            return self.action_from_probability(output), output, hidden
        return self.action_from_probability(output)
        # The number returned from this function determines how "violently" the paddle
        # is moved up. So 10 will move up and -10 will move down
    def playFullGame(self):
        game = Game(ai.getAction, self.getAction, False)
        game.runComp();

        while not game.done:
            game.step()
        pygame.quit()
        
        
    def train(self, save_name):
        game = Game(ai.getAction, self.getAction, False)
        pygame.init()
        pygame.display.set_caption("Pong Competition")
        
        episode_rewards = []
        episode = 0
        while True:
            info  = game.reset()
            
            if self.mode == Mode.image:
                prev_input = self.screenProcess(np.array(info[0]))
            else:
                prev_input = self.infoProcses(info)
            
            
            rewards = []
            errors = []
            inputs = []
            hidden_values = []
            
            while not info[5]:
                info = game.step(True)
                inputs.append(prev_input)
                
                reward = info[4]
                rewards.append(reward)
                
                action = info[6]
                probability = info[7] # get the probability of the action we took
                action = 1 if action == 10 else 0 # map back to 0 to 
                # this is formally known as a gradient log probability
                error = action - probability
                errors.append(error)
                
                hidden_layer_values  = info[8]
                hidden_values.append(hidden_layer_values)
                
                if self.mode == Mode.image:
                     prev_input = self.screenProcess(np.array(info[0]))
                else:
                    prev_input = self.infoProcses(info)
            
            #compute backprop to adjust weights
            self.backwards_pass(inputs, hidden_values, errors, rewards)              
            print("episode:", episode, "reward:", sum(rewards))
            episode_rewards.append(sum(rewards))
            episode += 1
            # end of episode do backprop
            if episode % self.save_interval == 0:
                self.saveWeights()
                self.saveRewards(episode_rewards)
            

    #@np.vectorize
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-1*x))

    def relu(self, vector):
        vector[vector < 0] = 0
        return vector
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        raise("missing args")
    mode = sys.argv[1]
    name = sys.argv[2]
    train_mode = sys.argv[3]
    if train_mode == "image":
        train_mode = Mode.image
    else:
        train_mode = Mode.coords
    try: 
        save_name = sys.argv[4]
    except:
        save_name = name
    player =  Player(name, mode)
    if mode == "train":
        player.train(save_name)
    else:
        player.playFullGame()