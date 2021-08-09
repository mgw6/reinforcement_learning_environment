"""
MacGregor Winegard, June 2020
CSE 368, University at Buffalo
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class Deterministic_Environment(gym.Env):
    
    def __init__(self):
        self.observation_space = spaces.Discrete(16)
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = 8
        self.reward = 0 #I want the reward to be in the environment since there will be different places to bonuses
        
    def reset(self):
        self.timestep = 0
        self.agent_pos = [0,0]
        self.goal_pos = [3,3]
        self.bonus = [1,1] #sets a bonus location
        self.trap = [2,0] #sets a trap location
        self.state = np.zeros((4,4))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        obervation = self.state.flatten()
        self.reward = 0 #program this into the reset so i don't have to worry about it later
        done = False #program this into the reset so i don't have to worry about it later
        return obervation
    
    def step(self, action):
        if action == 0: #down
            self.agent_pos[0] +=1
        if action == 1: #up
            self.agent_pos[0] -= 1
        if action == 2:#right
            self.agent_pos[1] += 1
        if action == 3:#left
            self.agent_pos[1] -= 1
        
        self.agent_pos = np.clip(self.agent_pos, 0,3)
        self.state = np.zeros((4,4))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        observation = self.state.flatten()
        
        if(self.agent_pos == self.goal_pos).all(): #if it lands on the goal
            self.reward +=3

        if(self.agent_pos == self.bonus).all(): #if it lands on the bonus
            self.reward  +=1

        if(self.agent_pos == self.trap).all(): #If it lands on the trap
            self.reward  -=1
        
        self.timestep += 1
        done = True if self.timestep >= self.max_timesteps or (self.agent_pos == self.goal_pos).all() else False
        info = {}
        return observation, self.reward, done, info
        
    def render(self):
        plt.imshow(self.state)


class Stochastic_Environment(gym.Env):
    
    def __init__(self):
        self.observation_space = spaces.Discrete(16)
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = 8
        self.reward = 0

    def reset(self):
        self.timestep = 0
        self.agent_pos = [0,0]
        self.goal_pos = [3,3]
        self.trap1 = [2,1] #sets a trap location
        self.trap2 = [3,0] #sets a trap location
        self.state = np.zeros((4,4))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        obervation = self.state.flatten()
        self.reward = 0 #programmed into reset
        done = False #programmed into reset
        return obervation
    
    def step(self, action):
        a = np.random.random()

        if action == 0: #down
            if (a<.901): #this is the move it "should" be
                self.agent_pos[0] += 1
            elif(a>.901 and a<.95):
                self.agent_pos[0] -= 1
            else:
                self.agent_pos[1] += 1
       
        if action == 1: # up
            if (a<.87):#move it should be 
                self.agent_pos[0] -= 1
            elif (a>=.87 and a<.97): #do nothing
                self.agent_pos = self.agent_pos
            else: #just make it restart (insert evil laugh)
                self.agent_pos = [0,0] 

        if action == 2:#right
            if (a<.91): #move it should be 
                self.agent_pos[1] += 1
            elif (a>=.91 and a>.94):  #move two to the right
                self.agent_pos[1] += 2
            else: #move to the left
                self.agent_pos[1] -= 1

        if action == 3: #left
            if (a<.875): # left
                self.agent_pos[1] -= 1
            elif (a>=.875 and a<94): #left and down
                 self.agent_pos[0] += 1
                 self.agent_pos[1] -= 1   
            else:
               self.agent_pos = [2,0]  

        """
        if self.agent_pos == [0,2]: #if it happens to wind up on this space
            if (a>.98): #There is a small probability that we will flip you accross the board, oops... sorry
                self.agent_pos = [2,0] 
        #Well this actually does not appear to work, something about a truth value of an array, but I tried
        """

        
        self.agent_pos = np.clip(self.agent_pos, 0,3)
        self.state = np.zeros((4,4))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        observation = self.state.flatten()
        
        if(self.agent_pos == self.goal_pos).all(): #if it lands on the goal
            self.reward +=3

        if(self.agent_pos == self.trap1).all()  or (self.agent_pos == self.trap2 ).all(): #if it lands on the bonus
            #I like to think of this as the question mark box in Mario, anything could happen
            if (a<.33):
                self.reward -=1
            elif (a>=.33 and a<.8):
                self.reward += 1
            else:  
                self.reward +=2

        
        self.timestep += 1
        done = True if self.timestep >= self.max_timesteps or (self.agent_pos == self.goal_pos).all() else False
        info = {}
        return observation, self.reward, done, info
        
    def render(self):
        plt.imshow(self.state)

class Random_Agent:
    
    def __init__(self, env):
        
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def step(self, obs):
        return np.random.choice(self.action_space.n)
    
    def policy(self):
        print("policy defined")

class Learn: #I want to make learning a function so I can call it for either environment
    #Because Utkarsh went over Q-Learning in recitation I feel like I have a really good grasp on that and will use it

    def Learning(env, runs): #in trouble shooting I decided to make the number of runs changeable becauase 1000 satisfied the deterministic, but for Stochastic it appears to need closer to 10,000 runs
        reward_hist = []
        epsilon = 1
        prob_table = np.zeros((16, 4)) #Sets up the table of probabilities like Utkarsh showed us
        #print(prob_table) #since there are 16 spaces in a 4x4 grid and 4 move choices, that dictates the size
        Alpha = .65 #Learning Rate, I took the exact middle of what Alina Reccomended
        Gamma = .895 # Discount factor, exact middle of Alina's Reccomendation 

        for i in range(runs): #Loop to go through the unputted amount of runs tests
            obs = env.reset() #reset environment
            done = False #I probably could have made a boolean as a self. in the environment so I didn't need this but oh well
            current_score = 123312123 #these are numbers that should not show up in the end unless there is a problem
            new_score = 12344
            score_difference = 6192020 #like today's date
            current_state = 1234567890 #Slid my hand down the number keys like a fancy piano player
            new_state = 987654321
            
         
            while not done: #Yeah we looooove Boolean while loops
                current_score = env.reward
                current_state = (4*(env.agent_pos[0]) + env.agent_pos[1]) 
                #I tried messing around with the observation_space but its discrete and I can't figure out how to convert it

                greedy = np.random.random() #are we going to do a random move or go tried and true?
                if greedy > epsilon: #We're greedy like the cookie monster and we are taking ALL the cookies                 
                    action = np.argmax(prob_table[current_state,:])
                    #https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html 
                else: #we are learning through a random move
                    action = agent.step(obs) #So this uses the random action to choose a random move
                           
                obs, reward, done, _ = env.step(action) #Do the thing
     
                new_state = (4*(env.agent_pos[0]) + env.agent_pos[1]) #I tried messing around with the observation_space but its discrete and I can't figure out how to convert it
                new_score = env.reward #actually I could have just combined this with the next line and condensed it but whatever
                score_difference = new_score - current_score #This is to find out how much our score changed from the last go round
                #earlier I said I wanted to make the reward an element of the environment but now I see why Utkarsh did it the way he did
                #oh well, ya live and ya learn

                prob_table[current_state, action] = prob_table[current_state, action] + Alpha * (score_difference + Gamma * np.max(prob_table[new_state, :]) - prob_table[current_state, action])
                #I had to reference slide 24 of lecture 7.1 about 20 times for this
                #also it annoys me how far it goes over, but oh well
            #This ends the boolean while loop
               
            reward_hist.append(reward) #Once we have a reward add it to the list
            epsilon -= (1/runs) #Epsilon decreases a small amunt for every run, so that on the last run it takes the best choice of moves
            #so on run one epsilon is 1, on the last run it is 1 - (1/runs). If run is 1000 then epsilon is .001
        

        print("Here is the reward history:")
        print(reward_hist) #Self explanatory

        window = 10 #sets up graph
        plt.xlabel("Episode") #tbh this is a straight copy and paste from Utkarsh's workbook
        plt.ylabel("Rewards")
        plt.plot([np.mean(reward_hist[tr: tr+window]) for tr in range(window, len(reward_hist))]) #prints data on Graph

        print("\nHere is the transition probability table:")
        print(prob_table)


if __name__ == "__main__":
    env = Deterministic_Environment() #initialize the determistic environment
    agent = Random_Agent(env) #initialize agent

    Learn.Learning(env,1000) #okay and then lets learn