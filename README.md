# Reinforcement Learning Environment
Reinforcement Learning Environment using Open AI Gym. This was an 
ssignment for Intro to Artificial Intelligence (CSE 368) at the Univeristy at Buffalo. Written June 2020. 

# From Assignment:
The goal of the assignment is to explore and get an experience of building reinforcement learning environments, 
following the OpenAI Gym standards. The project consists of building deterministic and stochastic
environments that are based on Markov decision process, and applying a tabular method to solve them.

# Final Report:
MacGregor Winegard

CSE 368 - Artificial Intelligence

Assignment 2 Report

For assignment two, the student was required to set up a deterministic environment, a stochastic environment based on the 
deterministic one, and then implement a tabular method to solve both of these environments. The deterministic environment 
I set up contains 16 states in a 4 by 4 grid, with 4 possible moves in each state. The max number of timesteps is set at 8, 
because 6 are needed to go directly from one corner to the other. By giving the machine 8 it has room for a couple of missteps. 
At the start the agent is in the top left corner of the board and the goal is in the bottom right corner. 
There is a bonus at (1,1) which gives an extra reward, and a trap at (2,0) which takes away 1 point. 
For the deterministic environment, whichever move choice is selected by the machine is certain to happen. 
When the agent reaches the goal it gets a +3 reward and the game terminates. 
The game also terminates when it has reached the max number of timesteps, however without the +3 reward. 

For the stochastic environment, the board is set up in the same 4 by 4 grid with 4 move choices, and 8 timesteps. 
The major difference between the deterministic and stochastic environment is that with the stochastic when a move is 
selected there is a varying probability that that move will actually occur. 
The most likely scenario is that the selected move occurs, but there are varying probabilities that others will occur. 
For instance, if the agent chooses left, there is an 87.5% chance that it will go left, but there is also a 6.5% 
chance that it actually goes down and to the left, as well as a 6% chance it will actually get thrown to position (2,0).  
The other difference is the variation on rewards. The rewards are located in positions (2,1) and (3,0). 
When the agent lands on either of those squares there is a .33 chance that it will lose a point, a .47% chance that 
it gains a point and a .3% chance that it gains 2 points. 

My tabular method was designed as a function because I wanted to be able to write one function to test on both environments, 
similar to how in assignment one we wrote two search functions to test on one environment. 
I later found that this was handy because I could make the number of runs an input of the function so that I could change 
how many runs were used in each environment. 
The stochastic environment needs more runs to show a positive correlation than the deterministic did. 
I set up my learning function based on the Q - Learning algorithm. 
The first thing the function does is set initialize a reward history list, declares epsilon as 1, makes 
a blank Transition Probability Table, and sets our Alpha = .65 and  Gamma = .895. 
These numbers were chosen because they were the exact middle of the range Professor Alina suggested. 
There is then a for loop that goes for the number of runs input into the function. 
For every pass, it resets the environment, sets done = to false, and then resets 5 variables 
which are used in the next while loop: current score, new score, score difference, current state, and new state. 
I purposefully have this ???reset??? be something crazy because they all should be changed later. 
If they are not reset then they will show up later and I will know to start looking there for errors. 

I then go into a while boolean loop for every time the game is played. 
I set the current score and state the values they should be. 
I then have the function generate a random number for greedy. 
In the next line this will dictate whether or not we choose a random move or take a move we believe will yield a higher reward. 
At first epsilon was first initialized as 1, so it is impossible for the greedy value to be higher than epsilon. 
Over time epsilon will decrease which will make it more and more likely to choose the option that yields a higher reward.  
I then set the new state, new score, and find the difference between the new and the old score to figure out how 
much the score changed. 

Then we update our probability table. 
I used the Q-Learning equation on slide 24 of lecture 7.1-7.2. 
When written out in code, what it says is that the state we were just in now equals that same thing, plus 
all of the newly found scores plus gamma times the highest option in the new state, minus the old value of 
the probability table at the old state, times alpha. 

Once it has solved the problem, it adds the final reward to the reward history list and decreases epsilon by a small amount. 
After it has gone through the problem the inputted amount of times, it prints 
out the reward history as a list as well as a graph of the history over time. 
It also prints out the updated transition probability table. 
