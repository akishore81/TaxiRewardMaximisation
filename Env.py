# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[i,j] for i in range(m) for j in range(m) if i!=j] + [(0,0)] # list of all actions that can happen
        self.state_space = [[x,y,z] for x in range(m) for y in range(t) for z in range(d)] # list of all possible states
        self.state_init = random.choice(self.state_space) # pick up a random state from all possible state values

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        # One hot encoding of state space
        state_encod = [0 for i in range (m+t+d)]
        
        #print("state 0: ", state[0])
        state_encod[np.int(state[0])] = 1
        
        #print("state 1: ", state[1])
        #print("state 1+M: ", (m+state[1]))
        state_encod[m + np.int(state[1])] = 1
        
        #print("state 2: ", state[2])
        #print("state 2+M+T: ", (m+t+state[2]))
        state_encod[m + t + np.int(state[2])] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
            
        state_encod = [0 for i in range(m+t+d+m+m)]
        # one hot encoding of state space
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        
        # one hot encoding of action space
        if action[0]!=0:
            state_encod[m+t+d+action[0]] = 1 # one hot encoding pickup location
        if action[1]!=0:
            state_encod[m+t+d+m+action[1]] = 1 # one hot encoding of drop location
        
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        location = state[0]
        
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        
        if requests >15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        #actions.append([0,0])

        return possible_actions_index,actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        #next_state, ride_time, transit_time, hold_time, terminal_state = self.next_state_func(state, action, Time_matrix)
        
        next_state, ride_time, transit_time, hold_time = self.next_state_func(state, action, Time_matrix)
        
        revenue = R * ride_time
        cost = C * (ride_time + transit_time + hold_time)
        
        reward = revenue - cost
        
        total_time = ride_time + transit_time + hold_time
        
        #return reward, next_state, terminal_state
        return reward, next_state, total_time


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        total_time = 0
        transit_time = 0 # travel from current location to pickup
        ride_time = 0 # travel time from pickup to drop location
        hold_time = 0 # incase the requests are rejected
        
        current_loc = np.int(state[0])
        current_time = np.int(state[1])
        current_day_of_week = np.int(state[2])
        
        pickup_loc = np.int(action[0])
        drop_loc = np.int(action[1])
        
        
        # first check if the request is rejected
        if (pickup_loc == 0) and (drop_loc == 0):
            hold_time = 1
            
            #total_time = total_time + hold_time
            
            new_time, new_day_of_week = self.get_new_time_day(current_time,current_day_of_week,hold_time)
            
            next_loc = current_loc
        
        # if the request is not rejcted and current location is same as pickup location
        elif current_loc == pickup_loc:
            ride_time = Time_matrix[current_loc][drop_loc][current_time][current_day_of_week]
            
            #total_time = total_time + ride_time
            
            new_time, new_day_of_week = self.get_new_time_day(current_time,current_day_of_week,ride_time)
            
            next_loc = drop_loc
        
        # if the request is not rejcted and current location is not same as pickup location
        elif current_loc != pickup_loc:
            transit_time = Time_matrix[current_loc][pickup_loc][current_time][current_day_of_week]
            
            new_time, new_day_of_week = self.get_new_time_day(current_time,current_day_of_week,transit_time)
            
            ride_time = Time_matrix[pickup_loc][drop_loc][np.int(new_time)][np.int(new_day_of_week)]
            
            new_time, new_day_of_week = self.get_new_time_day(new_time,new_day_of_week,ride_time)
            
            #total_time = total_time + transit_time + ride_time
            
            next_loc = drop_loc
        
        next_state = (next_loc,new_time,new_day_of_week)
        
        total_time = hold_time + transit_time + ride_time
        
        #print("Total Time: ",total_time)
        
        #if (total_time >= 24*30) :
        #    terminal_state = True
        #    total_time = 0
        #else:
        #    terminal_state = False
        
        #return next_state, ride_time, transit_time, hold_time, terminal_state
        return next_state, ride_time, transit_time, hold_time
        
    
    
    def get_new_time_day(self, curr_time,curr_day_of_week, ride_duration):
        
        # check if the day has changed or not
        if curr_time + ride_duration < 24:
            # day has not changed
            new_time = curr_time + ride_duration
            new_day_of_week = curr_day_of_week
        else:
            new_time = curr_time + ride_duration - 24
            
            if curr_day_of_week == 6:
                new_day_of_week = 0
            else:
                new_day_of_week = curr_day_of_week + 1
        
        return new_time, new_day_of_week


    def reset(self):
        return self.action_space, self.state_space, self.state_init
