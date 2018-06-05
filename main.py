import gym
import random
import numpy as np
env = gym.make('MountainCar-v0')

# create Qtable
Qtable = {} 

# constant 
ALPHA = 1.0 
GAMMA = 0.5
epsilon = 0.2

def get_action(observation, index):
    pos = int(round(observation[0]))
    velo = int(round(observation[1]))
    act = 0

    if np.random.random_sample() < epsilon or ((pos, velo, act) not in Qtable):
        return env.action_space.sample()
    else:
        maxQ = -1
        index = 0
        for a in range(2):
            if maxQ < Qtable[(pos, velo, a)]:
                maxQ = Qtable[(pos, velo, a)]
                index = a
        return a

def update_Q(pre_observation, observation, action, reward):
    pos, velo = int(round(observation[0])), int(round(observation[1]))
    pre_pos, pre_velo = int(round(pre_observation[0])), int(round(pre_observation[1]))
    maxQ = -1 

    if (pos, velo, action) in Qtable:
        for a in range(2):
            maxQ = max(maxQ, Qtable[(pos, velo, a)])
    else:
        for a in range(2):
            Qtable[(pos, velo, a)] = 0
            maxQ = 0
    Qtable[(pre_pos, pre_velo, a)] += ALPHA * (reward + GAMMA * maxQ - Qtable[(pre_pos, pre_velo, a)])

if __name__ == '__main__':
    for i_episode in range(100):
        print("round ", i_episode) 
        # initialize state
        observation = env.reset()
        pre_observation = observation
        ALPHA *= 0.99
        for t in range(10000):
            env.render()

            # select an action
            action = get_action(pre_observation, t) 

            # execute action
            observation, reward, done, info  = env.step(action)

            # update Qtable
            update_Q(pre_observation, observation, action, reward)
        
        if observation[0] == 0.5:
            print("steps: ", t)
            break
    env.close()
