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
    pos = int(round(observation[0], 1))
    velo = int(round(observation[1], 1))
    act = 0

    if np.random.random_sample() < epsilon or ((pos, velo, act) not in Qtable):
        return env.action_space.sample()
    elif ((pos, velo, act) not in Qtable):
        for a in range(3):
            Qtable[(pos, velo, a)] = 0
        return env.action_space.sample()

    else:
        maxQ = Qtable[(pos, velo, 0)] 
        index = 0
        for a in range(3):
            if maxQ < Qtable[(pos, velo, a)]:
                maxQ = Qtable[(pos, velo, a)]
                index = a
        return a

def update_Q(pre_observation, observation, action, reward):
    pos, velo = int(round(observation[0], 1)), int(round(observation[1], 1))
    pre_pos, pre_velo = int(round(pre_observation[0], 1)), int(round(pre_observation[1], 1))
    maxQ = 0

    if (pos, velo, action) in Qtable:
        maxQ = Qtable[(pos, velo, 0)] 
        for a in range(3):
            maxQ = max(maxQ, Qtable[(pos, velo, a)])
    else:
        for a in range(3):
            Qtable[(pos, velo, a)] = 0
    if (pre_pos, pre_velo, action) not in Qtable:
        for a in range(3):
            Qtable[(pre_pos, pre_velo, a)] = 0
    Qtable[(pre_pos, pre_velo, action)] += ALPHA * (reward + GAMMA * maxQ - Qtable[(pre_pos, pre_velo, action)])

if __name__ == '__main__':
    for i_episode in range(100):
        print("round ", i_episode) 

        # initialize state
        observation = env.reset()
        pre_observation = observation
        GAMMA *= 0.99

        for t in range(10000):
            env.render()

            # select an action
            action = get_action(pre_observation, t) 

            # execute action
            observation, reward, done, info  = env.step(action)

            # update Qtable
            reward = abs(observation[0]) - 0.5 + abs(observation[1])
            update_Q(pre_observation, observation, action, reward)
            ALPHA *= 0.99
        
            if observation[0] >= 0.5:
                print("steps: ", t)
                break
    env.close()
