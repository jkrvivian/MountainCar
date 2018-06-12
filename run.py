import gym
import numpy as np
from brain import DeepQNetwork

env = gym.make('MountainCar-v0')
steps = []

def run():
    step = 0

    for episode in range(3000):
        # initial observation
        observation = env.reset()
        step = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            # update reward
            reward = (observation_[0]-observation[0]) * observation_[1]
            if observation_[0] > -0.5:
                reward +=  3 * abs(observation_[0] - (-0.5)) 
            else:
                reward += abs(observation_[0] - (-0.5))

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if observation[0] >= 0.5:
                print("steps: ", step)
                steps.append(step)
                break
            step += 1

    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', 
					default='False',
					help='is testing mode or not')
    args = parser.parse_args()

    RL = DeepQNetwork(3, 2,
                      args.reuse,
                      learning_rate=0.1,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run()

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(steps)), steps)
    plt.ylabel('steps cost')
    plt.xlabel('episode')
    plt.savefig('steps_picture.png')
    plt.show()

