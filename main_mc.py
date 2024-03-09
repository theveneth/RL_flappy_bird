import os, sys
import gymnasium as gym
import time
from agent import MCQAgent, NStepTreeBackupAgent
import text_flappy_bird_gym
from tqdm import trange

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    agent = MCQAgent(env)

    # Learning phase
    num_episodes = 50000
    scores = []
    rewards = []

    for episode in trange(num_episodes):

        state = env.reset()
        state = state[0]

        done = False
        episode_score = 0
        total_reward = 0

        while not done:
            
            action = agent.policy(state)

            next_state, reward, done, _, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            #MAJ
            state = next_state


            episode_score = info['score']
            total_reward += reward
            if episode_score > 1000: # if the agent is doing well, break the episode
                print('limit reached')
                break


        rewards.append(total_reward)
        scores.append(episode_score)
        #print(f"Episode {episode + 1} completed. Score: {episode_score}")
    env.close()


#plot evolution scores
def moving_average(data, window_size=100):
    """Calculate the moving average of a list."""
    moving_avg = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1): i + 1]
        avg = sum(window) / len(window) if window else 0
        moving_avg.append(avg)
    return moving_avg


import matplotlib.pyplot as plt
plt.plot(moving_average(rewards))
plt.show()
# Save the model