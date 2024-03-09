import os, sys
import gymnasium as gym
import time
from agent import MCQAgent, NStepTreeBackupAgent
import text_flappy_bird_gym
from tqdm import trange
import numpy as np
import random

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    agent = NStepTreeBackupAgent(env)

    # Learning phase
    num_episodes = 14000
    scores = []
    rewards = []

    for episode in trange(num_episodes):

        state = env.reset()
        state = state[0]

        done = False
        episode_score = 0
        total_reward = 0
        
        sequence = {0:[state, 0, None]}

        
        T = np.inf
        tau = 0
        t = 0
        while tau < T - 1:
            if t < T:
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)

                sequence[t+1] = [next_state, None, reward]
                
                # Update total reward
                total_reward += reward

                if done :
                    T = t + 1
                else:
                    next_action = agent.select_action(next_state)
                    sequence[t+1][1] = next_action

            tau = t - agent.n + 1

            if tau>=0:
                G = 0
                if t+1>T:
                    G = sequence[T][2]
                else :
                    current_sum = 0
                    for a in range(env.action_space.n): #get_pi_for_action(self, state, action)
                        if (sequence[t+1][0], a) not in agent.q_table:
                            agent.q_table[(sequence[t+1][0], a)] = random.random()

                        current_sum += agent.get_pi_for_action(sequence[t+1][0], a) * agent.q_table[(sequence[t+1][0], a)]
                    G = sequence[t+1][2] + agent.discount_factor * current_sum

                for k in range(min(t, T-1), tau, -1):
                    current_sum = 0
                    for a in range(env.action_space.n):
                        if a != sequence[k][1]:
                            if (sequence[k][0], a) not in agent.q_table:
                                agent.q_table[(sequence[k][0], a)] = random.random()

                            current_sum += agent.get_pi_for_action(sequence[k][0], a) * agent.q_table[(sequence[k][0], a)]
                    
                    G = sequence[k][2] + agent.discount_factor * current_sum + agent.discount_factor * G * agent.get_pi_for_action(sequence[k][0], sequence[k][1])
                
                if (sequence[tau][0], sequence[tau][1]) not in agent.q_table:
                    agent.q_table[(sequence[tau][0], sequence[tau][1])] = random.random()
                agent.q_table[(sequence[tau][0], sequence[tau][1])] += agent.learning_rate * (G - agent.q_table[(sequence[tau][0], sequence[tau][1])])


            state = next_state
            t+=1

            episode_score = info['score']

            if episode_score > 600: # if the agent is doing well, break the episode
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
