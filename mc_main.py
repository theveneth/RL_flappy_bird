import os, sys
import gymnasium as gym
import time
from agent import MCTSAgent
import text_flappy_bird_gym

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-screen-v0', height=15, width=20, pipe_gap=4)
    agent = MCTSAgent(env)

    # Learning phase
    num_episodes = 1000
    scores = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_score = 0

        while not done:
            action = agent.policy(obs)
            obs, reward, done, _, info = env.step(action)
            episode_score += reward

        scores.append(episode_score)
        print(f"Episode {episode + 1} completed. Score: {episode_score}")

    # Display the improving score after learning
    print("\nLearning completed! Now showing a game:")
    obs = env.reset()
    done = False
    score = 0

    while not done:
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.05)  # FPS

        action = agent.policy(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward

    print(f"\nGame over. Score: {score}")

    env.close()
