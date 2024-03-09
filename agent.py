import numpy as np

class MCQAgent: #Working well
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state = None
        self.q_table = {}

    def policy(self, state):
        # Convert the state into a tuple of its components
        #round distance to 1 decimal places to reduce the number of states
        
        if state not in self.q_table.keys():
            self.q_table[state] = np.zeros(self.env.action_space.n)

        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action

    def update(self, state, action, reward, next_state, done):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.env.action_space.n)

        q_value = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])

        self.q_table[state][action] = q_value + self.learning_rate * (target - q_value)


class NStepTreeBackupAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, n=3, epsilon=0.05):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n = n
        #q_table : (state, action) -> value
        self.q_table = {}

        self.epsilon = epsilon

    def get_pi_for_action(self, state, action):
        # Calculate the policy probability for the given action
        k = self.env.action_space.n
        pi_action = self.epsilon / k + (1 - self.epsilon) if action == self.policy(state) else self.epsilon / k
        return pi_action
    
    def policy(self, state):
        values = [self.q_table.get((state, action), 0) for action in range(self.env.action_space.n)]
        #on peut ajouter un epsilon greedy
    
        return np.argmax(values)
    


    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        
        return self.policy(state)

    def update(self, sequence):
        states, actions, rewards = sequence[:-1], sequence[1:-1], sequence[2:]
        T = len(states)

        for t in range(T - self.n):
            tau = t + 1
            G = rewards[t + self.n] if t + self.n < T else 0

            for k in range(t + self.n, t, -1):
                state = states[k]
                action = actions[k]

                # Initialize q_table and policy for state if it doesn't exist
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(self.env.action_space.n)
                    self.policy[state] = np.argmax(self.q_table[state])

                G = rewards[k] + self.discount_factor * (np.dot(self.policy[state], self.q_table[state]) if k + 1 < T else 0) + self.discount_factor * (1 - self.policy[state][action]) * G

            state = states[tau]
            action = actions[tau]

            # Initialize q_table and policy for state if it doesn't exist
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.env.action_space.n)
                self.policy[state] = np.argmax(self.q_table[state])

            self.q_table[state][action] += self.learning_rate * (G - self.q_table[state][action])
            self.policy[state] = np.argmax(self.q_table[state])