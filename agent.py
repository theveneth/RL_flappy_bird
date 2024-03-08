import numpy as np

class MCTSAgent:
    def __init__(self, env, max_simulations=1000, exploration_constant=2):
        self.env = env
        self.max_simulations = max_simulations
        self.exploration_constant = exploration_constant
        self.action_space = env.action_space

    def policy(self, state):
        root = MCTSNode(state, None)

        # Expand the root node before running simulations
        state, root = self.expand(root)

        for _ in range(self.max_simulations):
            node = root
            state = root.state

            # Selection
            while not node.is_terminal:
                state, node = self.select(node)

            # Expansion
            if not node.is_leaf:
                state, node = self.expand(node)

            # Simulation
            reward = self.simulate(state)

            # Backpropagation
            self.backpropagate(node, reward)
        # Choose the action with the highest visit count
        if root.children:
            action = root.children.index(max(root.children, key=lambda child: child.visit_count))
        else:
            action = self.env.action_space.sample()
        
        return action

    def select(self, node):
        while not node.is_terminal and not node.is_leaf:
            node = max(node.children, key=lambda child: child.uct_value(self.exploration_constant))
            state = self.env.render()
        return state, node

    def expand(self, node):
        action = self.env.action_space.sample()
        state = self.env.render()
        new_node = MCTSNode(state, action, parent=node)
        node.children.append(new_node)
        return state, new_node

    def simulate(self, state):
        done = False
        total_reward = 0

        while not done:
            action = self.env.action_space.sample()
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward

        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

class MCTSNode:
    def __init__(self, state, action, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def uct_value(self, exploration_constant):
        if self.parent is None:
            return float('inf')
        return self.total_reward / self.visit_count + exploration_constant * np.sqrt(np.log(self.parent.visit_count) / self.visit_count)

