import random
import numpy as np

def generate_genes(number_of_individuals, total_nodes, number_of_processing_nodes, number_of_judgement_nodes):
    return_list = [0 for _ in range(number_of_individuals)]
    for num in range(len(return_list)):
        element = np.zeros([total_nodes, 8])

        for i in range(number_of_judgement_nodes):
            element[i,0] = random.randint(1,20)
            for j in range(1,6):
                element[i,j] = random.randint(0,total_nodes-1)

        for i in range(number_of_judgement_nodes, total_nodes):
            element[i,0] = random.randint(1,2)
            if element[i,0] == 1:
                element[i,1] = random.random()
            else:
                element[i,1] = -1*random.random()
            element[i,4] = random.randint(1,2)
            if element[i,4] == 1:
                element[i,5] = random.random()
            else:
                element[i,5] = -1*random.random()
            element[i,3] = random.randint(0,total_nodes-1)
            element[i,7] = random.randint(0,total_nodes-1)
        
        return_list[num] = element
    return return_list

def generate_starts(number_of_individuals, total_nodes):
    return_list = [0 for _ in range(number_of_individuals)]
    for num in range(len(return_list)):
        return_list[num] = random.randint(0,total_nodes-1)
    return return_list



'''
class GNP_Sarsa:
    def __init__(self, num_nodes, num_actions, alpha, gamma, epsilon):
        self.nodes = [Node(random.choice([0, 1, 2]), i) for i in range(num_nodes)]
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate

    def select_action(self, node):
        if node.node_type == 1:
            # Implement the action selection logic for judgment nodes (state transitions)
            # Choose an action based on Q-values and epsilon-greedy policy
            if random.random() < self.epsilon:
                # Exploration: Choose a random action
                action = random.choice(range(self.num_actions))
            else:
                # Exploitation: Choose the action with the highest Q-value
                action = max(node.Q_values, key=node.Q_values.get)
            return action
        elif node.node_type == 2:
            # Implement the action selection logic for processing nodes (buying/selling)
            # Choose an action based on the threshold
            return action if node.Q_values.get(action, 0) >= self.threshold else None
        else:
            # Start node, no action to select
            return None

    def update_q_values(self, node, action, reward, next_node, next_action):
        # Update Q-values using SARSA update rule
        # Q(node, action) = Q(node, action) + alpha * (reward + gamma * Q(next_node, next_action) - Q(node, action)
        if node.node_type == 1:
            if action is not None:
                node.Q_values[action] = node.Q_values.get(action, 0) + self.alpha * (reward + self.gamma * next_node.Q_values.get(next_action, 0) - node.Q_values.get(action, 0))
        elif node.node_type == 2:
            # Implement the update logic for processing nodes
            if action is not None:
                # Update Q-values for buying and selling actions
                node.Q_values[action] = node.Q_values.get(action, 0) + self.alpha * (reward - node.Q_values.get(action, 0)

    def trading_simulation(self, training_data):
        capital = 5000000  # Initial capital
        state = self.nodes[0]  # Start from the start node
        while True:
            action = self.select_action(state)
            if action is None:
                break  # No more actions to take
            # Execute the action (buying/selling logic)
            # Update capital and move to the next node
            # Update Q-values using the SARSA learning process
            next_node = self.nodes[state.connections[action]]  # Assuming connections are indices of next nodes
            next_action = self.select_action(next_node)
            reward = 0  # Calculate the reward based on the trading action and market data
            capital += reward
            self.update_q_values(state, action, reward, next_node, next_action)
            state = next_node
        return capital

if __name__ == "__main__":
    num_nodes = 31  # Total number of nodes in an individual
    num_actions = 2  # Number of possible actions (buy, sell)
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount rate
    epsilon = 0.1  # Exploration rate
    threshold = 0.2  # Threshold for buying or selling

    gnp_sarsa = GNP_Sarsa(num_nodes, num_actions, alpha, gamma, epsilon, threshold)

    # Training phase: Train the GNP-Sarsa algorithm using historical data
    # You need to implement this part by processing the historical data and updating Q-values

    # Testing phase: Use the trained algorithm to simulate trading
    test_data = []  # Historical data for testing
    final_capital = gnp_sarsa.trading_simulation(test_data)
    print("Final capital:", final_capital)
'''