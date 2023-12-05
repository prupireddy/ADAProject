import random
from winreg import EnableReflectionKey
import numpy as np


def map_num_to_fun(num):
    list_funs = [('rod_5_IMX','rod_5_branch'),
            ('rod_13_IMX','rod_13_branch'),
            ('rod_26_IMX','rod_26_branch'),
            ('rsi_5_IMX','rsi_5_branch'),
            ('rsi_13_IMX','rsi_13_branch'),
            ('rsi_26_IMX','rsi_26_branch'),
            ('roc_5_IMX','roc_5_branch'),
            ('roc_13_IMX','roc_13_branch'),
            ('roc_26_IMX','roc_26_branch'),
            ('volume_5_IMX','volume_5_branch'),
            ('volume_13_IMX','volume_13_branch'),
            ('volume_26_IMX','volume_26_branch'),
            ('rci_9_IMX','rci_9_branch'),
            ('rci_18_IMX','rci_18_branch'),
            ('rci_27_IMX','rci_27_branch'),
            ('d_12_IMX','d_12_branch'),
            ('d_20_IMX','d_20_branch'),
            ('d_30_IMX','d_30_branch')]       
    return list_funs[num-1]
    

class GNP_Sarsa:
    def __init__(self,num_individuals, num_nodes, num_processing_nodes, num_judgement_nodes, alpha, gamma, epsilon, train):
        self.genes = self.generate_genes(num_individuals, num_nodes, num_processing_nodes, num_judgement_nodes)
        self.starts = self.generate_starts(num_individuals, num_nodes)
        self.num_judgement_nodes = num_judgement_nodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate

    def generate_genes(self, number_of_individuals, total_nodes, number_of_judgement_nodes):
        return_list = [0 for _ in range(number_of_individuals)]
        for num in range(len(return_list)):
            element = np.zeros([total_nodes, 12])

            for i in range(number_of_judgement_nodes):
                element[i,0] = random.randint(1,20)
                for j in range(1,6):
                    element[i,j] = random.randint(0,total_nodes-1)

                element[i,6] = random.randint(1,20)
                for j in range(7,12):
                    element[i,j] = random.randint(0,total_nodes-1)

            for i in range(number_of_judgement_nodes, total_nodes):
                element[i,0] = random.randint(21,22)
                if element[i,0] == 21:
                    element[i,1] = random.random()
                else:
                    element[i,1] = -1*random.random()
                element[i,4] = random.randint(1,2)
                if element[i,4] == 21:
                    element[i,5] = random.random()
                else:
                    element[i,5] = -1*random.random()
                element[i,3] = random.randint(0,total_nodes-1)
                element[i,7] = random.randint(0,total_nodes-1)
            
            return_list[num] = element
        return return_list

    def generate_starts(self, number_of_individuals, total_nodes):
        return_list = [0 for _ in range(number_of_individuals)]
        for num in range(len(return_list)):
            return_list[num] = random.randint(0,total_nodes-1)
        return return_list

    def individual_trading_run(self, index, train):
        gd_pos = 0
        gd_neg = 0
        dead_pos = 0
        dead_neg = 0
        curr_node = self.starts[index]
        imx_values = []
        cash = 5e6
        asset = 0
        fitness = 0 
        last_price = 0
        initial_trade = True 
        for i in range(len(train)):
            number_units = 0
            if gd_pos != 0:
                imx_values.append(1)
                gd_pos = gd_pos - 1
            if gd_neg != 0:
                imx_values.append(-1)
                gd_neg = gd_neg - 1
            if dead_pos != 0:
                imx_values.append(1)
                dead_pos = dead_pos - 1
            if dead_neg != 0:
                imx_values.append(-1)
                dead_neg = dead_neg - 1           
            while number_units < 5:
                if curr_node <= 20:
                    action = random.randint(0, self.num_actions - 1)
                    if prev_transaction:
                        self.genes[index][prev_node, 2 + prev_action*4] += self.alpha*(reward + self.gamma*self.genes[index][curr_node, 2 + action*4] - self.genes[index][prev_node, 2 + prev_action*4])
                        prev_transaction = False    
                    if self.genes[index][curr_node, action*6] < 19:
                        imx, branch = map_num_to_fun(self.genes[index][curr_node, action*6])
                        imx_values.append(train[i,imx])
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, train[i,branch] + 1]
                    elif self.genes[index][curr_node, action*6] == 19:
                        imx = train[i,'gd_cross']
                        branch = imx + 1
                        imx_values.append(imx)
                        if imx == 1:
                            gd_pos = 3
                        elif imx == -1:
                            gd_neg = 3
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, branch + 1]
                    elif self.genes[index][curr_node, action*6] == 20:
                        imx = train[i,'macd_signal']
                        branch = imx + 1
                        imx_values.append(imx)
                        if imx == 1:
                            dead_pos = 3
                        elif imx == -1:
                            dead_neg = 3
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, branch + 1]
                else:
                    if random.random() < self.epsilon:
                        action = random.randint(0, self.num_actions - 1)
                    else:
                        if self.genes[index][curr_node, 2] > self.genes[index][curr_node, 6]:
                            action = 0 
                        else:
                            action = 1
                    if prev_transaction:
                        self.genes[index][prev_node, 2 + prev_action*4] += self.alpha*(reward + self.gamma*self.genes[index][curr_node, 2 + action*4] - self.genes[index][prev_node, 2 + prev_action*4])
                        prev_transaction = False
                    if self.genes[index][curr_node, action*4] == 21:
                        if asset != 0:
                            if (sum(imx_values))/(len(imx_values)) >= self.genes[index][curr_node, 1 + action*4]:    
                                asset = cash/train[i,'Close']
                                cash = 0
                                imx_values = []
                                if initial_trade:
                                    last_price = train[i,'Close']
                                    initial_trade = False
                                else:
                                    reward = last_price - train[i,'Close']
                                    fitness += reward
                                    last_price = train[i,'Close']
                                prev_transaction = True 
                                prev_node = curr_node
                                prev_action = action

                    else:
                        if asset == 0:
                            if (sum(imx_values))/(len(imx_values)) <= self.genes[index][curr_node, 1 + action*4]:    
                                asset = 0
                                cash = train[i,'Close']*asset
                                imx_values = []
                                if initial_trade:
                                    last_price = train[i,'Close']
                                    initial_trade = False
                                else: 
                                    reward = train[i,'Close'] - last_price
                                    fitness += reward
                                    last_price = train[i,'Close']
                                prev_transaction = True 
                                prev_node = curr_node
                                prev_action = action

                    number_units = number_units + 5
                    if action == 0:
                        curr_node = self.genes[index][curr_node, 3]
                    elif action == 1:
                        curr_node = self.genes[index][curr_node, 7]
                


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

num_nodes = 30
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount rate
epsilon = 0.1  # Exploration rate

gnp_sarsa = GNP_Sarsa(num_nodes, num_actions, alpha, gamma, epsilon, threshold)

# Training phase: Train the GNP-Sarsa algorithm using historical data
# You need to implement this part by processing the historical data and updating Q-values

