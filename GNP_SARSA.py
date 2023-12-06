import random
from tempfile import tempdir
from winreg import EnableReflectionKey
import numpy as np
import pandas as pd


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
    return list_funs[int(num)-1]
    

class GNP_Sarsa:
    def __init__(self, num_actions, num_individuals, num_nodes, num_processing_nodes, num_judgement_nodes, alpha, gamma, epsilon, train, p_mut, p_cross, num_mut, num_cross, num_generations, test):
        self.num_judgement_nodes = num_judgement_nodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.num_actions = num_actions
        self.fitness = [0 for _ in range(num_individuals)]
        self.num_individuals = num_individuals
        self.p_mut = p_mut
        self.p_cross = p_cross
        self.total_nodes = num_nodes
        self.num_judgement_nodes = num_judgement_nodes
        self.num_mut = num_mut
        self.num_cross = num_cross
        self.train = train
        self.num_generations = num_generations
        self.genes = self.generate_genes()
        self.starts = self.generate_starts()
        self.test = test

    def generate_genes(self):
        return_list = [0 for _ in range(self.num_individuals)]
        for num in range(len(return_list)):
            element = np.zeros([self.total_nodes, 12])

            for i in range(self.num_judgement_nodes):
                element[i,0] = random.randint(1,20)
                for j in range(1,6):
                    element[i,j] = random.randint(0,self.total_nodes-1)

                element[i,6] = random.randint(1,20)
                for j in range(7,12):
                    element[i,j] = random.randint(0,self.total_nodes-1)

            for i in range(self.num_judgement_nodes, self.total_nodes):
                element[i,0] = random.randint(21,22)
                if element[i,0] == 21:
                    element[i,1] = random.random()
                else:
                    element[i,1] = -1*random.random()
                element[i,4] = random.randint(21,22)
                if element[i,4] == 21:
                    element[i,5] = random.random()
                else:
                    element[i,5] = -1*random.random()
                element[i,3] = random.randint(0,self.total_nodes-1)
                element[i,7] = random.randint(0,self.total_nodes-1)
            
            return_list[num] = element
        return return_list

    def generate_starts(self):
        return_list = [0 for _ in range(self.num_individuals)]
        for num in range(len(return_list)):
            return_list[num] = random.randint(0,self.total_nodes-1)
        return return_list

    def individual_trading_run(self, index):
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
        prev_transaction = False
        prev_node = 0
        prev_action = 0
        for i in range(len(self.train)):
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
                curr_node = int(curr_node)
                if curr_node <= 19:
                    action = random.randint(0, self.num_actions - 1)
                    if prev_transaction:
                        self.genes[index][prev_node, 2 + prev_action*4] += self.alpha*(reward - self.genes[index][prev_node,2 + prev_action*4])
                        prev_transaction = False
                    if self.genes[index][curr_node, (action*6)] < 19:
                        imx, branch = map_num_to_fun(self.genes[index][curr_node, action*6])
                        imx_values.append(train.loc[i, imx])
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, int(train.loc[i, branch]) + 1]
                    elif self.genes[index][curr_node, action*6] == 19:
                        imx = train.loc[i, 'gd_cross']
                        branch = imx + 1
                        imx_values.append(imx)
                        if imx == 1:
                            gd_pos = 3
                        elif imx == -1:
                            gd_neg = 3
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, branch + 1]
                    elif self.genes[index][curr_node, action*6] == 20:
                        imx = train.loc[i, 'macd_signal']
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
                        if asset == 0:
                            if (len(imx_values) != 0) and ((sum(imx_values))/(len(imx_values)) >= self.genes[index][curr_node, 1 + action*4]):    
                                asset = cash/train.loc[i, 'Close']
                                cash = 0
                                imx_values = []
                                if initial_trade:
                                    last_price = train.loc[i, 'Close']
                                    initial_trade = False
                                else: 
                                    reward = last_price - train.loc[i, 'Close']
                                    fitness += reward
                                    last_price = train.loc[i, 'Close']
                                    prev_transaction = True 
                                    prev_node = curr_node
                                    prev_action = action

                    else:
                        if asset != 0:
                            if (len(imx_values) != 0) and (sum(imx_values))/(len(imx_values)) <= self.genes[index][curr_node, 1 + action*4]:    
                                cash = train.loc[i, 'Close']*asset
                                asset = 0
                                imx_values = []
                                if initial_trade:
                                    last_price = train.loc[i, 'Close']
                                    initial_trade = False
                                else: 
                                    reward = train.loc[i, 'Close'] - last_price
                                    fitness += reward
                                    last_price = train.loc[i, 'Close']
                                    prev_transaction = True 
                                    prev_node = curr_node
                                    prev_action = action

                    imx_values = []
                    number_units = number_units + 5
                    if action == 0:
                        curr_node = self.genes[index][curr_node, 3]
                    elif action == 1:
                        curr_node = self.genes[index][curr_node, 7]
        return fitness
    
    def generation_trading_run(self):
        for index in range(self.num_individuals):
            self.fitness[index] = self.individual_trading_run(index)
                
    def tournament_selection(self, proportion = .1):

        tournament_indices = np.random.choice(int(self.num_individuals), size=int(proportion*self.num_individuals), replace=False)
        tournament_fitness = [self.fitness[i] for i in tournament_indices]

        return tournament_indices[np.argmax(tournament_fitness)]
    
    def single_mutation_g(self, index):
        temp = self.genes[index].copy()
        for i in range(self.num_judgement_nodes):
            if random.random() < self.p_mut:
                temp[i,0] = random.randint(1,20)
            for j in range(1,6):
                if random.random() < self.p_mut:
                    temp[i,j] = random.randint(0,self.total_nodes-1)
            
            if random.random() < self.p_mut:
                temp[i,6] = random.randint(1,20)
            for j in range(7,12):
                if random.random() < self.p_mut:
                    temp[i,j] = random.randint(0,self.total_nodes-1)

        for i in range(self.num_judgement_nodes, self.total_nodes):
            if random.random() < self.p_mut:
                temp[i,0] = random.randint(21,22)
            if random.random() < self.p_mut:
                if temp[i,0] == 21:
                    temp[i,1] = random.random()
                else:
                    temp[i,1] = -1*random.random()
            if random.random() < self.p_mut:
                temp[i,4] = random.randint(21,22)
            if random.random() < self.p_mut:
                if temp[i,4] == 21:
                    temp[i,5] = random.random()
                else:
                    temp[i,5] = -1*random.random()
            if random.random() < self.p_mut:
                temp[i,3] = random.randint(0,self.total_nodes-1)
            if random.random() < self.p_mut:
                temp[i,7] = random.randint(0,self.total_nodes-1)
        
        return temp
    
    def single_mutation_s(self, index):
        if random.random() < self.p_mut:
            return random.randint(0,self.total_nodes-1)
        else:
            return self.starts[index]

    def single_crossover_g(self, index1, index2):
        to_return1, to_return2 = self.genes[index1].copy(), self.genes[index2].copy()
        for i in range(self.total_nodes):
            if random.random() < self.p_cross:
                temp = to_return1[i,:].copy()
                to_return1[i,:] = to_return2[i,:].copy() 
                to_return2[i,:] = temp
        return to_return1, to_return2

    def single_crossover_s(self, index1, index2):
        if random.random() < self.p_cross:
            return self.starts[index2], self.starts[index1]
        else:
            return self.starts[index1], self.starts[index2]
        
    def evolution_step(self):
        temp = self.genes.copy()
        temp_s = self.starts.copy()
        max_index = np.argmax(self.fitness)
        print(np.max(self.fitness))
        temp[0] = self.genes[max_index]
        temp_s[0] = self.starts[max_index]
        for i in range(1, num_mut+1):
            index = self.tournament_selection()
            temp[i] = self.single_mutation_g(index)
            temp_s[i] = self.single_mutation_s(index)
        for j in range(int(self.num_cross/2)):
            index1 = self.tournament_selection()
            index2 = self.tournament_selection()
            temp[i+2*j+1], temp[i+2*j+2] = self.single_crossover_g(index1, index2)
            temp_s[i+2*j+1], temp_s[i+2*j+2] = self.single_crossover_s(index1, index2)
        self.genes = temp
        self.starts = temp_s
    
    def full_training_run(self):
        for i in range(self.num_generations):
            self.generation_trading_run()
            self.evolution_step()
    
    def test_run(self):
        index = np.argmax(self.fitness)
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
        pnl = 0 
        for i in range(len(self.test)):
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
                curr_node = int(curr_node)
                if curr_node <= 19:
                    action = random.randint(0, self.num_actions - 1)   
                    if self.genes[index][curr_node, (action*6)] < 19:
                        imx, branch = map_num_to_fun(self.genes[index][curr_node, action*6])
                        imx_values.append(test.loc[i, imx])
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, int(test.loc[i, branch]) + 1]
                    elif self.genes[index][curr_node, action*6] == 19:
                        imx = test.loc[i, 'gd_cross']
                        branch = imx + 1
                        imx_values.append(imx)
                        if imx == 1:
                            gd_pos = 3
                        elif imx == -1:
                            gd_neg = 3
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, branch + 1]
                    elif self.genes[index][curr_node, action*6] == 20:
                        imx = test.loc[i, 'macd_signal']
                        branch = imx + 1
                        imx_values.append(imx)
                        if imx == 1:
                            dead_pos = 3
                        elif imx == -1:
                            dead_neg = 3
                        number_units = number_units + 1
                        curr_node = self.genes[index][curr_node, branch + 1]
                else:
                    if self.genes[index][curr_node, 2] > self.genes[index][curr_node, 6]:
                        action = 0 
                    else:
                        action = 1
                    if self.genes[index][curr_node, action*4] == 21:
                        if asset == 0:
                            if (len(imx_values) != 0) and ((sum(imx_values))/(len(imx_values)) >= self.genes[index][curr_node, 1 + action*4]):    
                                asset = cash/test.loc[i, 'Close']
                                cash = 0
                                imx_values = []
                                if initial_trade:
                                    last_price = test.loc[i, 'Close']
                                    initial_trade = False
                                else:
                                    reward = last_price - test.loc[i, 'Close']
                                    fitness += reward
                                    last_price = test.loc[i, 'Close']

                    else:
                        if asset != 0:
                            if (len(imx_values) != 0) and (sum(imx_values))/(len(imx_values)) <= self.genes[index][curr_node, 1 + action*4]:    
                                cash = test.loc[i, 'Close']*asset
                                pnl += cash - last_price*asset
                                asset = 0
                                imx_values = []
                                if initial_trade:
                                    last_price = test.loc[i, 'Close']
                                    initial_trade = False
                                else: 
                                    reward = test.loc[i, 'Close'] - last_price
                                    fitness += reward
                                    last_price = test.loc[i, 'Close']

                    imx_values = []
                    number_units = number_units + 5
                    if action == 0:
                        curr_node = self.genes[index][curr_node, 3]
                    elif action == 1:
                        curr_node = self.genes[index][curr_node, 7]
        return pnl
    

df = pd.read_csv('SPY_processed.csv')   
train_test_data = df.iloc[-(737+246):, :]
train = train_test_data.iloc[:737, :]
train = train.reset_index(drop=True)
test = train_test_data.iloc[737:, :]
test = test.reset_index(drop=True)
num_individuals = 300
num_processing_nodes = 10
num_judgement_nodes = 20
num_nodes = 30
num_actions = 2
alpha = 0.1 
gamma = 0.4  
epsilon = 0.1  
num_actions = 2 
p_mut = .02
p_cross = .1
num_mut = 179
num_cross = 120
num_generations = 300

def buy_and_hold(test):
    cash = 5e6
    asset = cash/test['Close'].iloc[0]
    return (test['Close'].iloc[-1]*asset - cash)

gnp_sarsa = GNP_Sarsa(num_actions, num_individuals, num_nodes, num_processing_nodes, num_judgement_nodes, alpha, gamma, epsilon, train, p_mut, p_cross, num_mut, num_cross, num_generations, test)
gnp_sarsa.full_training_run()
print(gnp_sarsa.test_run())

#print(buy_and_hold(test))





