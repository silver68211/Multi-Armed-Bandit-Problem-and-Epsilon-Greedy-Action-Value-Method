import numpy as np
import matplotlib.pyplot as plt

class BanditProblem():
    def __init__(self, values, num_arms = 10, var_val = 1,
                 eps = 0.1, num_steps = int(1e3), alpha = None):
        
        self.values = values #true values used for generating rewards
        self.eps = eps 
        self.num_steps = num_steps
        self.num_arms = num_arms
        self.var_val = var_val
        self.opt_action = np.argmax(values)
        self.optimal_actin_count = np.zeros(num_steps)
        self.alpha = alpha

        self.wt_his = np.zeros_like(self.values)

        self.tim_step = 0 
        self.curn_reward = 0


        self.est_values = np.zeros_like(self.values)

        self.avg_reward = np.zeros(self.num_steps)
        

    def choose_action(self):

        prob = np.random.rand() 
        
        if (self.tim_step == 0) or (prob<=self.eps):
            action = np.random.choice(self.num_arms)
            
        else: 
            action = np.argmax(self.est_values)

        return action
    
    def get_reward(self, action): 
        mean = self.values[action]

        r = np.random.normal(loc=mean, scale=self.var_val)

        return r
    
    def update_action_values(self, action, reward):
        self.wt_his[action] += 1

        if self.alpha==None:
            alpha = 1/self.wt_his[action]
        else:
            alpha = self.alpha

        self.est_values[action] += alpha*(reward-self.est_values[action])

    


        
    
    def learn(self):

        for t in range(self.num_steps):
            action = self.choose_action()
            reward = self.get_reward(action=action)
            self.update_action_values(action=action, reward=reward)
            self.avg_reward[t] += reward
            self.tim_step = t
            self.optimal_actin_count[t] = (action==self.opt_action)