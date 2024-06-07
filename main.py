import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import BanditProblem
from utils import plotLearning

if __name__ == '__main__':
    num_steps   = 1000
    num_runs    = 2000
    num_actions = 10
    epsilons = [0.0, 0.1]
    avg_rewards = np.zeros((len(epsilons),num_steps))
    opt_actions = np.zeros((len(epsilons),num_steps))


    for i, eps in enumerate(epsilons):
        for t in range(num_runs): 
            if eps ==0: 
                values = np.random.normal(loc=0, scale=1, size=(num_actions))
                values[0] = 5
            else: 
                values = np.random.normal(loc=0, scale=1, size=(num_actions))
            alg = BanditProblem(values=values, num_arms= num_actions, 
                                eps=eps, num_steps=num_steps, alpha=0.1)
            alg.learn()
            avg_rewards[i,:] += alg.avg_reward
            opt_actions[i,:] += alg.optimal_actin_count

    avg_rewards/=num_runs
    opt_actions/=num_runs

    plotLearning(avg_rewards,epsilons=epsilons,
                  filename= 'average_rewards.png', 
                  xlabel='Steps', ylabel='Average rewards')
    plotLearning(opt_actions,epsilons=epsilons,
                    filename= 'optimal_actions.png', 
                    xlabel='Steps', ylabel='optimal actions')

