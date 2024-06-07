import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, epsilons, filename, x=None, window=5, xlabel= 'Game', ylabel='Score'): 
    plt.figure()
    n,m = np.shape(scores)  

    x = range(1,m+1)
    for i, eps in enumerate(epsilons):
        plt.plot(x, scores[i,:],label = f'$\epsilon$ = {eps}')
    
    

    plt.ylabel(ylabel)       
    plt.xlabel(xlabel)
    plt.legend()                     
    #plt.grid()
    plt.savefig(filename)
    