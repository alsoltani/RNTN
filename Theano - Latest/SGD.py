import numpy as np
import random
import time
import theano
import theano.tensor as T


class SGD:
    
    """
    Performs Stochastic Gradient Descent using 
    cost_and_gradients and update_params.
    """

    def __init__(self, model, learning_rate=1e-2, mini_batch=30):

        self.model = model
        assert self.model is not None, "Please provide a model to optimize."
        
        self.it = 0
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch
        self.cost = []

    def optimize(self, trees):
        
        """
        Runs stochastic gradient descent with model as objective.
        """
        
        m = len(trees)

        # Randomly shuffle data.
        random.shuffle(trees)

        for i in xrange(0, m - self.mini_batch + 1, self.mini_batch):

            self.it += 1

            start = time.time()
            mini_batch_data = trees[i:i + self.mini_batch]
            cost, grad = self.model.cost_and_updates(mini_batch_data)

            self.cost.append(cost)

            self.model.update_params(1./len(mini_batch_data), self.learning_rate)
            for j in self.model.dL.iterkeys():
                self.model.L[:, j] -= self.learning_rate * self.model.dL[j]

            print "Iteration %d : Cost = %.4f, Time = %.4f.\r" \
                  % (self.it, self.cost[-1], time.time() - start),