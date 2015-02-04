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

        for i in xrange(0, m-self.mini_batch+1, self.mini_batch):

            start = time.time()

            self.it += 1

            mb_data = trees[i:i+self.mini_batch]

            cost, grad = self.model.cost_and_updates(mb_data)
            update = grad
            scale = -self.learning_rate

            # update params
            self.model.update_params(scale, update)

            self.cost.append(cost)
            if self.it % 1 == 0:
                print "Iter %d : Cost=%.4f, Time=%.4f." \
                      % (self.it, cost, time.time() - start)