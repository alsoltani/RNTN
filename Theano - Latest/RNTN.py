import numpy as np
import theano
import theano.tensor as T
import collections

ran = np.random.randn
floatX = theano.config.floatX
shared = theano.shared


class RNTN:

    """
    Implements a Recursive Neural Tensor Network.
    """

    def __init__(self,
                 vec_dim,
                 output_dim,
                 num_words,
                 mini_batch_size=30,
                 rho=1e-6):

        """
        :param vec_dim: Dimension of a single word vector.
        :param output_dim: Output dimension.
        :param num_words: Number of different words.
        :param mini_batch_size: Size of mini-batch.
        :param rho: L2 penalization coefficient in the cross-entropy error.
        """

        self.vec_dim = vec_dim
        self.output_dim = output_dim
        self.num_words = num_words
        self.mini_batch_size = mini_batch_size
        self.default_vec = lambda: np.zeros(self.vec_dim).astype(floatX)
        self.rho = rho

        # Embedding matrix L.
        # --------------------------
        # Size : (single-word dimension, number of words).
        # L is trained jointly with the comp. models.

        self.L = 0.01 * ran(self.vec_dim, self.num_words).astype(floatX)

        # Neural Tensor Layer weights.
        # --------------------------
        # V is the tensor that defines multiple bilinear forms.
        # W, b are classical-RNN weight and bias matrices.

        self.V = shared(0.01 * ran(self.vec_dim, 2 * self.vec_dim, 2 * self.vec_dim)
                        .astype(floatX), name='V', borrow=True)
        self.W = shared(0.01 * ran(self.vec_dim, 2 * self.vec_dim)
                        .astype(floatX), name='W', borrow=True)
        self.b = shared(np.zeros(self.vec_dim).astype(floatX), name='b', borrow=True)

        # Softmax weights.
        # --------------------------
        # W_s, b_s are the sentiment classification weight and bias matrices.

        self.W_s = shared(0.01 * ran(self.output_dim, self.vec_dim)
                          .astype(floatX), name='W_s', borrow=True)
        self.b_s = shared(np.zeros(self.output_dim).astype(floatX), name='b_s', borrow=True)
        self.params = [self.V, self.W, self.b, self.W_s, self.b_s]  # Only shared variables

        # Gradients.
        # --------------------------

        self.np_dV = np.empty((self.vec_dim, 2 * self.vec_dim, 2 * self.vec_dim))
        self.np_dW = np.empty((self.vec_dim, 2 * self.vec_dim))
        self.np_db = np.empty(self.vec_dim)
        self.np_dW_s = np.empty((self.output_dim, self.vec_dim))
        self.np_db_s = np.empty(self.output_dim)

        self.dV = shared(self.np_dV.astype(floatX), name='dV', borrow=True)
        self.dW = shared(self.np_dW.astype(floatX), name='dW', borrow=True)
        self.db = shared(self.np_db.astype(floatX), name='db', borrow=True)
        self.dW_s = shared(self.np_dW_s.astype(floatX), name='dW_s', borrow=True)
        self.db_s = shared(self.np_db_s.astype(floatX), name='db_s', borrow=True)

        # As L is jointly trained with the above parameters, we need a "gradient" for L.
        # This comes in the form of a dictionary
        self.dL = collections.defaultdict(self.default_vec)

        # Theano variables for the computational graph.
        # --------------------------

        self.p_a = T.vector('Parent activation')
        self.lr = T.vector('Stacked activation')

        self.prob = T.vector('Probabilities')
        self.diff = T.vector('Distribution differences')
        self.node_error = T.vector('Soft-max node error')
        self.label = T.iscalar('Label')
        self.cost = T.scalar('Cost')
        self.rate = T.scalar('Learning rate')
        self.scale = T.scalar('Batch scale')

        prob = T.dot(self.W_s, self.p_a) + self.b_s
        prob -= T.max(prob)
        prob = T.exp(prob)
        prob /= T.sum(prob)
        
        outer = T.outer(self.node_error, self.lr)

        # Recombination
        # --------------------------
        # Returns parent activation via children activation.

        self.recombination = theano.function([self.lr], T.tanh(T.dot(self.W, self.lr)
                                                          + self.b + T.tensordot(self.V,
                                                                                 T.outer(self.lr, self.lr),
                                                                                 axes=([1, 2], [0, 1]))), 
                                             allow_input_downcast=True)

        # Probabilities
        # --------------------------
        # Returns posterior probabilities given parent activation.

        self.probabilities = theano.function([self.p_a], prob, 
                                             allow_input_downcast=True)

        # Soft-max node error
        # --------------------------
        # Pre-computes softmax node error given distribution difference (target - real).
        # The Hadamard product is added afterwards

        updates_1 = collections.OrderedDict()
        updates_1[self.dW_s] = self.dW_s + T.outer(self.diff, self.p_a)
        updates_1[self.db_s] = self.db_s + self.diff

        self.softmax_node_error = theano.function([self.diff, self.p_a], T.dot(self.W_s.T, self.diff),
                                                  updates=updates_1,
                                                  allow_input_downcast=True)

        # Soft-max node error
        # --------------------------
        #Add penalization term to the cost.

        self.add_penalization_term = theano.function([self.cost],
                                                     self.cost + (self.rho / 2) * (T.sum(self.V ** 2) + T.sum(self.W ** 2) 
                                                                                   + T.sum(self.W_s ** 2)), 
                                                     allow_input_downcast=True)

        # Prop error
        # --------------------------
        # Back-propagates error and updates gradients.

        updates_2 = collections.OrderedDict()
        updates_2[self.dV] = self.dV + (T.outer(self.lr, self.lr)[:, :, None] * self.node_error).T
        updates_2[self.dW] = self.dW + outer
        updates_2[self.db] = self.db + self.node_error

        self.prop_error = theano.function([self.node_error, self.lr], T.dot(self.W.T, self.node_error) + T.tensordot(self.V.transpose((0, 2, 1)) + self.V,
                                                                                                         outer.T, axes=([1, 0], [0, 1])),
                                      updates=updates_2,
                                      allow_input_downcast=True)

        # Update params
        # --------------------------
        # Updates all weights & biases during gradient descent.

        updates_3 = collections.OrderedDict()
        updates_3[self.V] = self.V - self.rate * self.scale * (self.dV + self.rho * self.V)
        updates_3[self.W] = self.W - self.rate * self.scale * (self.dW + self.rho * self.W)
        updates_3[self.b] = self.b - self.rate * self.scale * self.db
        updates_3[self.W_s] = self.W_s - self.rate * self.scale * (self.dW_s + self.rho * self.W_s)
        updates_3[self.b_s] = self.db_s - self.rate * self.scale * self.db_s

        self.update_params = theano.function([self.scale, self.rate], self.scale, updates=updates_3,
                                             allow_input_downcast=True)

    def cost_and_updates(self, mini_batch_data, test=False):

        """
        Computes cost and gradients for mini-batch data.
        Data is propagated and back-propagated in each tree.

        :param mini_batch_data: List of data pieces (i.e. trees).
        :return: Cost, Gradients of W, W_s, b, b_s, L.
        """

        cost = correct = total = 0.0

        # Set gradients to zero.
        # --------------------------
        self.V, self.W, self.b, self.W_s, self.b_s = self.params
        
        self.dV.set_value(np.zeros(self.np_dV.shape).astype(floatX))
        self.dW.set_value(np.zeros(self.np_dW.shape).astype(floatX))
        self.db.set_value(np.zeros(self.np_db.shape).astype(floatX))
        self.dW_s.set_value(np.zeros(self.np_dW_s.shape).astype(floatX))
        self.db_s.set_value(np.zeros(self.np_db_s.shape).astype(floatX))

        self.dL = collections.defaultdict(self.default_vec)

        # Propagate data in each tree in a mini-batch fashion.
        # --------------------------

        for tree in mini_batch_data:
            c, corr, tot = self.forward_prop(tree.root)
            cost += c
            correct += corr
            total += tot
        if test:

            # When testing, we simply return scaled cost, correct, total
            # to compute the mis-classification rate.

            return (1./len(mini_batch_data))*cost, correct, total

        # Back-propagate data in each tree.
        # --------------------------
        for tree in mini_batch_data:
            self.back_prop(tree.root)

        # Scale cost and gradients by mini-bach size.
        # --------------------------
        scale = (1./self.mini_batch_size)
        for v in self.dL.itervalues():
            v *= scale

        # Add L2 Regularization term.
        # --------------------------

        cost = self.add_penalization_term(cost) * scale

        # Update parameters.
        # --------------------------

        g_params = [scale*(self.dV + self.rho * self.V),
                    scale * (self.dW + self.rho * self.W), scale * self.db,
                    scale * (self.dW_s + self.rho * self.W_s), scale * self.db_s]

        return cost, g_params

    def forward_prop(self, node):

        """
        Forward propagation at node.
        :return: (Cross-entropy cost,
        Number of correctly classified items,
        Number of classified items).
        """

        cost, correct, total = 0.0, 0.0, 0.0

        if node.is_leaf:

            # Hidden activations at leaves are L elements
            node.h_activation = self.L[:, node.word]
            node.f_prop = True

        else:
            if not node.left.f_prop:
                c, corr, tot = self.forward_prop(node.left)
                cost += c
                correct += corr
                total += tot

            if not node.right.f_prop:
                c, corr, tot = self.forward_prop(node.right)
                cost += c
                correct += corr
                total += tot

            # Compute parent vector.
            # --------------------------

            lr = np.concatenate([node.left.h_activation, node.right.h_activation])\
                .astype(floatX)
            node.h_activation = self.recombination(lr)

        # Compute classification labels via softmax.
        # --------------------------

        node.prob = self.probabilities(node.h_activation)
        node.f_prop = True

        return cost - np.log(node.prob[node.label]), correct + (np.argmax(node.prob) == node.label), total + 1

    def back_prop(self, node, error=None):

        """
        Backward propagation in node.
        """

        # Clear node.
        node.f_prop = False

        # Softmax gradients.
        # --------------------------

        diff_prob = node.prob
        diff_prob[node.label] -= 1.0
        softmax_node_error = self.softmax_node_error(diff_prob.astype(floatX), node.h_activation)

        if error is not None:

            # To back-propagate error recursively
            softmax_node_error += error

        softmax_node_error *= (1 - node.h_activation ** 2)

        # Update L at leaf nodes.
        if node.is_leaf:
            self.dL[node.word] += softmax_node_error
            return

        # Hidden gradients.
        # --------------------------

        if not node.is_leaf:

            lr = np.concatenate([node.left.h_activation, node.right.h_activation])
            softmax_node_error = self.prop_error(softmax_node_error, lr)

            self.back_prop(node.left, softmax_node_error[:self.vec_dim])
            self.back_prop(node.right, softmax_node_error[self.vec_dim:])