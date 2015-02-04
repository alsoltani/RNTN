import numpy as np
import theano
import theano.tensor as T
import collections

ran = np.random.randn
floatX = theano.config.floatX
shared = theano.shared


# --------------------------
# Theano functions
# --------------------------

def _recombination():

    """Returns parent activation via children activation."""

    lr = T.vector('Stacked left-right children activation')
    W = T.matrix('Weight matrix')
    b = T.vector('Bias vector')
    V = T.tensor3('Tensor')

    new_act = T.tanh(T.dot(W, lr) + b + T.tensordot(V, T.outer(lr, lr), axes=([1, 2], [0, 1])))

    # Recombination function for data propagation.
    # Allow input downcast to make sure float64-data can be processed.

    return theano.function([lr, V, W, b], new_act, allow_input_downcast=True)


def _probabilities():

    """Returns posterior probabilities given parent activation."""

    parent_activation = T.vector('Parent activation')
    W_s = T.matrix('Soft-max weight matrix')
    b_s = T.vector('Soft-max bias vector')

    prob = T.dot(W_s, parent_activation) + b_s
    prob -= T.max(prob)
    prob = T.exp(prob)
    prob /= T.sum(prob)

    # Compute probabilities.
    return theano.function([parent_activation, W_s, b_s], prob, allow_input_downcast=True)


def _softmax_node_error():

    """Pre-computes softmax node error given distribution difference (target - real).
    The Hadamard product is added afterwards. 
    """

    diff = T.vector('Distribution difference')
    W_s = T.matrix('Soft-max weight matrix')

    softmax_node_error = T.dot(W_s.T, diff)
    return theano.function([diff, W_s], softmax_node_error, allow_input_downcast=True)


def _add_penalization_term():

    """Add penalization term to the cost."""

    cost = T.scalar('cost')
    V = T.tensor3('V')
    W = T.matrix('W')
    W_s = T.matrix('W_s')
    rho = T.scalar('rho')

    new_cost = cost + (rho / 2) * (T.sum(V ** 2) + T.sum(W ** 2) + T.sum(W_s ** 2))
    return theano.function([cost, V, W, W_s, rho], new_cost, allow_input_downcast=True)


def _prop_error():

    """Back-propagates error and updates gradients."""

    lr = T.vector('Stacked activation')
    node_error = T.vector('Node error')
    V = T.tensor3('Tensor')
    W = T.matrix('Weight Matrix')
    dV = T.tensor3('Tensor Gradient')
    dW = T.matrix('Weight Matrix Gradient')
    db = T.vector('Bias Gradient')

    outer = T.outer(node_error, lr)

    # Back-propagate error. Update gradients.

    new_dV = dV + (T.outer(lr, lr)[:, :, None] * node_error).T
    new_dW = dW + outer
    new_db = db + node_error

    child_node_error = T.dot(W.T, node_error) + T.tensordot(V.transpose((0, 2, 1)) + V, outer.T, axes=([1, 0], [0, 1]))

    return theano.function([node_error, lr, V, W, dV, dW, db],
                           (child_node_error, new_dV, new_dW, new_db), allow_input_downcast=True)


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
        # Size : (single-word dimension, size of vocabulary).
        # L is trained jointly with the comp. models :
        # its update amounts to add, for each word, the errors
        # obtained for each associated leaf.

        self.L = 0.01 * ran(self.vec_dim, self.num_words).astype(floatX)

        # Neural Tensor Layer weights.
        # --------------------------
        # V: tensor that defines multiple bilinear forms.
        # W, b : weight and bias matrices.

        self.V = 0.01 * ran(self.vec_dim, 2 * self.vec_dim, 2 * self.vec_dim).astype(floatX)
        self.W = 0.01 * ran(self.vec_dim, 2 * self.vec_dim).astype(floatX)
        self.b = np.zeros(self.vec_dim).astype(floatX)

        # Softmax weights.
        # --------------------------
        # W_s, b_s : sentiment classification weight and bias matrices.

        self.W_s = 0.01 * ran(self.output_dim, self.vec_dim).astype(floatX)
        self.b_s = np.zeros(self.output_dim).astype(floatX)
        self.params = [self.L, self.V, self.W, self.b, self.W_s, self.b_s]

        # Gradients.
        # --------------------------

        self.dV = np.zeros((self.vec_dim, 2 * self.vec_dim, 2 * self.vec_dim))
        self.dW = np.zeros((self.vec_dim, 2 * self.vec_dim))
        self.db = np.zeros(self.vec_dim)
        self.dW_s = np.zeros((self.output_dim, self.vec_dim))
        self.db_s = np.zeros(self.output_dim)

        # As L is jointly trained with the above parameters, we need a "gradient" for L.
        # This comes in the form of a dictionary.

        self.dL = collections.defaultdict(self.default_vec)

        # Theano functions.
        # --------------------------

        self.recombination = _recombination()
        self.probabilities = _probabilities()
        self.softmax_node_error = _softmax_node_error()
        self.prop_error = _prop_error()
        self.add_penalization_term = _add_penalization_term()

    def cost_and_updates(self, mini_batch_data, test=False):

        """
        Computes cost and gradients for mini-batch data.
        Data is propagated and back-propagated in each tree.

        :param mini_batch_data: List of data pieces (i.e. trees).
        :param test:
        :return: Cost, Gradients of W, W_s, b, b_s, L.
        """

        cost = correct = total = 0.0

        # Set gradients to zero.
        # --------------------------
        self.L, self.V, self.W, self.b, self.W_s, self.b_s = self.params

        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dW_s[:] = 0
        self.db_s[:] = 0

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

        cost = self.add_penalization_term(cost, self.V, self.W, self.W_s, self.rho) * scale

        # Return scaled cost and gradients for parameter update.
        # --------------------------

        g_params = [self.dL, scale*(self.dV + self.rho * self.V),
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

            # Hidden activations at leaves are L elements.
            node.h_activation = self.L[:, node.word]
            node.f_prop = True

        else:

            # Propagate recursively through the tree.
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

            lr = np.concatenate([node.left.h_activation, node.right.h_activation])
            node.h_activation = self.recombination(lr, self.V, self.W, self.b)

        # Compute classification probabilities.
        # --------------------------

        node.prob = self.probabilities(node.h_activation, self.W_s, self.b_s)
        node.f_prop = True

        return cost - np.log(node.prob[node.label]), correct + (np.argmax(node.prob) == node.label), total + 1

    def back_prop(self, node, error=None):

        """
        Backward propagation in node.
        """

        # Clear node.
        node.f_prop = False

        # Compute soft-max errors.
        # --------------------------

        diff_prob = node.prob
        diff_prob[node.label] -= 1.0
        softmax_node_error = self.softmax_node_error(diff_prob, self.W_s)
        self.dW_s += np.outer(diff_prob, node.h_activation)
        self.db_s += diff_prob

        if error is not None:
            softmax_node_error += error
        softmax_node_error *= (1 - node.h_activation ** 2)

        # Soft-max errors are used to update L gradient.

        if node.is_leaf:
            self.dL[node.word] += softmax_node_error
            return

        # Hidden gradients.
        # --------------------------

        if not node.is_leaf:

            lr = np.concatenate([node.left.h_activation, node.right.h_activation])
            softmax_node_error, self.dV, self.dW, self.db = self.prop_error(softmax_node_error, lr, self.V, self.W, self.dV, self.dW, self.db)

            self.back_prop(node.left, softmax_node_error[:self.vec_dim])
            self.back_prop(node.right, softmax_node_error[self.vec_dim:])

    def update_params(self, scale, update):
        """
        Updates parameters as
        p := p + scale * update."""

        self.params[1:] = [P+scale*dP for P, dP in zip(self.params[1:], update[1:])]

        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale * dL[j]