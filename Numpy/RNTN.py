import numpy as np
import collections
np.seterr(over='raise', under='raise')


class RNTN:

    def __init__(self, vect_dim, output_dim, num_words,
                 mini_batch_size=30,
        rho=1e-6):

        """
        :param vect_dim: Dimension of a single word vector.
        :param output_dim: Output dimension.
        :param num_words: Number of different words.
        :param mini_batch_size: Size of mini-batch.
        :param rho: L2 penalization coefficient in the cross-entropy error.
        """

        self.vect_dim = vect_dim
        self.output_dim = output_dim
        self.num_words = num_words
        self.default_vec = lambda: np.zeros((vect_dim,))
        self.mini_batch_size = mini_batch_size
        self.rho = rho

        # Embedding matrix L.
        # --------------------------
        # Size : (single-word dimension, number of words).
        # L is seen as a parameter that is trained jointly with the comp. models.

        self.L = 0.01 * np.random.randn(self.vect_dim, self.num_words)

        # Neural Tensor Layer weights.
        # --------------------------
        # V is the tensor that defines multiple bilinear forms.
        self.V = 0.01 * np.random.randn(self.vect_dim, 2 * self.vect_dim, 2 * self.vect_dim)

        # W, b are classical-RNN weight and bias matrices.
        self.W = 0.01 * np.random.randn(self.vect_dim, 2 * self.vect_dim)
        self.b = np.zeros(self.vect_dim)

        # Softmax weights.
        # --------------------------
        # W_s, b_s are the sentiment classification weight and bias matrices.

        self.W_s = 0.01 * np.random.randn(self.output_dim, self.vect_dim)
        self.b_s = np.zeros(self.output_dim)
        self.stack = [self.L, self.V, self.W, self.b, self.W_s, self.b_s]

        # Gradients.
        # --------------------------
        self.dV = np.zeros((self.vect_dim, 2 * self.vect_dim, 2 * self.vect_dim))
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.vect_dim)
        self.dW_s = np.zeros(self.W_s.shape)
        self.db_s = np.zeros(self.output_dim)

    def cost_and_gradients(self, mini_batch_data, test=False):

        """
        Computes cost and gradients for mini-batch data.
        Data is propagated and back-propagated in each tree.

        :param mini_batch_data: List of data pieces (i.e. trees).
        :return: Cost, Gradients of W, W_s, b, b_s, L.
        """

        cost, correct, total = 0.0, 0.0, 0.0
        self.L, self.V, self.W, self.b, self.W_s, self.b_s = self.stack

        # Set gradients to zero.

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
        cost += self.rho / 2 * np.sum(self.V ** 2)
        cost += self.rho / 2 * np.sum(self.W ** 2)
        cost += self.rho / 2 * np.sum(self.W_s ** 2)

        return scale*cost, [self.dL, scale*(self.dV + self.rho * self.V),
                            scale * (self.dW + self.rho * self.W), scale * self.db,
                            scale * (self.dW_s + self.rho * self.W_s), scale * self.db_s]

    def forward_prop(self, node):

        """
        Forward propagation at node.
        :return: (Cross-entropy cost,
        Number of correctly classified items,
        Number of classified items).
        """

        cost, correct, total = 0.0, 0.0, 0.0

        if node.is_leaf:
            # Hidden activations at leaves are occurences of self.word.
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

            # Stack left-right children word vectors. Compute matrix operations for parent vector.
            lr = np.hstack([node.left.h_activation, node.right.h_activation])
            node.h_activation = np.dot(self.W, lr) + self.b
            node.h_activation += np.tensordot(self.V, np.outer(lr, lr), axes=([1, 2], [0, 1]))

            # Compute parent vector.
            node.h_activation = np.tanh(node.h_activation)

        # Compute classification labels via softmax.
        node.probs = np.dot(self.W_s, node.h_activation) + self.b_s
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)
        node.f_prop = True

        return cost - np.log(node.probs[node.label]), correct + (np.argmax(node.probs) == node.label), total + 1

    def back_prop(self, node, error=None):

        """
        Backward propagation in node.
        """

        # Clear node.
        node.f_prop = False

        # Softmax gradients.
        # --------------------------
        softmax_node_error = node.probs  # Predicted distribution
        softmax_node_error[node.label] -= 1.0  # Targeted distribution equals 1 for node.label, else 0.

        self.dW_s += np.outer(softmax_node_error, node.h_activation)
        self.db_s += softmax_node_error
        softmax_node_error = np.dot(self.W_s.T, softmax_node_error)

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
            lr = np.hstack([node.left.h_activation, node.right.h_activation])  # Left-right stacked activation
            outer = np.outer(softmax_node_error, lr)

            self.dV += (np.outer(lr, lr)[:, :, None] * softmax_node_error).T
            self.dW += outer
            self.db += softmax_node_error

            # Compute error for children.
            softmax_node_error = np.dot(self.W.T, softmax_node_error)
            softmax_node_error += np.tensordot(self.V.transpose((0, 2, 1)) + self.V,
                                               outer.T, axes=([1, 0], [0, 1]))
            self.back_prop(node.left, softmax_node_error[:self.vect_dim])
            self.back_prop(node.right, softmax_node_error[self.vect_dim:])

    def update_params(self, scale, update):

        self.stack[1:] = [P+scale*dP for P, dP in zip(self.stack[1:], update[1:])]

        # Update dictionary separately.
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:, j] += scale*dL[j]