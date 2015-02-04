import time
import os.path
import SGD as optimizer
import RNTN as RNTN
import Tree as tr
import optparse


class SentimentAnalysis:

    def __init__(self, args=None):

        """
        :param mini_batch_size: Size of mini-batch.
        :param optim_epochs: number of optimization epochs.
        :param learning_rate: SGD learning rate.
        :param output_dim: Output dimension.
        :param vect_dim: Dimension of a single word vector.

        N.B. : Pickling interconnected data such as Theano SharedVariables
        can raise recursion limit errors. Thus we directly load
        the trained model's parameters from the class and no storing file is used.
        """

        usage = "usage : %prog [options]"
        parser = optparse.OptionParser(usage=usage)

        parser.add_option("--train_file", dest="training_file", type="string", default='train.txt')
        parser.add_option("--test_file", dest="testing_file", type="string", default='test.txt')
        parser.add_option("--batch", dest="mini_batch_size", type="int", default=30)
        parser.add_option("--epochs", dest="optim_epochs", type="int", default=1)
        parser.add_option("--lr", dest="learning_rate", type="float", default=1e-2)
        parser.add_option("--classes", dest="output_dim", type="int", default=5)
        parser.add_option("--vec_dim", dest="vect_dim", type="int", default=30)
        parser.add_option("--folder", dest="data_folder", type="string",
                          default='/home/alain/Dropbox/Stage 2A/Deep Learning - RNTN/trees/')

        self.opts, self.args = parser.parse_args(args)

    def fit(self, word_map_file='word_map.bin'):

        data_train_file = self.opts.training_file

        print '================\nTRAINING\n================'

        # If the word map file does not exist, create it.
        if not os.path.exists(word_map_file):
            tr.build_word_map(data_train_file, word_map_file)
        self.word_map_file = word_map_file

        # Load trees and set the RNTN.
        self.trees = tr.load_trees(self.opts.data_folder + data_train_file, self.word_map_file)
        self.num_words = len(tr.load_word_map(self.word_map_file))

        self.rntn = RNTN.RNTN(vec_dim=self.opts.vect_dim,
                              output_dim=self.opts.output_dim,
                              num_words=self.num_words,
                              mini_batch_size=self.opts.mini_batch_size)

        self.sgd = optimizer.SGD(self.rntn, self.opts.learning_rate, self.opts.mini_batch_size)

        for e in range(self.opts.optim_epochs):

            # Fit model.
            # After the training phase, model specifications
            # are in self.rntn.stack.
            # --------------------------

            start = time.time()
            print "Running epoch %d" % e

            self.sgd.optimize(self.trees)

            end = time.time()
            print "Time per epoch : %f" % (end-start)

    def test(self):

        data_test_file = self.opts.testing_file

        print '\n================\nTESTING\n================'

        trees = tr.load_trees(self.opts.data_folder + data_test_file)

        # Load the test RNTN.
        test_rntn = RNTN.RNTN(self.opts.vect_dim,
                              self.opts.output_dim,
                              self.num_words,
                              self.opts.mini_batch_size)

        test_rntn.params = self.rntn.params
        test_rntn.V, test_rntn.W, test_rntn.b, test_rntn.W_s, test_rntn.b_s = \
            self.rntn.V, self.rntn.W, self.rntn.b, self.rntn.W_s, self.rntn.b_s
        test_rntn.L = self.rntn.L

        cost, correct, total = test_rntn.cost_and_updates(trees, 1, test=True)
        print "Cost : %f, Correct : %d/%d, Acc %f %%" % (cost, correct, total, 100 * correct / float(total))


if __name__ == '__main__':

    training_file = 'train.txt'
    test_file = 'test.txt'

    SA = SentimentAnalysis()
    SA.fit()
    if SA.test:
        SA.test()
