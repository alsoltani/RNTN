import inspect
import cPickle as pickle
import time
import os.path
import SGD as optimizer
import RNTN as nnet
import Tree as tr
import optparse


class SentimentAnalysis:
    def __init__(self,
                 data_folder,
                 mini_batch_size=30,
                 optim_epochs=1,
                 learning_rate=1e-2,

                 output_dim=5,
                 vect_dim=30):

        """
        :param mini_batch_size: Size of mini-batch.
        :param optim_algorithm: learning algorithm performed.
        :param optim_epochs: number of optimization epochs.
        :param learning_rate: SGD learning rate.
        :param output_dim: Output dimension.
        :param vect_dim: Dimension of a single word vector.
        """

        self.mini_batch_size = mini_batch_size
        self.optim_epochs = optim_epochs
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.vect_dim = vect_dim
        self.data_folder = data_folder

        frame = inspect.currentframe()
        self.args, _, _, self.values = inspect.getargvalues(frame)

    def fit(self, training_file, output_file, word_map_file='word_map.bin'):

        if not os.path.exists(word_map_file):
            tr.build_word_map(training_file, word_map_file)
        self.word_map_file = word_map_file

        self.trees = tr.load_trees(self.data_folder + training_file, self.word_map_file)
        self.num_words = len(tr.load_word_map(self.word_map_file))

        self.rntn = nnet.RNTN(self.vect_dim, self.output_dim, self.num_words, self.mini_batch_size)
        self.sgd = optimizer.SGD(self.rntn, self.learning_rate, self.mini_batch_size)

        for e in range(self.optim_epochs):

            # Fit model
            # --------------------------

            start = time.time()
            print "Running epoch %d" % e

            self.sgd.optimize(self.trees)

            end = time.time()
            print "Time per epoch : %f" % (end-start)

            # Save model specifications
            with open(output_file, 'w+') as fid:
                pickle.dump([(i, self.values[i]) for i in self.args][1:], fid)
                pickle.dump(self.sgd.cost_list, fid)
                pickle.dump(self.rntn.stack, fid)

    def test(self, specs_file, data_test_file):

        trees = tr.load_trees(self.data_folder + data_test_file)
        assert specs_file is not None, "Please provide a model to test."

        with open(specs_file, 'r') as fid:
            _ = pickle.load(fid)
            _ = pickle.load(fid)
            rntn = nnet.RNTN(self.output_dim, self.num_words, self.mini_batch_size)
            rntn.stack = pickle.load(fid)

        print "Testing..."
        cost, correct, total = rntn.cost_and_gradients(trees, test=True)
        print "Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total))


def run(args=None):

    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Folder containing the PTB Dataset.
    parser.add_option("--folder", dest="data_folder", type="string", default=os.path.dirname(os.getcwd()) + "/trees/")
    parser.add_option("--train_file", dest="training_file", type="string", default="train.txt")
    parser.add_option("--specs_file", dest="specs_file", type="string", default="model_specifications.bin")
    parser.add_option("--test_file", dest="test_file", type="string", default="test.txt")

    parser.add_option("--batch_size", dest="mini_batch_size", type="int", default=30)
    parser.add_option("--max_epochs", dest="optim_epochs", type="int", default=1)
    parser.add_option("--learning_rate", dest="learning_rate", type="float", default=1e-2)
    parser.add_option("--n_classes", dest="output_dim", type="int", default=5)
    parser.add_option("--word_dim", dest="vect_dim", type="int", default=30)

    (opts, args) = parser.parse_args(args)

    sa = SentimentAnalysis(
        data_folder=opts.data_folder,
        mini_batch_size=opts.mini_batch_size,
        optim_epochs=opts.optim_epochs,
        learning_rate=opts.learning_rate,
        output_dim=opts.output_dim,
        vect_dim=opts.vect_dim)

    sa.fit(opts.training_file, opts.specs_file)
    sa.test(opts.specs_file, opts.test_file)

if __name__ == '__main__':
    run()