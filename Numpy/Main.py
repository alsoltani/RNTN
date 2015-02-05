import inspect
import cPickle as pickle
import time
import os.path
import SGD
import RNTN
import Tree as tr
import optparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np


class BagOfWordsSentimentAnalysis:
    def __init__(self,
                 data_folder, model):

        self.data_folder = data_folder
        self.model = model

        frame = inspect.currentframe()
        self.args, _, _, self.values = inspect.getargvalues(frame)

        self.vocabulary = []
        self.training_texts = []
        self.training_labels = []
        self.test_texts = []
        self.test_labels = []

    def fit(self, training_file):

        with open(self.data_folder + training_file, 'r') as fid:
            for l in fid.readlines():

                current_tree = tr.Tree(l)
                self.vocabulary.append(current_tree.vocabulary)
                self.training_texts.append(current_tree.text)
                self.training_labels.append(current_tree.label)

        self.vocabulary = list(set([item for sublist in self.vocabulary for item in sublist]))
        self.cv = CountVectorizer(vocabulary=self.vocabulary)

        print 'Vectorizing training data...'
        self.training_mat = self.cv.fit_transform(self.training_texts).toarray()

        print 'Fitting the model...'
        self.model.fit(self.training_mat, np.asarray(self.training_labels))

    def test(self, test_file):

        with open(self.data_folder + test_file, 'r') as fid:
            for l in fid.readlines():

                current_tree = tr.Tree(l)
                self.test_texts.append(current_tree.text)
                self.test_labels.append(current_tree.label)

        print 'Vectorizing test data...'
        self.test_mat = self.cv.transform(self.test_texts).toarray()

        print 'Predicting labels...'

        x = self.model.predict(self.test_mat)
        nb_well_classified_items = len(self.test_labels) - np.linalg.norm(x - np.asarray(self.test_labels), ord=0)

        print "\nCorrectly labelled : %d/%d \nAccuracy : %f." % (nb_well_classified_items, len(self.test_labels),
                                                                nb_well_classified_items / float(len(self.test_labels)))


class RNTNSentimentAnalysis:
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

        self.rntn = RNTN.RNTN(self.vect_dim, self.output_dim, self.num_words, self.mini_batch_size)
        self.sgd = SGD.SGD(self.rntn, self.learning_rate, self.mini_batch_size)

        for e in range(self.optim_epochs):

            # Fit model
            # --------------------------

            start = time.time()
            print "Running epoch %d" % e

            self.sgd.optimize(self.trees)

            end = time.time()
            print "\nTime per epoch : %f" % (end-start)

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
            rntn = RNTN.RNTN(self.output_dim, self.num_words, self.mini_batch_size)
            rntn.stack = pickle.load(fid)

        print "Testing..."
        cost, correct, total = rntn.cost_and_gradients(trees, test=True)
        print "\nCost : :%f \nCorrectly labelled : %d/%d \nAccuracy : %f." % (cost, correct, total, correct / float(total))


def run(args=None):

    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Folder containing the PTB Dataset.
    parser.add_option("--rntn", dest="rntn", default=True)
    parser.add_option("--bayes", dest="nb", default=False)
    parser.add_option("--folder", dest="data_folder", type="string", default=os.path.dirname(os.getcwd()) + "/trees/")
    parser.add_option("--train_file", dest="training_file", type="string", default="train.txt")
    parser.add_option("--specs_file", dest="specs_file", type="string", default="model_specifications.bin")
    parser.add_option("--test_file", dest="test_file", type="string", default="test.txt")

    parser.add_option("--batch_size", dest="mini_batch_size", type="int", default=10)
    parser.add_option("--max_epochs", dest="optim_epochs", type="int", default=1)
    parser.add_option("--learning_rate", dest="learning_rate", type="float", default=1e-2)
    parser.add_option("--n_classes", dest="output_dim", type="int", default=5)
    parser.add_option("--word_dim", dest="vect_dim", type="int", default=30)

    (opts, args) = parser.parse_args(args)

    if opts.nb:

        print "\n==========================="
        print "NAIVE BAYES"
        print "===========================\n"

        nb_sa = BagOfWordsSentimentAnalysis(
            data_folder=opts.data_folder,
            model=MultinomialNB())
        nb_sa.fit(opts.training_file)
        nb_sa.test(opts.test_file)

    """

    print "\n==========================="
    print "SVM"
    print "===========================\n"

    svm_sa = BagOfWordsSentimentAnalysis(
        data_folder=opts.data_folder,
        model=SVC())
    svm_sa.fit(opts.training_file)
    svm_sa.test(opts.test_file)"""

    if opts.rntn:

        print "\n==========================="
        print "RNTN"
        print "===========================\n"

        rntn_sa = RNTNSentimentAnalysis(
            data_folder=opts.data_folder,
            mini_batch_size=opts.mini_batch_size,
            optim_epochs=opts.optim_epochs,
            learning_rate=opts.learning_rate,
            output_dim=opts.output_dim,
            vect_dim=opts.vect_dim)

        rntn_sa.fit(opts.training_file, opts.specs_file)
        rntn_sa.test(opts.specs_file, opts.test_file)

if __name__ == '__main__':
    run()