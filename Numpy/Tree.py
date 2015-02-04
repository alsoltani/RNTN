import collections
import cPickle as Pickle


class Node:
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.f_prop = False


class Tree:

    def __init__(self, tree_string, open_char='(', close_char=')'):
        tokens = []
        self.open = open_char
        self.close = close_char
        for toks in tree_string.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)

    def parse(self, tokens, parent=None):

        """
        Parsing function associated with PTB Tree Format.
        The dataset is formed of imbricated tuples (label, string_text).
        """

        assert tokens[0] == self.open, "Error in tree structure."
        assert tokens[-1] == self.close, "Error in tree structure."

        split = 2  # Position after opening character and label.
        count_open = count_close = 0

        # Non-leaf nodes
        # --------------------------
        # If word does not correspond to a leaf, tokens will start at 2nd position with a label.
        # Hence tag all these tokens with count_open = 1.

        if tokens[split] == self.open:
            count_open += 1
            split += 1

        # Find where left child and right child split.

        while count_open != count_close:
            if tokens[split] == self.open:
                count_open += 1
            if tokens[split] == self.close:
                count_close += 1
            split += 1

        # Create new node.

        node = Node(label=int(tokens[1])-1)  # tokens[1]-1 is a zero-indexed label.
        node.parent = parent

        # Leaf nodes
        # --------------------------

        if count_open == 0:
            node.word = ''.join(tokens[2:-1]).lower()  # Join tokens to form word. Convert lower-case.
            node.is_leaf = True
            return node

        # Recursively parse over children nodes.

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node


def left_traverse(root, node_function=None, args=None):

    """
    Recursive function that traverses tree
    from left to right.
    Calls node_function at each node.
    """

    node_function(root, args)
    if root.left is not None:
        left_traverse(root.left, node_function, args)
    if root.right is not None:
        left_traverse(root.right, node_function, args)


def count_words(node, words):
    if node.is_leaf:
        words[node.word] += 1


def map_words(node, word_map):
    if node.is_leaf:
        if node.word not in word_map:
            node.word = word_map['UNKNOWN']
        else:
            node.word = word_map[node.word]


def load_word_map(word_map_file='word_map.bin'):
    with open(word_map_file, 'r') as fid:
        return Pickle.load(fid)


def build_word_map(training_file='trees/train.txt', output_file='word_map.bin'):

    """
    Builds map of all words in training set
    to integer values.
    """

    print 'Building Word Map\n--------------------------'
    print "1 - Reading trees."
    with open(training_file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    print "2 - Counting words.."
    words = collections.defaultdict(int)

    for tree in trees:
        # Count recursively word occurences in tree.
        left_traverse(tree.root, node_function=count_words, args=words)

    word_map = dict(zip(words.iterkeys(), xrange(len(words))))
    word_map['UNKNOWN'] = len(words)  # Add unknown as a word.

    with open(output_file, 'w') as fid:
        Pickle.dump(word_map, fid)


def load_trees(training_file='trees/train.txt', word_map_file='word_map.bin'):

    """
    Loads training trees. Maps leaf node words to word IDs.
    """

    word_map = load_word_map(word_map_file)

    print '\nLoading training trees\n--------------------------'
    print "Reading trees.."
    with open(training_file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    for tree in trees:

        # Map recursively words in tree.
        left_traverse(tree.root, node_function=map_words, args=word_map)
    return trees
