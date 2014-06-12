import numpy
from collections import Counter
from decision_tree import DTNode

def build_tree(feature_vectors, labels):
    m_examples = len(labels)
    examples_matrix = numpy.concatenate((feature_vectors,\
                      numpy.reshape(labels, (m_examples, 1))), axis=1)

    S = range(0, m_examples)
    root = DTNode(S, labels)
    root.name = 'root'
    impure_leaves = []

    if root.is_impure():
       impure_leaves.append(root)

    while impure_leaves:
        # perform BFS traversal for construction of the tree
        u = impure_leaves.pop(0)

        # find splitting rule for impure leaves
        examples_subset = examples_matrix[list(u.subset), :]
        (decision_function, (f ,t)) = find_splitting_rule(examples_subset)
        u.decision_question = decision_function
        u.name = "x(%d) <= %.2f?" % (f,t)

        # change the impure leaf into an internal decision node with 2 children
        yes_subset = set([i if u.decision_question(examples_matrix[i,:])\
                            else -1 for i in u.subset]) - set([-1])
        no_subset = u.subset - yes_subset
        yes_labels = examples_matrix[list(yes_subset), -1]
        no_labels = examples_matrix[list(no_subset), -1]

        u.yes_tree = DTNode(yes_subset, yes_labels)
        u.no_tree = DTNode(no_subset, no_labels)

        # repeat until there are no remaining impure leaves
        if u.yes_tree.is_impure():
            impure_leaves.append(u.yes_tree)
        else:
            u.yes_tree.name = u.yes_tree.predict()

        if u.no_tree.is_impure():
            impure_leaves.append(u.no_tree)
        else:
            u.no_tree.name = u.no_tree.predict()

    return root

def find_splitting_rule(examples_subset):
    feature_vectors = examples_subset[:, 0:-1]
    example_labels = examples_subset[:, (-1)]

    def calculate_entropy(labels):
        frequency_table = Counter(list(labels))
        total_frequency = float(sum(frequency_table.values()))
        relative_frequencies = \
            numpy.array(frequency_table.values()) / total_frequency
        probabilities_of_possible_values = relative_frequencies
        entropy = -(numpy.dot(probabilities_of_possible_values,\
                  numpy.log(probabilities_of_possible_values)))
 
        return entropy

    prior_entropy = calculate_entropy(example_labels)
    information_gain_table = dict([])
    n_dimensions = feature_vectors.shape[1]

    for f in range(0, n_dimensions):
        threshold_intervals_concatenated = sorted(feature_vectors[:, f])

        if len(threshold_intervals_concatenated) == 1:
            thresholds = [threshold_intervals_concatenated[0]]
        else:
        # as a convention, pick midpoints of values along the current dimension
                                                              #     [a, b, c]
            slide_left = threshold_intervals_concatenated[:-1]# ..., b, c]
            slide_right = threshold_intervals_concatenated[1:]#     [a, b, ...
            mid = lambda left, right : (left+right)/2         #[(a+b)/2,(b+c)/2]
            thresholds = [mid(a,b) for (a,b) in zip(slide_right, slide_left)]

        # find conditional entropy for (f, t) pairs
        for t in thresholds:

            # consider when the proposition Z(f,t): "x_f <= t" is true
            yes_examples = filter(lambda x : x[f] <= t, examples_subset)

            if len(yes_examples) > 0:
                yes_examples_labels = numpy.array(yes_examples)[:, (-1)]
                conditional_entropy_of_X_when_Z_is_true =\
                        calculate_entropy(yes_examples_labels)#\
            else:
                conditional_entropy_of_X_when_Z_is_true = 0

            # consider when the proposition Z(f,t): "x_f < t" is false
            no_examples = filter(lambda x : x[f] > t, examples_subset)

            if len(no_examples) > 0:
                no_examples_labels = numpy.array(no_examples)[:, (-1)]
                conditional_entropy_of_X_when_Z_is_false =\
                        calculate_entropy(no_examples_labels)
            else:
                conditional_entropy_of_X_when_Z_is_false = 0

            probability_of_Z_is_1 =float(len(yes_examples))/len(examples_subset)
            probability_of_Z_is_0 = (1 - probability_of_Z_is_1)

            conditional_entropy_of_X_given_Z = \
                    probability_of_Z_is_1*conditional_entropy_of_X_when_Z_is_true +\
                    probability_of_Z_is_0*conditional_entropy_of_X_when_Z_is_false

            information_gain_table[(f, t)] = \
                    prior_entropy - conditional_entropy_of_X_given_Z

    optimal_feature_threshold_pair = max(information_gain_table.iterkeys(), \
                                     key=lambda ft : information_gain_table[ft])
    (f_opt, t_opt) = optimal_feature_threshold_pair
    splitting_rule = lambda x : x[f_opt] <= t_opt

    return splitting_rule, (f_opt, t_opt)

def make_predictions(decision_tree, feature_vectors):
    X = numpy.array(feature_vectors)
    Y = numpy.arange(X.shape[0],)

    root = decision_tree
    current_node = root

    for (i, x) in enumerate(X):
        current_node = root

        while current_node:
            if not current_node.is_impure():
                Y[i] = current_node.predict()
                break
            elif current_node.decision_question(x):
                current_node = current_node.yes_tree
            else:
                current_node = current_node.no_tree

    return Y

