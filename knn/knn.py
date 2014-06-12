import numpy as np
from collections import Counter
from random import randint
from kdtree import KDTree

# break ties among the majority vote classes uniformly at random
def mode_with_random_tie_breaking(v):
    v = list(v)
    mode = max(set(v), key=v.count)
    frequency_table = Counter(v)
    max_freq = frequency_table[mode]
    same_frequencies = lambda freq : freq[1]==max_freq
    mode_candidates = filter(same_frequencies, frequency_table.items())
    (random_mode, _) =mode_candidates[randint(1,len(mode_candidates))-1]

    return random_mode

def knn_classification(k, dist_func, X_train, Y_train, X_predict):
    (m_examples, n_dimensions) = X_train.shape

    # use kd tree structure for knn searching
    labelled_points = np.append(X_train, Y_train.reshape(m_examples,1),axis=1)
    t = KDTree.build_tree(labelled_points, n_dimensions)

    # store results in the predictions vector
    Y_predict = np.empty(X_predict.shape[0])

    # record the number of points searched for benchmark/comparison purposes
    total_points_searched = 0

    # perform knn search for each test data
    for i, x in enumerate(X_predict):
        (labelled_nearest_neighbors, _, search_space_size) = \
                KDTree.knn_search(t, x, k, n_dimensions, dist_func)

        # nearest neighbor labels are the last column
        nearest_neighbors_labels = np.array(labelled_nearest_neighbors)[:,-1]
        Y_predict[i] = mode_with_random_tie_breaking(nearest_neighbors_labels)
        total_points_searched += search_space_size
        
    return Y_predict

def one_vs_all_knn_classification(k, dist_func, X_train, Y_train, X_predict):
    (m_examples, n_dimensions) = X_train.shape

    # use kd tree structure for knn searching
    train_indices = (np.arange(0, m_examples)).reshape(m_examples, 1)
    indexed_points = np.append(X_train, train_indices, axis=1)
    t = KDTree.build_tree(indexed_points, n_dimensions)

    # store results in the predictions vector
    Y_predict = np.empty(X_predict.shape[0])
 
    # perform knn search for each test data
    for i, x in enumerate(X_predict):
        indexed_nearest_neighbors = \
                KDTree.knn_search(t, x, k, n_dimensions, dist_func)[0]
 
        # http://en.wikipedia.org/wiki/Multiclass_classification
        # use one-vs-all strategy to predict the label
        possible_labels = set(Y_train) # supposing that each class has at \
                                       # least one representative ...
        zero_based_indexed_integer_labels = range(0, len(possible_labels))
        assert possible_labels.issubset(zero_based_indexed_integer_labels), \
               "accept only zero-based indexed, integer labels"

        # the predicted label will be the one from the classifier that gives
        # the most votes, so store the votes in a table
        classifier_votes_tab = {c: 0 for c in zero_based_indexed_integer_labels}

        for c in zero_based_indexed_integer_labels:
            Y_c = np.zeros(m_examples)
            Y_c[Y_train == c] = 1
            
            # neighbor indices are the last column
            nearest_neighbors_indices= np.array(indexed_nearest_neighbors)[:,-1]
            votes = int(sum(Y_c[nearest_neighbors_indices.astype(int)]))
            classifier_votes_tab[c] = votes

        flattened_table = list(Counter(classifier_votes_tab).elements())
        Y_predict[i] = mode_with_random_tie_breaking(flattened_table)

    return Y_predict

def construct_confusion_matrix(prediction_labels,ground_truth_labels,k_classes):
    # check if or suppose that parameters are vectors of equal dimensions
    prediction_labels = np.array(prediction_labels)
    ground_truth_labels = np.array(ground_truth_labels)
    assert prediction_labels.shape == ground_truth_labels.shape

    confusion_matrix = np.zeros((k_classes, k_classes))
    
    for (i, j) in zip(list(prediction_labels), list(ground_truth_labels)):
        confusion_matrix[i, j] += 1

    N = map(lambda clazz : sum(ground_truth_labels==clazz), range(0, k_classes))
    for j in range(0, k_classes):
        if N[j] > 0:
            relative_frequency = confusion_matrix[:, j] / float(N[j])
            confusion_matrix[:, j] = np.around(relative_frequency, decimals=3)

    return confusion_matrix
