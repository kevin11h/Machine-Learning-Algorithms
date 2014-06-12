import numpy

class Perceptron:

    TYPES = ['basic', 'voted', 'averaged']

    def __init__(self, classification_type):
        if classification_type in Perceptron.TYPES:
            self.classification_type = classification_type
        else:
            raise ValueError('Only types from ' + \
                             set(Perceptron.TYPES) + \
                             ' are allowed')

    def train(self, X, Y, d_features, epochs=1):
        w = numpy.zeros(d_features)
        ws = []
        m = 1# classifier id
        c = 1# count the number of test examples the current hyperplane survived
        cs = []
         
        for _ in range(epochs):
            for (t, x) in enumerate(numpy.array(X)):
                if Y[t] *(numpy.dot(w, x)) <= 0: # mistake has occurred
                    ws.append(w)
                    cs.append(c)
                    w = numpy.array(w + Y[t]*x)
                    m += 1
                    c = 1
                else:
                    c += 1

        ws.append(w)
        cs.append(c)
        n_updates = (m-1)
        
        def find_closest_distance_to(linear_separator):
            return min(map(lambda x : abs(numpy.dot(linear_separator, x)), X)) /\
                        float(numpy.linalg.norm(linear_separator))
                 
        if self.classification_type == 'basic':
            self.hyperplane = ws[n_updates]
            self.margin = find_closest_distance_to(self.hyperplane)
        elif self.classification_type == 'voted':
            self.hyperplanes = ws
            self.votes = cs
            return True
        elif self.classification_type == 'averaged':
            self.averaged_hyperplane = numpy.dot(cs, ws)
            self.margin = find_closest_distance_to(self.averaged_hyperplane)

        return n_updates<=(max(map(numpy.linalg.norm, X))/float(self.margin))**2

    def predict(self, x):
        if self.classification_type == 'basic':
            return numpy.sign(numpy.dot(self.hyperplane, x))
        elif self.classification_type == 'voted':
            signs = map(lambda w : numpy.sign(numpy.dot(w, x)), self.hyperplanes)
            voted_signs = numpy.dot(self.votes, signs)
            majority_vote = numpy.sign(voted_signs)
            return majority_vote
        elif self.classification_type == 'averaged':
            return numpy.sign(numpy.dot(self.averaged_hyperplane, x))
 
