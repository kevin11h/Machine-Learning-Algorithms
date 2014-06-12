import numpy

def boosting(X, Y, learners, n_epochs):
    X = numpy.array(X)
    Y = numpy.array(Y)
    learners = numpy.array(learners)
    n_examples = len(X) ; assert (n_examples > 0)
    n_learners = len(learners)
    uniform_weighting = 1/float(n_examples)
    distribution_weights = uniform_weighting * numpy.ones(n_examples)
    learners_errors = numpy.empty(n_learners)
    alphas = numpy.empty(n_epochs)
    opt_classifier_indices_list = numpy.empty(n_epochs)
    classifiers = list(numpy.empty(n_epochs))

    def calculate_ensemble_classifier(t):
        return lambda x: numpy.sign(numpy.dot(alphas[:t],map(lambda h: h(x),\
                learners[map(int,opt_classifier_indices_list[:t])])))

    def err_t(weights, h_t):
        y_pre = map(h_t, X) 
        return numpy.dot(weights, (y_pre != Y))
        
    for t in range(n_epochs):
        for i in range(n_learners):
            h_t_i = learners[i]
            e_t = err_t(distribution_weights, h_t_i)
            learners_errors[i] = e_t                

        min_error_weak_learner_index = numpy.argmin(learners_errors)
        h_t_opt = learners[min_error_weak_learner_index]
        e_t_opt = err_t(distribution_weights, h_t_opt)
        alpha_t = 0.5*numpy.log(float(1-e_t_opt)/e_t_opt)
        matching_classifications = Y * map(h_t_opt, X)
        exp = numpy.exp(-alpha_t*(matching_classifications))
        numerator = distribution_weights * exp
        z = sum(numerator)
        current_weak_learner_weighting = numerator / z
        distribution_weights = numpy.array(current_weak_learner_weighting)

        alphas[t] = alpha_t
        opt_classifier_indices_list[t] = min_error_weak_learner_index
        classifiers[t] = calculate_ensemble_classifier(t)
                
    return opt_classifier_indices_list, classifiers
