function[Covariance, eigenvectors, eigenvalues] = my_singular_value_decomposition(feature_vectors, k, sort)

    % 1. Find the mean vector
    mean_vector = calculate_mean_vector(feature_vectors);
    
    % 2. Center data by subtracting the mean face
    T = feature_vectors - repmat(mean_vector, 1, k);
    
    % 3. Find the covariance matrix
    Covariance = (1/k)*(T*T');
    
    % 4. Calculate the eigenvectors and eigenvalues based on a trick:
    %    let the centered features T of dimensions: n features by k vectors
    %        and covariance matrix S = TT'
    %        and small covariance matrix s = T'T
    %
    %    if u_i is an eigenvector of T'Tu = ?u (small covariance matrix)
    %    then v_i = Tu_i is an eigenvector of TT'v = ?v (covariance matrix)
    L = transpose(T)*T;
    if sort
    [U, D] = eigs(L, size(L,1)); % we want larger eigenvalues first because
                                 % the eigenvalues represent the variances
                                 % of the features, and variances represent
                                 % the contrast.
    else
    [U, D] = eig(L);
    end
                           
    V = (1/k)*(T*U); % find the real eigenvectors based on the small matrix
   
    eigenvectors = normc(V);
    eigenvalues = diag(D);
end