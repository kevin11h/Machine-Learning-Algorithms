function [J, Clusters] = iterative_kmeans(feature_vectors, K,...
                                          DELTA_THRESHOLD, MAX_ITERATIONS)
    J = 0;
    delta_J = DELTA_THRESHOLD + 1;
    n_obsv = length(feature_vectors);
    
    % 1. Set iteration count to 1
    ic = 1
    
    % 2. Choose randomly an initial set of K means
    %    (in the version, initially assign K feature vectors at random)
    Means = initial_mean_set(feature_vectors, K, n_obsv);
    
    while ~(ic == MAX_ITERATIONS || abs(delta_J) < DELTA_THRESHOLD)
        % 3. Assign each feature vector to Clusters w/ the nearest centroid
        Clusters = cell(K, 1);
        
        for i = 1:n_obsv
            x_i = feature_vectors(i, 1, :);
            distances = [K 1];

            for k = 1:K
                mean_k = Means(k,:);
                distances(k) = euclidian_distance(x_i(:), mean_k(:));
            end

            [~, argmin_K] = min(distances);
            Clusters{argmin_K} = [Clusters{argmin_K} x_i(:)];
        end
        
        prev_J = J;
        J = calculate_within_clusters_sum_square_error(K, Clusters, Means)
        delta_J = J - prev_J;

        % 4. Update new set of means and increment iteration counter
        ic = ic + 1
        Means = create_mean_set(Clusters, K);
    
        % 5. Repeat steps 3 and 4 until either:
        %    - iteration count is reached
        %    - threshold on the change in J is met
        %    - no more changes in the Clusters
    end
end

function [Initial_Mean_Set] = initial_mean_set(feature_vectors, K, n_obsv)
    Initial_Mean_Set = zeros([K 1 3]);
    used_indices = zeros([K 1]);
    
    for k = 1:K
       random_vector_index = randi(n_obsv);
       already_used = ...
           ~isempty(find(ismember(used_indices, random_vector_index), 1));
       
       if already_used
           k = k - 1;
       else
           Initial_Mean_Set(k,:)=feature_vectors(random_vector_index, 1,:);
           used_indices(k) = random_vector_index;
       end
    end
end

function [Mean_Set] = create_mean_set(Clusters, K)
    Mean_Set = zeros([K 1 3]);

    for k = 1:K
        Mean_Set(k,:) = mean(Clusters{k}, 2);
    end
end

function [d] = euclidian_distance(a, b)
    assert( ~isrow(a) || ~iscolumn(a) || ~isrow(b) || ~iscolumn(b) );
    d = sqrt(sum((b - a) .^2));
end

function [J] = calculate_within_clusters_sum_square_error(K, Clusters, Means)
    J = 0;
    
    for k = 1:K
        mean_k = Means(k,:);
        
        for x = Clusters{k}
            J = J + norm(x - mean_k(:))^2;
        end
    end
end