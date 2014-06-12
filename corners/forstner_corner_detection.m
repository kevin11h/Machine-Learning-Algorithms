function [PHIS] = forstner_corner_detection2(image_file, tau)

IMAGE = imresize(imread(image_file), .25);
EIGENVALS = zeros(size(IMAGE));
PHIS = zeros(size(IMAGE));
TAU = tau;

WINDOW_SIZE = 3;
WINDOW = (1/(WINDOW_SIZE ^ 2)) * ones(WINDOW_SIZE);
WINDOW_RADIUS = floor(WINDOW_SIZE/2);

% 1. Compute image gradient
[Ix, Iy] = gradient(double(IMAGE));

% 2. For each point, find the covariance matrix over the neighborhood,
%    and save the coordinates of p along with the smaller eigenvalue
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;

S_Ix2 = conv2(Ix2, WINDOW, 'same');
S_Iy2 = conv2(Iy2, WINDOW, 'same');
S_Ixy = conv2(Ixy, WINDOW, 'same');

for k = 1:numel(EIGENVALS)
    C = [ S_Ix2(k) S_Ixy(k) ;
          S_Ixy(k) S_Iy2(k) ];
      
    tr = trace(C); discriminant = tr^2 - 4*det(C);
    lambda1 = (tr + (sqrt(discriminant))) / 2;
    lambda2 = (tr - (sqrt(discriminant))) / 2;
    
    EIGENVALS(k) = min([lambda1, lambda2]);
    PHIS(k) = 0.5 * atan((pi/180)*(2*C(1,2))/(C(1,1) - C(2,2)));
end

% 3. Threshold the eigenvalues
threshold = TAU * max(EIGENVALS(:));
corner_candidate_points = find(EIGENVALS > threshold);
[candidate_points_x, candidate_points_y] = ...
    ind2sub(size(EIGENVALS), corner_candidate_points);

% 4. Sort the remaining eigenvalues in decreasing order, saving into a list
corner_candidate_points_and_eigenvals = ...
    [candidate_points_x candidate_points_y EIGENVALS(corner_candidate_points)];

ordering = [-3 1 2]; % sort by ascending point vals and decreasing eigenvals
L = sortrows(corner_candidate_points_and_eigenvals, ordering);

% 5. Delete all points appearing further on the list that belong in the
%    neighborhood of p
eigenval_ordered_corner_candidate_points = L(:,[1, 2])';
is_corner = true(1, length(eigenval_ordered_corner_candidate_points));

K = 0;  

for P = eigenval_ordered_corner_candidate_points
    K = K + 1;
    k = 0;
    
    if ~is_corner(K)
        continue;
    end
    
    X = P(1);
    Y = P(2);
    
    for p = eigenval_ordered_corner_candidate_points
        k = k + 1;
        x = p(1);
        y = p(2);

        if k <= K || ~is_corner(k)
            continue;
        elseif (x >= X-WINDOW_RADIUS && x <= X+WINDOW_RADIUS) && ...
           (y >= Y-WINDOW_RADIUS && y <= Y+WINDOW_RADIUS)
            is_corner(k) = false;
        end
    end
end

corners = eigenval_ordered_corner_candidate_points(:,find(is_corner));
corners_x = corners(1,:);
corners_y = corners(2,:);

figure;
imshow(IMAGE);
hold on;
plot(corners_y, corners_x, 'r.');
hold off;

end