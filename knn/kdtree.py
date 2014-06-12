from bpq import BoundedPriorityQueue

class KDNode:
    def __init__(self, p, l, r, s):
        self.point = p
        self.left = l
        self.right = r
        self.size = s

class KDTree:
    @staticmethod
    def build_tree(points, k_dimensions):
        def make_kdnode(points, depth):
            if len(points) == 0:
                return None
            elif len(points) == 1:
                return KDNode(points[0], None, None, 1)
            else:
                axis = depth % k_dimensions
                projection = sorted(points, key=lambda p : p[axis])
                median_idx = (len(projection)-1)/2
                splitpoint = projection[median_idx]
                left  = make_kdnode(projection[:median_idx], depth+1)
                right = make_kdnode(projection[median_idx+1:],depth+1)
                size = 1+(left.size if left else 0)+(right.size if right else 0)

                return KDNode(splitpoint, left, right, size)

        return make_kdnode(points, 0)

    @staticmethod
    def knn_search(root, query_point, k_neighbors, k_dimensions, distance_func):
        bpq = BoundedPriorityQueue(k_neighbors)
        visited_tree_paths = set([])
        root_tree_path = ''
        num_points_visited = {'count': 1}
 
        def knn_search_rec(current_kdnode, parent_kdnode, depth, tree_path):
            if current_kdnode is None:
                return

            # determine in which half of the hyperplane-splitted region
            # in which the query point lies
            split_point = current_kdnode.point[:k_dimensions]
            axis = depth % k_dimensions
            splitting_hyperplane = split_point[axis]

            if query_point[axis] < splitting_hyperplane:
                knn_search_rec(current_kdnode.left, current_kdnode,\
                               depth + 1, tree_path + '0')
            else:
                knn_search_rec(current_kdnode.right, current_kdnode,\
                               depth + 1, tree_path + '1')

            # keep track of the K nearest distances and corresponding points
            d = distance_func(split_point, query_point)
            bpq.maxheap_insert(d, current_kdnode.point)

            # if the hypersphere radius intersects the hyperplane of the
            # current axis, then it is possible that a nearer neighbor lies
            # on the other half of the hyperplane
            knn_distances = bpq.get_priorities()
            hypersphere_radius = max(knn_distances)
            hyperplane_distance = abs(splitting_hyperplane - query_point[axis])
            hypersphere_will_cross_hyperplane =\
                    (hypersphere_radius > hyperplane_distance)

            if (current_kdnode is root) or (tree_path in visited_tree_paths):
                return
            else:
                # mark kdnode as visited_tree_paths
                visited_tree_paths.add(tree_path)
                num_points_visited['count'] += 1
    
            if current_kdnode is parent_kdnode.left:
                subtree_on_other_side_of_hyperplane = parent_kdnode.right
                other_tree_path = tree_path[:-1] + '1'
            else:
                subtree_on_other_side_of_hyperplane = parent_kdnode.left
                other_tree_path = tree_path[:-1] + '0'

            if not bpq.is_full() or hypersphere_will_cross_hyperplane:
                if other_tree_path not in visited_tree_paths:
                    knn_search_rec(subtree_on_other_side_of_hyperplane, \
                                   parent_kdnode, depth, other_tree_path)
            else:
                visited_tree_paths.add(other_tree_path)

        knn_search_rec(root, parent_kdnode=None, depth=0, tree_path=root_tree_path)
        nearest_points = bpq.get_elements()
        distances = bpq.get_priorities()

        return nearest_points, distances, num_points_visited['count']
