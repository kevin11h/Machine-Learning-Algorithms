import heapq

class BoundedPriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.maxheap = [] # we are working with shortest distances, therefore
                          # a maxheap's ordering property is used here
                          #
                          # data is mapped by using integer keys for priorities:
                          # (distance, feature_vector) |-> (priority, element)

    def is_full(self):
        return (len(self.maxheap) == self.max_size)

    def maxheap_insert(self, priority, e):
        if not self.is_full():
            heapq.heappush(self.maxheap, (-priority, list(e)))
        else:
            (greatest_priority, greatest_element) = self.maxheap[0]
            has_higher_priority = (-priority > greatest_priority)

            if has_higher_priority:
                heapq.heappushpop(self.maxheap, (-priority, list(e)))

    def get_priorities(self):
        return map(lambda pr_e : -pr_e[0], self.maxheap)

    def get_elements(self):
        return map(lambda pr_e : pr_e[1], self.maxheap)
