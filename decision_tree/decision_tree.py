class DTNode:

    def __init__(self,subset, labels):
        self.subset = set(subset)
        self.labels = set(labels)

    def is_impure(self):
        return len(self.labels) != 1

    def predict(self):
        if not self.is_impure():
            return list(self.labels)[0]
        else:
            return None

