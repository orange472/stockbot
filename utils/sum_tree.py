from typing import Tuple

import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pointer = 0

    def add(self, priority: float, data: Tuple):
        tree_index = self.pointer + self.capacity - 1
        self.data[self.pointer] = data
        self.update(tree_index, priority)
        self.pointer = (self.pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_index: int, priority: float):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, s: float) -> Tuple[int, float, Tuple]:
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if s <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    s -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total(self) -> float:
        return self.tree[0]
